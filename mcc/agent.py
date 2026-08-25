"""The agent loop."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from mcc.config import Config
from mcc.context import (
    SUMMARY_PROMPT, message_tokens, needs_compaction, split_for_compaction,
    total_tokens,
)
from mcc.db import sessions as db
from mcc.db.index import CodeIndex
from mcc.llm.ollama import Chunk, OllamaClient, ToolCall
from mcc.prompts import system_prompt
from mcc.tools.base import ToolContext, ToolResult, format_validation_error
from mcc.tools.registry import BY_NAME, schemas


class Events:
    """UI callbacks. Defaults are silent so the agent is testable headless."""

    def assistant_delta(self, text: str) -> None: ...
    def thinking_delta(self, text: str) -> None: ...
    def assistant_done(self, text: str) -> None: ...
    def tool_start(self, name: str, args: dict) -> None: ...
    def tool_end(self, name: str, result: ToolResult, ms: int) -> None: ...
    def tool_denied(self, name: str, args: dict) -> None: ...
    def notice(self, text: str) -> None: ...


class Agent:
    def __init__(
        self,
        cfg: Config,
        session_id: str,
        permissions,
        events: Events | None = None,
    ):
        self.cfg = cfg
        self.session_id = session_id
        self.permissions = permissions
        self.events = events or Events()
        self.llm = OllamaClient(cfg)
        self.index: CodeIndex | None = CodeIndex(cfg.cwd, self.llm.embed)
        self.ctx = ToolContext(cwd=cfg.cwd, session_id=session_id, agent=self)
        self.interrupted = False
        self.last_usage = {"prompt": 0, "completion": 0}

    # --------------------------------------------------------------- history
    def _history(self) -> list[dict[str, Any]]:
        """Rebuild the wire-format message list from Postgres."""
        out: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt(str(self.cfg.cwd))}
        ]
        for row in db.load_messages(self.session_id):
            role = row["role"]
            if role == "summary":
                out.append({
                    "role": "user",
                    "content": f"[Summary of earlier conversation]\n{row['content']}",
                })
            elif role == "tool":
                out.append({
                    "role": "tool",
                    "content": row["content"],
                    "tool_name": row["tool_name"] or "",
                })
            elif role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": row["content"]}
                if row["tool_calls"]:
                    msg["tool_calls"] = row["tool_calls"]
                out.append(msg)
            else:
                out.append({"role": role, "content": row["content"]})
        return out

    # -------------------------------------------------------------- the loop
    def run_turn(self, user_input: str) -> str:
        self.interrupted = False
        db.add_message(self.session_id, "user", user_input)
        db.set_title(self.session_id, user_input.strip().splitlines()[0][:120])

        final_text = ""
        arg_failures: dict[str, int] = {}

        for step in range(self.cfg.max_steps):
            self.maybe_compact()
            messages = self._history()

            final: Chunk | None = None
            try:
                for chunk in self.llm.stream_chat(messages, tools=schemas()):
                    if chunk.done:
                        final = chunk
                        break
                    if chunk.thinking:
                        self.events.thinking_delta(chunk.thinking)
                    if chunk.text:
                        self.events.assistant_delta(chunk.text)
            except KeyboardInterrupt:
                self.interrupted = True
                self.events.notice("interrupted")
                db.add_message(self.session_id, "assistant", "[interrupted by user]")
                return ""

            if final is None:
                break

            self.last_usage = {
                "prompt": final.prompt_eval_count,
                "completion": final.eval_count,
            }
            calls = final.tool_calls or []
            self.events.assistant_done(final.text)

            db.add_message(
                self.session_id, "assistant", final.text,
                tool_calls=[{"function": {"name": c.name, "arguments": c.args}}
                            for c in calls] or None,
                token_count=final.eval_count,
            )

            if not calls:
                final_text = final.text
                break

            for call in calls:
                keep_going = self._execute(call, arg_failures)
                if not keep_going:
                    return ""
        else:
            self.events.notice(
                f"stopped after {self.cfg.max_steps} steps without finishing"
            )

        return final_text

    # --------------------------------------------------------- tool dispatch
    def _execute(self, call: ToolCall, arg_failures: dict[str, int]) -> bool:
        """Run one tool call. Returns False if the turn should abort."""
        tool = BY_NAME.get(call.name)
        if tool is None:
            known = ", ".join(BY_NAME)
            self._tool_message(
                call.name,
                f"No such tool `{call.name}`. Available tools: {known}.",
                is_error=True,
            )
            return True

        # Validate args; hand failures back to the model as feedback.
        try:
            parsed = tool.validate(call.args)
        except ValidationError as e:
            arg_failures[call.name] = arg_failures.get(call.name, 0) + 1
            if arg_failures[call.name] > self.cfg.max_arg_retries:
                self._tool_message(
                    call.name,
                    f"`{call.name}` was called with invalid arguments "
                    f"{arg_failures[call.name]} times. Stop using it and try "
                    f"another approach.",
                    is_error=True,
                )
                return True
            self._tool_message(call.name, format_validation_error(call.name, e),
                               is_error=True, args=call.args)
            return True

        verdict = self.permissions.check(tool, call.args)
        if not verdict.allowed:
            self.events.tool_denied(call.name, call.args)
            self._tool_message(call.name, verdict.feedback, is_error=True,
                               args=call.args, approved_by=verdict.approved_by)
            return True

        self.events.tool_start(call.name, call.args)
        started = time.monotonic()
        try:
            result = tool.run(parsed, self.ctx)
        except KeyboardInterrupt:
            self.interrupted = True
            self.events.notice("interrupted")
            self._tool_message(call.name, "[interrupted by user]", is_error=True)
            return False
        except Exception as e:  # a tool crash must not kill the session
            result = ToolResult(f"{type(e).__name__}: {e}", is_error=True)
        ms = int((time.monotonic() - started) * 1000)

        self.events.tool_end(call.name, result, ms)
        self._tool_message(
            call.name, result.content, is_error=result.is_error,
            args=call.args, duration_ms=ms, approved_by=verdict.approved_by,
        )
        return True

    def _tool_message(
        self,
        name: str,
        content: str,
        is_error: bool = False,
        args: dict | None = None,
        duration_ms: int = 0,
        approved_by: str = "auto",
    ) -> None:
        msg_id = db.add_message(
            self.session_id, "tool", content, tool_name=name,
            token_count=message_tokens({"content": content}),
        )
        db.record_tool_call(
            self.session_id, name, args or {}, content, is_error,
            duration_ms, approved_by, message_id=msg_id,
        )

    # ------------------------------------------------------------ compaction
    def maybe_compact(self, force: bool = False) -> bool:
        messages = self._history()
        if not force and not needs_compaction(
            messages, self.cfg.num_ctx, self.cfg.compact_at
        ):
            return False

        rows = db.load_messages(self.session_id)
        wire = [{"role": r["role"], "content": r["content"]} for r in rows]
        fold_wire, _ = split_for_compaction(wire)
        if not fold_wire:
            return False
        fold_rows = rows[: len(fold_wire)]

        self.events.notice(
            f"compacting {len(fold_rows)} messages "
            f"(~{total_tokens(messages)} tokens)"
        )

        transcript = "\n\n".join(
            f"[{r['role']}] {(r['content'] or '')[:2000]}" for r in fold_rows
        )
        summary = ""
        try:
            for chunk in self.llm.stream_chat([
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": transcript},
            ]):
                if chunk.done:
                    summary = chunk.text
        except Exception as e:
            self.events.notice(f"compaction failed ({e}); continuing uncompacted")
            return False

        if not summary.strip():
            return False

        # Originals are kept, just flagged, so nothing is lost.
        db.mark_superseded([r["id"] for r in fold_rows])
        db.add_message(self.session_id, "summary", summary.strip())
        return True

    # ----------------------------------------------------------------- index
    def build_index(self, progress: Callable[[str], None] | None = None) -> dict[str, int]:
        assert self.index is not None
        return self.index.build(progress)
