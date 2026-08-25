"""Terminal rendering."""
from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from mcc.agent import Events
from mcc.permissions import Decision
from mcc.tools.base import Tool, ToolResult

console = Console()

DIM = "grey50"
ACCENT = "cyan"


def _preview(args: dict[str, Any], limit: int = 68) -> str:
    """One-line argument preview for a tool card."""
    if "command" in args:
        body = str(args["command"])
    elif "path" in args:
        body = str(args["path"])
        if "old_string" in args:
            body += "  (edit)"
    elif "pattern" in args:
        body = str(args["pattern"])
    elif "query" in args:
        body = str(args["query"])
    else:
        body = json.dumps(args)[:limit]
    body = " ".join(body.split())
    return body if len(body) <= limit else body[: limit - 1] + "…"


class ConsoleEvents(Events):
    """Streams assistant text live and renders tool calls as compact cards."""

    def __init__(self, show_thinking: bool = False):
        self.show_thinking = show_thinking
        self._streaming = False
        self._thinking_open = False

    # ------------------------------------------------------------ assistant
    def assistant_delta(self, text: str) -> None:
        if self._thinking_open:
            console.print()
            self._thinking_open = False
        if not self._streaming:
            self._streaming = True
        console.print(text, end="", markup=False, highlight=False)

    def thinking_delta(self, text: str) -> None:
        if not self.show_thinking:
            return
        if not self._thinking_open:
            console.print(Text("thinking ", style=DIM), end="")
            self._thinking_open = True
        console.print(Text(text, style=DIM), end="", markup=False, highlight=False)

    def assistant_done(self, text: str) -> None:
        if self._streaming or self._thinking_open:
            console.print()
        self._streaming = False
        self._thinking_open = False

    # ---------------------------------------------------------------- tools
    def tool_start(self, name: str, args: dict) -> None:
        console.print(
            Text("  ● ", style=ACCENT)
            + Text(name, style="bold")
            + Text(f"  {_preview(args)}", style=DIM)
        )

    def tool_end(self, name: str, result: ToolResult, ms: int) -> None:
        mark, style = ("✗", "red") if result.is_error else ("✓", "green")
        detail = result.summary or ("error" if result.is_error else "ok")
        console.print(
            Text(f"    {mark} ", style=style)
            + Text(f"{detail}  {ms}ms", style=DIM)
        )
        if result.is_error:
            first = result.content.strip().splitlines()[:2]
            for line in first:
                console.print(Text(f"      {line[:100]}", style="red"))

    def tool_denied(self, name: str, args: dict) -> None:
        console.print(Text(f"    ✗ denied  {name}", style="yellow"))

    def notice(self, text: str) -> None:
        console.print(Text(f"  · {text}", style=DIM))


# ------------------------------------------------------------------ prompts
def ask_permission(tool: Tool, args: dict, danger: str | None) -> Decision:
    """Blocking approval prompt for a mutating tool call."""
    body = Text()
    body.append(f"{tool.name}\n", style="bold")
    if tool.name == "run_bash":
        body.append(str(args.get("command", "")), style="white")
    else:
        for key in ("path", "old_string", "new_string", "content"):
            if key in args:
                val = str(args[key])
                shown = val if len(val) < 400 else val[:400] + "\n…"
                body.append(f"{key}: ", style=DIM)
                body.append(f"{shown}\n")

    title = "Approve tool call"
    border = ACCENT
    if danger:
        title = f"⚠  Dangerous: {danger}"
        border = "red"

    console.print(Panel(body, title=title, border_style=border, padding=(0, 1)))
    console.print(
        Text("  [y] once   [a] always this tool   [p] always like this   "
             "[n] deny", style=DIM)
    )

    while True:
        try:
            choice = console.input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return Decision.DENY
        if choice in ("y", "yes", ""):
            return Decision.ALLOW_ONCE
        if choice == "a":
            return Decision.ALLOW_TOOL
        if choice == "p":
            return Decision.ALLOW_PATTERN
        if choice in ("n", "no", "d"):
            return Decision.DENY
        console.print(Text("  answer y / a / p / n", style="yellow"))


def banner(cfg, session_id: str, resumed: bool) -> None:
    console.print()
    console.print(
        Text("mini-claude-code", style=f"bold {ACCENT}")
        + Text(f"  {cfg.model}  ctx {cfg.num_ctx}", style=DIM)
    )
    console.print(Text(f"  {cfg.cwd}", style=DIM))
    state = "resumed" if resumed else "new"
    console.print(Text(f"  session {session_id[:8]} ({state})", style=DIM))
    if cfg.yolo:
        console.print(Text("  yolo mode: permission prompts disabled", style="yellow"))
    console.print(Text("  /help for commands", style=DIM))
    console.print()


def print_markdown(text: str) -> None:
    console.print(Markdown(text))
