"""Permission gating for mutating tools."""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from enum import Enum

from mcc.db import sessions as db
from mcc.tools.base import Tool
from mcc.tools.shell import danger_reason


class Decision(str, Enum):
    ALLOW_ONCE = "once"
    ALLOW_TOOL = "tool"      # always allow this tool
    ALLOW_PATTERN = "pattern"  # always allow matching invocations
    DENY = "deny"


@dataclass
class Verdict:
    allowed: bool
    approved_by: str          # auto|rule|user|yolo
    feedback: str = ""        # returned to the model when denied


def subject(tool: Tool, args: dict) -> str:
    """The thing a rule pattern matches against: a path or a command."""
    if tool.name == "run_bash":
        return str(args.get("command", ""))
    return str(args.get("path", ""))


class PermissionManager:
    """Decides whether a tool call may run.

    Read-only tools always pass. Mutating tools consult saved rules, then
    fall back to asking the user. Commands matching a dangerous pattern
    always ask, even when a rule would otherwise allow them.
    """

    def __init__(self, session_id: str | None, yolo: bool = False, asker=None):
        self.session_id = session_id
        self.yolo = yolo
        self.asker = asker  # callable(tool, args, danger) -> Decision
        self._rules = db.load_rules(session_id) if session_id else []

    def reload(self) -> None:
        if self.session_id:
            self._rules = db.load_rules(self.session_id)

    def _matching_rule(self, tool: Tool, args: dict) -> str | None:
        subj = subject(tool, args)
        for rule in self._rules:
            if rule["tool_name"] != tool.name:
                continue
            pattern = rule["pattern"]
            if pattern == "*" or fnmatch.fnmatch(subj, pattern):
                return rule["decision"]
        return None

    def check(self, tool: Tool, args: dict) -> Verdict:
        if not tool.mutating:
            return Verdict(True, "auto")

        danger = danger_reason(args.get("command", "")) if tool.name == "run_bash" else None

        if self.yolo and not danger:
            return Verdict(True, "yolo")

        if not danger:
            rule = self._matching_rule(tool, args)
            if rule == "allow":
                return Verdict(True, "rule")
            if rule == "deny":
                return Verdict(
                    False, "rule",
                    f"Denied by a standing rule for {tool.name}. Try another approach.",
                )

        if self.asker is None:
            return Verdict(False, "auto", "No approval channel available.")

        decision = self.asker(tool, args, danger)

        if decision is Decision.DENY:
            return Verdict(
                False, "user",
                f"The user denied this {tool.name} call. Do not retry it -- "
                f"ask what they would prefer, or take a different approach.",
            )
        if decision is Decision.ALLOW_TOOL:
            db.save_rule(tool.name, "*", "allow", self.session_id)
            self.reload()
        elif decision is Decision.ALLOW_PATTERN:
            db.save_rule(tool.name, _generalise(tool, args), "allow", self.session_id)
            self.reload()
        return Verdict(True, "user")


def _generalise(tool: Tool, args: dict) -> str:
    """Turn a concrete invocation into a reusable glob."""
    subj = subject(tool, args)
    if tool.name == "run_bash":
        head = subj.strip().split()
        if not head:
            return "*"
        # `git status -sb` -> `git status *`
        prefix = " ".join(head[:2]) if len(head) > 1 and not head[1].startswith("-") else head[0]
        return f"{prefix}*"
    if "." in subj.rsplit("/", 1)[-1]:
        return f"*.{subj.rsplit('.', 1)[-1]}"
    return subj or "*"
