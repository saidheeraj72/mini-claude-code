"""Shell execution, with the guards a 4B model makes mandatory."""
from __future__ import annotations

import re
import subprocess

from pydantic import BaseModel, Field

from mcc.tools.base import Tool, ToolContext, ToolResult

MAX_OUTPUT = 30_000

# Patterns that always prompt, even when the tool is otherwise allowlisted.
DANGEROUS = [
    (re.compile(r"\brm\s+(-[a-zA-Z]*\s+)*-?[a-zA-Z]*[rf]", re.I), "recursive/forced delete"),
    (re.compile(r"\bgit\s+push\b.*--force|\bgit\s+push\s+-f\b", re.I), "force push"),
    (re.compile(r"\bgit\s+reset\s+--hard\b", re.I), "hard reset"),
    (re.compile(r"\bgit\s+clean\b.*-[a-zA-Z]*f", re.I), "git clean -f"),
    (re.compile(r":\s*\(\s*\)\s*\{.*\}\s*;\s*:", re.S), "fork bomb"),
    (re.compile(r"\bmkfs\b|\bdd\s+if=.*of=/dev/", re.I), "disk write"),
    (re.compile(r"\bchmod\s+-R\s+777\b", re.I), "world-writable chmod"),
    (re.compile(r"\bsudo\b", re.I), "sudo"),
    (re.compile(r"\bcurl\b.*\|\s*(ba)?sh|\bwget\b.*\|\s*(ba)?sh", re.I), "pipe to shell"),
    (re.compile(r">\s*/(?!tmp|dev/null)[a-z]+", re.I), "write outside workspace"),
]


def danger_reason(command: str) -> str | None:
    for pattern, label in DANGEROUS:
        if pattern.search(command):
            return label
    return None


class BashArgs(BaseModel):
    command: str = Field(description="Shell command to run in the workspace directory.")
    timeout: int = Field(
        60, ge=1, le=600, description="Timeout in seconds."
    )


class RunBash(Tool):
    name = "run_bash"
    description = (
        "Run a shell command in the workspace. Use for builds, tests, git, and "
        "package managers. Do NOT use it to read or edit files -- the file tools "
        "are better. Never run interactive commands; they will hang."
    )
    args_model = BashArgs
    mutating = True

    def run(self, args: BashArgs, ctx: ToolContext) -> ToolResult:
        try:
            proc = subprocess.run(
                args.command,
                shell=True,
                cwd=str(ctx.cwd),
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                f"Command timed out after {args.timeout}s. If it is long-running or "
                f"interactive, that is why -- do not retry it as-is.",
                is_error=True,
            )
        except OSError as e:
            return ToolResult(f"Could not run command: {e}", is_error=True)

        out = (proc.stdout or "") + (
            ("\n[stderr]\n" + proc.stderr) if proc.stderr.strip() else ""
        )
        out = out.strip()
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + f"\n... (truncated, {len(out)} bytes total)"
        if not out:
            out = "(no output)"

        failed = proc.returncode != 0
        if failed:
            out = f"exit code {proc.returncode}\n{out}"
        return ToolResult(out, is_error=failed, summary=f"exit {proc.returncode}")
