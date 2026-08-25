"""Filesystem tools."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from pydantic import BaseModel, Field

from mcc.tools.base import Tool, ToolContext, ToolResult

MAX_READ_BYTES = 200_000
DEFAULT_LIMIT = 800


def syntax_error(path: Path, text: str) -> str | None:
    """Return a description of a syntax error in `text`, or None if it parses.

    Only covers formats we can check cheaply and exactly. Silence here means
    "not checkable", not "valid".
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".py":
            ast.parse(text)
        elif suffix == ".json":
            json.loads(text)
    except SyntaxError as e:
        return f"line {e.lineno}: {e.msg}"
    except json.JSONDecodeError as e:
        return f"line {e.lineno}: {e.msg}"
    return None


def rejects_edit(path: Path, before: str | None, after: str) -> str | None:
    """Guard against an edit that breaks a file which previously parsed.

    Small models get indentation wrong constantly -- the string replacement
    succeeds while the resulting file no longer compiles. Blocking the write
    and handing back the syntax error is far cheaper than letting the damage
    land and hoping the model notices. An already-broken file is never
    blocked, so the model can still repair it.
    """
    err = syntax_error(path, after)
    if err is None:
        return None
    if before is not None and syntax_error(path, before) is not None:
        return None  # it was already broken; allow the attempt
    return err


class ReadArgs(BaseModel):
    path: str = Field(description="File path, absolute or relative to the workspace.")
    offset: int = Field(0, ge=0, description="0-indexed line to start from.")
    limit: int = Field(DEFAULT_LIMIT, ge=1, le=5000, description="Max lines to read.")


class ReadFile(Tool):
    name = "read_file"
    description = (
        "Read a text file. Returns lines prefixed with line numbers. "
        "Use offset/limit for large files."
    )
    args_model = ReadArgs

    def run(self, args: ReadArgs, ctx: ToolContext) -> ToolResult:
        p = ctx.resolve(args.path)
        if not p.exists():
            return ToolResult(f"File not found: {args.path}", is_error=True)
        if p.is_dir():
            return ToolResult(f"{args.path} is a directory; use list_dir.", is_error=True)
        if p.stat().st_size > MAX_READ_BYTES:
            return ToolResult(
                f"File is {p.stat().st_size} bytes, too large to read whole. "
                f"Use offset/limit or grep.",
                is_error=True,
            )
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as e:
            return ToolResult(f"Could not read {args.path}: {e}", is_error=True)

        window = lines[args.offset : args.offset + args.limit]
        if not window:
            return ToolResult(
                f"(offset {args.offset} is past end of file; {len(lines)} lines total)"
            )
        body = "\n".join(
            f"{i + args.offset + 1:>5}\t{ln}" for i, ln in enumerate(window)
        )
        more = ""
        if args.offset + len(window) < len(lines):
            more = f"\n... ({len(lines) - args.offset - len(window)} more lines)"
        return ToolResult(body + more, summary=f"{len(window)} lines")


class WriteArgs(BaseModel):
    path: str = Field(description="File path to write.")
    content: str = Field(description="Full file content. Overwrites any existing file.")


class WriteFile(Tool):
    name = "write_file"
    description = (
        "Write a complete file, creating parent directories as needed. "
        "Overwrites existing content -- prefer edit_file for changes to a file "
        "that already exists."
    )
    args_model = WriteArgs
    mutating = True

    def run(self, args: WriteArgs, ctx: ToolContext) -> ToolResult:
        p = ctx.resolve(args.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        before = p.read_text(encoding="utf-8", errors="replace") if existed else None

        broke = rejects_edit(p, before, args.content)
        if broke is not None:
            return ToolResult(
                f"Write rejected: the content is not valid ({broke}). "
                f"{args.path} is unchanged. Fix the syntax and try again.",
                is_error=True,
            )

        p.write_text(args.content, encoding="utf-8")
        n = len(args.content.splitlines())
        verb = "Updated" if existed else "Created"
        return ToolResult(f"{verb} {args.path} ({n} lines)", summary=f"{verb.lower()} {n} lines")


class EditArgs(BaseModel):
    path: str = Field(description="File to edit.")
    old_string: str = Field(
        description=(
            "Exact text to replace, including whitespace and indentation. "
            "Must appear exactly once in the file."
        )
    )
    new_string: str = Field(description="Replacement text.")


class EditFile(Tool):
    name = "edit_file"
    description = (
        "Replace an exact string in a file. old_string must match the file "
        "byte-for-byte and appear exactly once -- include surrounding lines "
        "to make it unique. This is the preferred way to change a file."
    )
    args_model = EditArgs
    mutating = True

    def run(self, args: EditArgs, ctx: ToolContext) -> ToolResult:
        p = ctx.resolve(args.path)
        if not p.exists():
            return ToolResult(f"File not found: {args.path}", is_error=True)
        text = p.read_text(encoding="utf-8")

        count = text.count(args.old_string)
        if count == 0:
            return ToolResult(
                f"old_string not found in {args.path}. It must match exactly, "
                f"including indentation. Read the file again and copy the text verbatim.",
                is_error=True,
            )
        if count > 1:
            return ToolResult(
                f"old_string appears {count} times in {args.path}; it must be unique. "
                f"Include more surrounding context to disambiguate.",
                is_error=True,
            )
        if args.old_string == args.new_string:
            return ToolResult("old_string and new_string are identical.", is_error=True)

        updated = text.replace(args.old_string, args.new_string, 1)
        line_no = text[: text.index(args.old_string)].count("\n") + 1

        broke = rejects_edit(p, text, updated)
        if broke is not None:
            return ToolResult(
                f"Edit rejected: it would make {args.path} invalid ({broke}). "
                f"The file is unchanged. Check the indentation of new_string -- "
                f"it must line up with the surrounding code.",
                is_error=True,
            )

        p.write_text(updated, encoding="utf-8")
        return ToolResult(
            f"Edited {args.path} at line {line_no}.", summary=f"line {line_no}"
        )


class ListArgs(BaseModel):
    path: str = Field(".", description="Directory to list, relative to the workspace.")


SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
             ".pytest_cache", "dist", "build", ".next", "target"}


class ListDir(Tool):
    name = "list_dir"
    description = "List files and directories at a path. Does not recurse."
    args_model = ListArgs

    def run(self, args: ListArgs, ctx: ToolContext) -> ToolResult:
        p = ctx.resolve(args.path)
        if not p.is_dir():
            return ToolResult(f"Not a directory: {args.path}", is_error=True)
        rows = []
        for child in sorted(p.iterdir(), key=lambda c: (c.is_file(), c.name)):
            if child.name in SKIP_DIRS or child.name.startswith("."):
                continue
            rows.append(f"{child.name}/" if child.is_dir() else child.name)
        if not rows:
            return ToolResult(f"(empty: {args.path})")
        return ToolResult("\n".join(rows), summary=f"{len(rows)} entries")
