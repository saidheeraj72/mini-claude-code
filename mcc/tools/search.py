"""Search tools: literal grep, glob, and pgvector-backed semantic search."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from mcc.tools.base import Tool, ToolContext, ToolResult
from mcc.tools.fs import SKIP_DIRS

MAX_MATCHES = 60


class GrepArgs(BaseModel):
    pattern: str = Field(description="Regular expression to search for.")
    path: str = Field(".", description="Directory or file to search in.")
    glob: str = Field(
        "", description="Optional filename filter, e.g. '*.py'. Empty means all files."
    )


class Grep(Tool):
    name = "grep"
    description = (
        "Search file contents with a regular expression. Returns matching lines "
        "with file paths and line numbers. Use this to find where something is defined."
    )
    args_model = GrepArgs

    def run(self, args: GrepArgs, ctx: ToolContext) -> ToolResult:
        root = ctx.resolve(args.path)
        try:
            re.compile(args.pattern)
        except re.error as e:
            return ToolResult(f"Invalid regex: {e}", is_error=True)

        cmd = ["grep", "-rnI", "--color=never"]
        for d in SKIP_DIRS:
            cmd += ["--exclude-dir", d]
        if args.glob:
            cmd += ["--include", args.glob]
        cmd += ["-e", args.pattern, str(root)]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode not in (0, 1):
            return ToolResult(f"grep failed: {proc.stderr.strip()}", is_error=True)

        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if not lines:
            return ToolResult(f"No matches for {args.pattern!r}.")

        base = str(ctx.cwd.resolve())
        rel = [ln.replace(base + "/", "") for ln in lines[:MAX_MATCHES]]
        out = "\n".join(rel)
        if len(lines) > MAX_MATCHES:
            out += f"\n... ({len(lines) - MAX_MATCHES} more matches)"
        return ToolResult(out, summary=f"{len(lines)} matches")


class GlobArgs(BaseModel):
    pattern: str = Field(description="Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'.")


class Glob(Tool):
    name = "glob"
    description = "Find files by name pattern. Returns paths sorted by modification time."
    args_model = GlobArgs

    def run(self, args: GlobArgs, ctx: ToolContext) -> ToolResult:
        root = ctx.cwd.resolve()
        hits = [
            p for p in root.glob(args.pattern)
            if p.is_file() and not any(part in SKIP_DIRS for part in p.parts)
        ]
        if not hits:
            return ToolResult(f"No files match {args.pattern!r}.")
        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        rel = [str(p.relative_to(root)) for p in hits[:MAX_MATCHES]]
        out = "\n".join(rel)
        if len(hits) > MAX_MATCHES:
            out += f"\n... ({len(hits) - MAX_MATCHES} more)"
        return ToolResult(out, summary=f"{len(hits)} files")


class SemanticArgs(BaseModel):
    query: str = Field(
        description="Natural-language description of the code you are looking for."
    )
    k: int = Field(6, ge=1, le=20, description="Number of chunks to return.")


class SemanticSearch(Tool):
    name = "semantic_search"
    description = (
        "Find code by meaning rather than exact text, using the indexed codebase. "
        "Use when you do not know the exact name to grep for -- e.g. 'where is the "
        "retry logic' or 'how are sessions persisted'."
    )
    args_model = SemanticArgs

    def run(self, args: SemanticArgs, ctx: ToolContext) -> ToolResult:
        agent = ctx.agent
        if agent is None or agent.index is None:
            return ToolResult(
                "Semantic search is unavailable (no index). Use grep instead.",
                is_error=True,
            )
        try:
            hits = agent.index.search(args.query, k=args.k)
        except Exception as e:  # index may be empty or embeddings unavailable
            return ToolResult(f"Search failed: {e}. Use grep instead.", is_error=True)

        if not hits:
            return ToolResult(
                "No indexed matches. The codebase may not be indexed yet -- "
                "run /index, or use grep."
            )
        blocks = []
        for h in hits:
            blocks.append(
                f"--- {h['path']}:{h['start_line']}-{h['end_line']} "
                f"(similarity {h['score']:.2f})\n{h['content']}"
            )
        return ToolResult("\n\n".join(blocks), summary=f"{len(hits)} chunks")
