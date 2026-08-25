"""Tool registry."""
from __future__ import annotations

from mcc.tools.base import Tool
from mcc.tools.fs import EditFile, ListDir, ReadFile, WriteFile
from mcc.tools.search import Glob, Grep, SemanticSearch
from mcc.tools.shell import RunBash

ALL_TOOLS: list[Tool] = [
    ReadFile(), WriteFile(), EditFile(), ListDir(),
    Grep(), Glob(), SemanticSearch(), RunBash(),
]

BY_NAME: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}


def schemas() -> list[dict]:
    return [t.schema() for t in ALL_TOOLS]
