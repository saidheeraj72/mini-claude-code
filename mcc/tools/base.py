"""Tool contract: one pydantic model per tool drives both validation and schema."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Type

from pydantic import BaseModel, ValidationError


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    # Short line shown in the UI card, e.g. "read 42 lines".
    summary: str = ""


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    args_model: ClassVar[Type[BaseModel]]
    mutating: ClassVar[bool] = False  # gated by the permission system

    @abstractmethod
    def run(self, args: BaseModel, ctx: "ToolContext") -> ToolResult: ...

    # ------------------------------------------------------------------ glue
    def schema(self) -> dict[str, Any]:
        """Ollama/OpenAI-style function spec."""
        raw = self.args_model.model_json_schema()
        raw.pop("title", None)
        for prop in raw.get("properties", {}).values():
            prop.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": raw,
            },
        }

    def validate(self, args: dict[str, Any]) -> BaseModel:
        return self.args_model.model_validate(args)


def format_validation_error(tool_name: str, err: ValidationError) -> str:
    """Turn a pydantic error into feedback a small model can act on."""
    lines = [f"Invalid arguments for `{tool_name}`. Fix and call it again."]
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "(root)"
        lines.append(f"  - {loc}: {e['msg']}")
    return "\n".join(lines)


@dataclass
class ToolContext:
    """Everything a tool needs that is not in its arguments."""

    cwd: Path
    session_id: str | None = None
    agent: Any = None  # back-reference, used by search tools for embeddings

    def resolve(self, path: str) -> Path:
        """Resolve a user/model-supplied path, pinned inside the workspace."""
        p = Path(path).expanduser()
        p = p if p.is_absolute() else self.cwd / p
        p = p.resolve()
        root = self.cwd.resolve()
        if not (p == root or root in p.parents):
            raise ValueError(
                f"path escapes the workspace ({root}): {path}"
            )
        return p
