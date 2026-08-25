"""Environment-driven settings."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass
class Config:
    # --- inference ---
    model: str = field(default_factory=lambda: os.environ.get("MCC_MODEL", "qwen3:4b"))
    embed_model: str = field(
        default_factory=lambda: os.environ.get("MCC_EMBED_MODEL", "nomic-embed-text")
    )
    ollama_host: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    )
    # Ollama defaults num_ctx to 4096 and silently drops older messages, so this
    # is always sent explicitly. Size it against RAM, not ambition: qwen3:4b costs
    # roughly 147 KB of KV cache per token, so 16k context is ~2.4 GB on top of
    # 2.5 GB of weights -- enough to swap an 8 GB machine down to 0.1 tok/s.
    # 8192 measured ~100x faster than 16384 on an 8 GB M2.
    num_ctx: int = field(default_factory=lambda: _env_int("MCC_NUM_CTX", 8192))
    temperature: float = 0.3
    # Display only. Thinking is always requested from models that support it --
    # see OllamaClient.stream_chat for why suppressing it backfires.
    think: bool = field(
        default_factory=lambda: os.environ.get("MCC_THINK", "0") not in ("0", "", "false")
    )

    # --- agent loop ---
    max_steps: int = field(default_factory=lambda: _env_int("MCC_MAX_STEPS", 15))
    max_arg_retries: int = 2
    compact_at: float = 0.70  # fraction of num_ctx that triggers compaction

    # --- storage ---
    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "MCC_DATABASE_URL", "postgresql:///mcc"
        )
    )

    # --- workspace ---
    cwd: Path = field(default_factory=Path.cwd)
    yolo: bool = False

    @property
    def embed_dims(self) -> int:
        return 768  # nomic-embed-text; matches vector(768) in the schema


CONFIG = Config()
