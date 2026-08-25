"""Token budgeting and history compaction."""
from __future__ import annotations

from typing import Any

# No tokeniser for local models is exact; ~3.6 chars/token is a good working
# estimate for code-heavy English and errs slightly conservative.
CHARS_PER_TOKEN = 3.6


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def message_tokens(msg: dict[str, Any]) -> int:
    n = estimate_tokens(msg.get("content") or "")
    for call in msg.get("tool_calls") or []:
        n += estimate_tokens(str(call))
    return n + 4  # role/formatting overhead


def total_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(message_tokens(m) for m in messages)


SUMMARY_PROMPT = """\
Summarise the conversation so far into a compact handover note. Include:
- what the user asked for, in their own terms
- files read or modified, with paths
- decisions made and why
- what is still outstanding

Be specific about names and paths; drop pleasantries and tool noise. \
Write it as notes, not prose."""


def needs_compaction(messages: list[dict[str, Any]], num_ctx: int, threshold: float) -> bool:
    return total_tokens(messages) > num_ctx * threshold


def split_for_compaction(
    messages: list[dict[str, Any]], keep_recent: int = 6
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split into (to_fold, to_keep).

    Never folds the system message, and keeps the tail intact so the model
    retains immediate working state. The split point is nudged so a tool
    result is never separated from the assistant turn that requested it.
    """
    system = [m for m in messages if m.get("role") == "system"]
    body = [m for m in messages if m.get("role") != "system"]

    if len(body) <= keep_recent:
        return [], messages

    cut = len(body) - keep_recent
    while cut < len(body) and body[cut].get("role") == "tool":
        cut += 1

    return body[:cut], system + body[cut:]
