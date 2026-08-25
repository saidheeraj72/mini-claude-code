"""Session, message, tool-call and permission persistence."""
from __future__ import annotations

import json
from typing import Any

from psycopg.types.json import Jsonb

from mcc.db.pool import pool


# ------------------------------------------------------------------ sessions
def create_session(cwd: str, model: str) -> str:
    with pool().connection() as conn:
        row = conn.execute(
            "INSERT INTO sessions (cwd, model) VALUES (%s, %s) RETURNING id",
            (cwd, model),
        ).fetchone()
    return str(row["id"])


def set_title(session_id: str, title: str) -> None:
    with pool().connection() as conn:
        conn.execute(
            "UPDATE sessions SET title = %s WHERE id = %s AND title IS NULL",
            (title[:120], session_id),
        )


def touch(session_id: str) -> None:
    with pool().connection() as conn:
        conn.execute("UPDATE sessions SET updated_at = now() WHERE id = %s", (session_id,))


def list_sessions(cwd: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    sql = """
        SELECT s.id, s.title, s.model, s.cwd, s.updated_at,
               (SELECT count(*) FROM messages m WHERE m.session_id = s.id) AS n_messages
        FROM sessions s
        {where}
        ORDER BY s.updated_at DESC
        LIMIT %s
    """
    params: list[Any] = []
    where = ""
    if cwd:
        where = "WHERE s.cwd = %s"
        params.append(cwd)
    params.append(limit)
    with pool().connection() as conn:
        return conn.execute(sql.format(where=where), params).fetchall()


def latest_session(cwd: str) -> str | None:
    with pool().connection() as conn:
        row = conn.execute(
            "SELECT id FROM sessions WHERE cwd = %s ORDER BY updated_at DESC LIMIT 1",
            (cwd,),
        ).fetchone()
    return str(row["id"]) if row else None


def session_exists(session_id: str) -> bool:
    with pool().connection() as conn:
        return conn.execute(
            "SELECT 1 FROM sessions WHERE id = %s", (session_id,)
        ).fetchone() is not None


# ------------------------------------------------------------------ messages
def next_seq(session_id: str) -> int:
    with pool().connection() as conn:
        row = conn.execute(
            "SELECT COALESCE(max(seq), -1) + 1 AS n FROM messages WHERE session_id = %s",
            (session_id,),
        ).fetchone()
    return int(row["n"])


def add_message(
    session_id: str,
    role: str,
    content: str = "",
    tool_calls: list[dict] | None = None,
    tool_name: str | None = None,
    token_count: int = 0,
) -> int:
    with pool().connection() as conn:
        seq = conn.execute(
            "SELECT COALESCE(max(seq), -1) + 1 AS n FROM messages WHERE session_id = %s",
            (session_id,),
        ).fetchone()["n"]
        row = conn.execute(
            """INSERT INTO messages
                 (session_id, seq, role, content, tool_calls, tool_name, token_count)
               VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (
                session_id, seq, role, content,
                Jsonb(tool_calls) if tool_calls else None,
                tool_name, token_count,
            ),
        ).fetchone()
        conn.execute("UPDATE sessions SET updated_at = now() WHERE id = %s", (session_id,))
    return int(row["id"])


def load_messages(session_id: str, include_superseded: bool = False) -> list[dict[str, Any]]:
    sql = """SELECT id, seq, role, content, tool_calls, tool_name, token_count, superseded
             FROM messages WHERE session_id = %s {extra} ORDER BY seq"""
    extra = "" if include_superseded else "AND superseded = false"
    with pool().connection() as conn:
        return conn.execute(sql.format(extra=extra), (session_id,)).fetchall()


def mark_superseded(message_ids: list[int]) -> None:
    if not message_ids:
        return
    with pool().connection() as conn:
        conn.execute(
            "UPDATE messages SET superseded = true WHERE id = ANY(%s)", (message_ids,)
        )


# ---------------------------------------------------------------- tool calls
def record_tool_call(
    session_id: str,
    name: str,
    args: dict,
    result: str,
    is_error: bool,
    duration_ms: int,
    approved_by: str,
    message_id: int | None = None,
) -> None:
    with pool().connection() as conn:
        conn.execute(
            """INSERT INTO tool_calls
                 (session_id, message_id, name, args, result, is_error,
                  duration_ms, approved_by)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (session_id, message_id, name, Jsonb(args), result[:20000],
             is_error, duration_ms, approved_by),
        )


# --------------------------------------------------------------- permissions
def save_rule(tool_name: str, pattern: str, decision: str, session_id: str | None) -> None:
    with pool().connection() as conn:
        conn.execute(
            """INSERT INTO permissions (session_id, tool_name, pattern, decision)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (COALESCE(session_id,
                                     '00000000-0000-0000-0000-000000000000'::uuid),
                            tool_name, pattern)
               DO UPDATE SET decision = EXCLUDED.decision""",
            (session_id, tool_name, pattern, decision),
        )


def load_rules(session_id: str | None) -> list[dict[str, Any]]:
    with pool().connection() as conn:
        return conn.execute(
            """SELECT tool_name, pattern, decision FROM permissions
               WHERE session_id IS NULL OR session_id = %s""",
            (session_id,),
        ).fetchall()
