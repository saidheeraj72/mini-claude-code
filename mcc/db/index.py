"""Codebase indexing and hybrid retrieval over pgvector."""
from __future__ import annotations

import hashlib
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from mcc.db.pool import pool
from mcc.tools.fs import SKIP_DIRS

CHUNK_LINES = 40
CHUNK_OVERLAP = 10
MAX_FILE_BYTES = 400_000
EMBED_BATCH = 32

TEXT_SUFFIXES = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb", ".php",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".swift", ".kt", ".scala", ".sh", ".bash",
    ".sql", ".html", ".css", ".scss", ".md", ".rst", ".txt", ".toml", ".yaml",
    ".yml", ".json", ".ini", ".cfg", ".conf", ".zig", ".lua", ".ex", ".exs",
}

# Lines that plausibly start a top-level definition -- used to snap chunk
# boundaries so a function is less likely to be split down the middle.
DEF_RE = re.compile(
    r"^\s*(def |class |func |function |public |private |protected |const |"
    r"export |async def |impl |type |struct |interface |fn )"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _lang(path: Path) -> str:
    return path.suffix.lstrip(".") or "text"


def iter_source_files(root: Path) -> Iterable[Path]:
    """Repo files worth indexing, honouring .gitignore when inside a repo."""
    tracked: set[Path] | None = None
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=str(root), capture_output=True, text=True, timeout=20,
        )
        if proc.returncode == 0:
            tracked = {(root / line).resolve() for line in proc.stdout.splitlines() if line}
    except (OSError, subprocess.SubprocessError):
        tracked = None

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        if tracked is not None and path.resolve() not in tracked:
            continue
        yield path


def chunk_lines(lines: list[str]) -> list[tuple[int, int, str]]:
    """Split into overlapping windows, snapping to definition boundaries."""
    out: list[tuple[int, int, str]] = []
    i = 0
    n = len(lines)
    while i < n:
        end = min(i + CHUNK_LINES, n)
        # If a definition starts just past the window, cut before it instead.
        for probe in range(end, min(end + 8, n)):
            if DEF_RE.match(lines[probe - 1] if probe > 0 else ""):
                end = probe - 1
                break
        end = max(end, i + 1)
        body = "\n".join(lines[i:end]).strip()
        if body:
            out.append((i + 1, end, body))
        if end >= n:
            break
        i = max(end - CHUNK_OVERLAP, i + 1)
    return out


class CodeIndex:
    def __init__(self, root: Path, embedder: Callable[[list[str]], list[list[float]]]):
        self.root = root.resolve()
        self.embed = embedder

    # ------------------------------------------------------------- indexing
    def build(self, progress: Callable[[str], None] | None = None) -> dict[str, int]:
        """Index the repo. Files whose content hash is unchanged are skipped."""
        stats = {"scanned": 0, "indexed": 0, "skipped": 0, "chunks": 0}
        root_key = str(self.root)

        with pool().connection() as conn:
            existing = {
                r["path"]: r["sha256"]
                for r in conn.execute(
                    "SELECT path, sha256 FROM files WHERE repo_root = %s", (root_key,)
                ).fetchall()
            }

        seen: set[str] = set()
        for path in iter_source_files(self.root):
            stats["scanned"] += 1
            rel = str(path.relative_to(self.root))
            seen.add(rel)
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            digest = _sha256(raw)
            if existing.get(rel) == digest:
                stats["skipped"] += 1
                continue

            text = raw.decode("utf-8", errors="replace")
            pieces = chunk_lines(text.splitlines())
            if not pieces:
                continue
            if progress:
                progress(rel)

            vectors: list[list[float]] = []
            for start in range(0, len(pieces), EMBED_BATCH):
                batch = [p[2] for p in pieces[start : start + EMBED_BATCH]]
                vectors.extend(self.embed(batch))

            st = path.stat()
            with pool().connection() as conn:
                row = conn.execute(
                    """INSERT INTO files (repo_root, path, sha256, mtime, size, lang)
                       VALUES (%s, %s, %s, %s, %s, %s)
                       ON CONFLICT (repo_root, path) DO UPDATE
                         SET sha256 = EXCLUDED.sha256, mtime = EXCLUDED.mtime,
                             size = EXCLUDED.size, indexed_at = now()
                       RETURNING id""",
                    (root_key, rel, digest,
                     datetime.fromtimestamp(st.st_mtime, timezone.utc),
                     st.st_size, _lang(path)),
                ).fetchone()
                file_id = row["id"]
                conn.execute("DELETE FROM chunks WHERE file_id = %s", (file_id,))
                with conn.cursor() as cur:
                    cur.executemany(
                        """INSERT INTO chunks
                             (file_id, start_line, end_line, content, embedding)
                           VALUES (%s, %s, %s, %s, %s)""",
                        [
                            (file_id, s, e, body, _vec(v))
                            for (s, e, body), v in zip(pieces, vectors)
                        ],
                    )
            stats["indexed"] += 1
            stats["chunks"] += len(pieces)

        # Drop files that no longer exist on disk.
        stale = set(existing) - seen
        if stale:
            with pool().connection() as conn:
                conn.execute(
                    "DELETE FROM files WHERE repo_root = %s AND path = ANY(%s)",
                    (root_key, list(stale)),
                )
        return stats

    # ------------------------------------------------------------ retrieval
    def search(self, query: str, k: int = 6) -> list[dict[str, Any]]:
        """Vector search, blended with a literal match bonus.

        Pure embedding similarity underperforms on code -- identifiers carry
        signal that embeddings wash out -- so chunks containing the query's
        literal terms get a small boost.
        """
        vec = _vec(self.embed([query])[0])
        with pool().connection() as conn:
            rows = conn.execute(
                """SELECT f.path, c.start_line, c.end_line, c.content,
                          1 - (c.embedding <=> %s::vector) AS score
                   FROM chunks c JOIN files f ON f.id = c.file_id
                   WHERE f.repo_root = %s AND c.embedding IS NOT NULL
                   ORDER BY c.embedding <=> %s::vector
                   LIMIT %s""",
                (vec, str(self.root), vec, k * 3),
            ).fetchall()

        terms = [t.lower() for t in re.findall(r"\w{3,}", query)]
        for r in rows:
            body = r["content"].lower()
            hits = sum(1 for t in terms if t in body)
            r["score"] = float(r["score"]) + 0.04 * hits
        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows[:k]

    def stats(self) -> dict[str, int]:
        with pool().connection() as conn:
            row = conn.execute(
                """SELECT count(DISTINCT f.id) AS files, count(c.id) AS chunks
                   FROM files f LEFT JOIN chunks c ON c.file_id = f.id
                   WHERE f.repo_root = %s""",
                (str(self.root),),
            ).fetchone()
        return {"files": row["files"] or 0, "chunks": row["chunks"] or 0}


def _vec(values: list[float]) -> str:
    """pgvector literal."""
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"
