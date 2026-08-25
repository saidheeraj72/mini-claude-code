"""Apply SQL migrations in order. Idempotent -- every file uses IF NOT EXISTS."""
from __future__ import annotations

import sys
from pathlib import Path

import psycopg

from mcc.config import CONFIG

MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"


def run(dsn: str | None = None) -> None:
    dsn = dsn or CONFIG.database_url
    files = sorted(MIGRATIONS.glob("*.sql"))
    if not files:
        print(f"no migrations found in {MIGRATIONS}")
        return
    with psycopg.connect(dsn, autocommit=True) as conn:
        for path in files:
            print(f"applying {path.name}")
            conn.execute(path.read_text())
    print(f"applied {len(files)} migration(s) to {dsn}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
