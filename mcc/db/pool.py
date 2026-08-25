"""Postgres connection pool."""
from __future__ import annotations

from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

_pool: ConnectionPool | None = None


def init_pool(dsn: str) -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            dsn, min_size=1, max_size=4, kwargs={"row_factory": dict_row}, open=True
        )
    return _pool


def pool() -> ConnectionPool:
    if _pool is None:
        raise RuntimeError("pool not initialised; call init_pool() first")
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
