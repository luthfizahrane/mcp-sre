import os
import logging
from typing import Optional

import asyncpg
from pgvector.asyncpg import register_vector

logger = logging.getLogger(__name__)

# Global pool instance
_pool: Optional[asyncpg.pool.Pool] = None


class PooledConnection:
    """Wrapper around an asyncpg connection acquired from a pool.

    Provides attribute proxying to the underlying connection and a
    compatible async close() that returns the connection to the pool.
    """
    def __init__(self, conn: asyncpg.Connection, pool: asyncpg.pool.Pool):
        self._conn = conn
        self._pool = pool

    def __getattr__(self, name):
        return getattr(self._conn, name)

    async def close(self):
        # Release the connection back to the pool. If release fails,
        # attempt to actually close the underlying connection.
        try:
            # asyncpg Pool.release is a coroutine; await to ensure release completes
            try:
                await self._pool.release(self._conn)
            except TypeError:
                # Some asyncpg versions may implement release as a non-awaitable
                # function. Fall back to calling without await.
                self._pool.release(self._conn)
        except Exception:
            try:
                await self._conn.close()
            except Exception:
                logger.exception("Failed to close pooled connection")


async def _init_pool() -> asyncpg.pool.Pool:
    global _pool
    if _pool is not None:
        return _pool

    db_host = os.getenv("SUPABASE_DB_HOST")
    db_port = os.getenv("SUPABASE_DB_PORT")
    db_user = os.getenv("SUPABASE_DB_USER")
    db_password = os.getenv("SUPABASE_DB_PASSWORD")
    db_name = os.getenv("SUPABASE_DB_NAME")

    if not all([db_host, db_port, db_user, db_password, db_name]):
        raise ValueError("Postgres DB env vars not fully set (SUPABASE_DB_HOST/PORT/USER/PASSWORD/NAME)")

    _pool = await asyncpg.create_pool(
        host=db_host,
        port=int(db_port),
        user=db_user,
        password=db_password,
        database=db_name,
        min_size=1,
        max_size=10,
    )

    logger.info("Postgres pool created")
    return _pool


async def _get_pg_connection() -> PooledConnection:
    """Acquire a connection from the pool and ensure pgvector is registered.

    Returns a PooledConnection wrapper that exposes the same asyncpg
    methods and supports `await conn.close()` to return the connection.
    """
    global _pool
    if _pool is None:
        await _init_pool()

    conn = await _pool.acquire()
    try:
        # register pgvector codecs on this connection (safe to call multiple times)
        await register_vector(conn)
    except Exception:
        logger.exception("register_vector failed on connection")

    return PooledConnection(conn, _pool)


async def close_pg_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
