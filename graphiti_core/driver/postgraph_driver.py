"""
PostGraph driver — a PostgreSQL-backed graph backend for Graphiti.

Uses asyncpg for async PostgreSQL access with pgvector for embeddings,
tsvector for fulltext search, and recursive CTEs for graph traversal.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager, suppress
from typing import Any

try:
    import asyncpg
except ImportError:
    raise ImportError(
        'asyncpg is required for PostGraphDriver. '
        'Install it with: pip install graphiti-core[postgraph]'
    ) from None

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.driver.operations.community_edge_ops import CommunityEdgeOperations
from graphiti_core.driver.operations.community_node_ops import CommunityNodeOperations
from graphiti_core.driver.operations.entity_edge_ops import EntityEdgeOperations
from graphiti_core.driver.operations.entity_node_ops import EntityNodeOperations
from graphiti_core.driver.operations.episode_node_ops import EpisodeNodeOperations
from graphiti_core.driver.operations.episodic_edge_ops import EpisodicEdgeOperations
from graphiti_core.driver.operations.graph_ops import GraphMaintenanceOperations
from graphiti_core.driver.operations.has_episode_edge_ops import HasEpisodeEdgeOperations
from graphiti_core.driver.operations.next_episode_edge_ops import NextEpisodeEdgeOperations
from graphiti_core.driver.operations.saga_node_ops import SagaNodeOperations
from graphiti_core.driver.operations.search_ops import SearchOperations
from graphiti_core.driver.postgraph.operations.community_edge_ops import (
    PGCommunityEdgeOperations,
)
from graphiti_core.driver.postgraph.operations.community_node_ops import (
    PGCommunityNodeOperations,
)
from graphiti_core.driver.postgraph.operations.entity_edge_ops import PGEntityEdgeOperations
from graphiti_core.driver.postgraph.operations.entity_node_ops import PGEntityNodeOperations
from graphiti_core.driver.postgraph.operations.episode_node_ops import PGEpisodeNodeOperations
from graphiti_core.driver.postgraph.operations.episodic_edge_ops import PGEpisodicEdgeOperations
from graphiti_core.driver.postgraph.operations.graph_ops import PGGraphMaintenanceOperations
from graphiti_core.driver.postgraph.operations.has_episode_edge_ops import (
    PGHasEpisodeEdgeOperations,
)
from graphiti_core.driver.postgraph.operations.next_episode_edge_ops import (
    PGNextEpisodeEdgeOperations,
)
from graphiti_core.driver.postgraph.operations.saga_node_ops import PGSagaNodeOperations
from graphiti_core.driver.postgraph.operations.search_ops import PGSearchOperations
from graphiti_core.driver.query_executor import Transaction

logger = logging.getLogger(__name__)

EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '1024'))


class PostGraphDriverSession(GraphDriverSession):
    provider = GraphProvider.POSTGRAPH

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool
        self._conn: asyncpg.Connection | None = None

    async def __aenter__(self):
        self._conn = await self._pool.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._conn is not None:
            await self._pool.release(self._conn)
            self._conn = None

    async def run(self, query: str, **kwargs: Any) -> Any:
        conn = self._conn or self._pool
        return await _execute_sql(conn, query, kwargs)

    async def close(self):
        if self._conn is not None:
            await self._pool.release(self._conn)
            self._conn = None

    async def execute_write(self, func, *args, **kwargs):
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                return await func(conn, *args, **kwargs)


class _PGTransaction(Transaction):
    def __init__(self, conn: asyncpg.Connection):
        self._conn = conn

    async def run(self, query: str, **kwargs: Any) -> Any:
        return await _execute_sql(self._conn, query, kwargs)


async def _execute_sql(
    conn: asyncpg.Connection | asyncpg.Pool,
    query: str,
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], None, None]:
    clean_params = {
        k: v for k, v in params.items() if k not in ('routing_', 'database_')
    }

    # Convert $param_name to $1, $2, ... for asyncpg
    param_names: list[str] = []
    converted = query
    for key in sorted(clean_params.keys(), key=len, reverse=True):
        placeholder = f'${key}'
        if placeholder in converted:
            param_names.append(key)
            converted = converted.replace(placeholder, f'__PG_PARAM_{len(param_names)}__')

    for i, _ in enumerate(param_names):
        converted = converted.replace(f'__PG_PARAM_{i + 1}__', f'${i + 1}')

    param_values = [_serialize_param(clean_params[name]) for name in param_names]

    rows = await conn.fetch(converted, *param_values)
    records = [dict(r) for r in rows]
    return (records, None, None)


def _serialize_param(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps(value)
    if isinstance(value, list) and value and isinstance(value[0], float):
        return str(value)
    return value


class PostGraphDriver(GraphDriver):
    provider = GraphProvider.POSTGRAPH
    default_group_id: str = ''

    def __init__(
        self,
        dsn: str | None = None,
        *,
        host: str = 'localhost',
        port: int = 5432,
        user: str = 'postgres',
        password: str = '',
        database: str = 'graphiti',
        embedding_dim: int | None = None,
    ):
        super().__init__()
        self._dsn = dsn or f'postgresql://{user}:{password}@{host}:{port}/{database}'
        self._database = database
        self._pool: asyncpg.Pool | None = None
        self._embedding_dim = embedding_dim or EMBEDDING_DIM

        self._entity_node_ops = PGEntityNodeOperations()
        self._episode_node_ops = PGEpisodeNodeOperations()
        self._community_node_ops = PGCommunityNodeOperations()
        self._saga_node_ops = PGSagaNodeOperations()
        self._entity_edge_ops = PGEntityEdgeOperations()
        self._episodic_edge_ops = PGEpisodicEdgeOperations()
        self._community_edge_ops = PGCommunityEdgeOperations()
        self._has_episode_edge_ops = PGHasEpisodeEdgeOperations()
        self._next_episode_edge_ops = PGNextEpisodeEdgeOperations()
        self._search_ops = PGSearchOperations()
        self._graph_ops = PGGraphMaintenanceOperations(self._embedding_dim)

        self._init_task: asyncio.Task | None = None
        try:
            loop = asyncio.get_running_loop()
            self._init_task = loop.create_task(self._init())
        except RuntimeError:
            pass

    async def _init(self):
        await self._ensure_pool()
        await self.build_indices_and_constraints()

    async def _ensure_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        return self._pool

    @property
    def entity_node_ops(self) -> EntityNodeOperations:
        return self._entity_node_ops

    @property
    def episode_node_ops(self) -> EpisodeNodeOperations:
        return self._episode_node_ops

    @property
    def community_node_ops(self) -> CommunityNodeOperations:
        return self._community_node_ops

    @property
    def saga_node_ops(self) -> SagaNodeOperations:
        return self._saga_node_ops

    @property
    def entity_edge_ops(self) -> EntityEdgeOperations:
        return self._entity_edge_ops

    @property
    def episodic_edge_ops(self) -> EpisodicEdgeOperations:
        return self._episodic_edge_ops

    @property
    def community_edge_ops(self) -> CommunityEdgeOperations:
        return self._community_edge_ops

    @property
    def has_episode_edge_ops(self) -> HasEpisodeEdgeOperations:
        return self._has_episode_edge_ops

    @property
    def next_episode_edge_ops(self) -> NextEpisodeEdgeOperations:
        return self._next_episode_edge_ops

    @property
    def search_ops(self) -> SearchOperations:
        return self._search_ops

    @property
    def graph_ops(self) -> GraphMaintenanceOperations:
        return self._graph_ops

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Transaction]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            tx = conn.transaction()
            await tx.start()
            try:
                yield _PGTransaction(conn)
                await tx.commit()
            except BaseException:
                await tx.rollback()
                raise

    async def execute_query(self, cypher_query_: str, **kwargs: Any) -> Coroutine:
        pool = await self._ensure_pool()
        return await _execute_sql(pool, cypher_query_, kwargs)

    def session(self, database: str | None = None) -> GraphDriverSession:
        if self._pool is None:
            raise RuntimeError('PostGraphDriver pool not initialized. Await _ensure_pool() first.')
        return PostGraphDriverSession(self._pool)

    async def close(self) -> None:
        if self._init_task is not None:
            if not self._init_task.done():
                self._init_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._init_task
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def delete_all_indexes(self) -> Coroutine:
        return self.execute_query("SELECT 1")

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            if delete_existing:
                await self._graph_ops.delete_all_indexes_raw(conn)
            await self._graph_ops.build_indices_and_constraints_raw(conn)

    def build_fulltext_query(
        self, query: str, group_ids: list[str] | None = None, max_query_length: int = 128
    ) -> str:
        words = query.strip().split()[:max_query_length]
        ts_query = ' & '.join(w for w in words if w)
        return ts_query or ''
