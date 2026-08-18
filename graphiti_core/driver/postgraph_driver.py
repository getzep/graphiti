"""
PostGraph driver — a PostgreSQL-backed graph backend for Graphiti.

Uses the post-graph library (AsyncPostGraph) for connection management,
vertex/edge CRUD, vector search, and graph traversal.  Graphiti's domain
fields are stored in post-graph's ``payload`` JSONB column; embeddings
map to post-graph's ``embedding`` vector column; and Graphiti's
``group_id`` maps to post-graph's ``realm``.
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
    from post_graph import AsyncPostGraph, Vertex, Edge
except ImportError:
    raise ImportError(
        'post-graph is required for PostGraphDriver. '
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
from graphiti_core.driver.postgraph.graph_operations_interface import PostGraphOperationsInterface
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

    def __init__(self, client: AsyncPostGraph):
        self._client = client
        self._conn = None

    async def __aenter__(self):
        pool = self._client.connection
        self._conn = await pool.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._conn is not None:
            pool = self._client.connection
            await pool.release(self._conn)
            self._conn = None

    async def run(self, query: str, **kwargs: Any) -> Any:
        conn = self._conn or self._client.connection
        return await _execute_sql(conn, query, kwargs)

    async def close(self):
        if self._conn is not None:
            pool = self._client.connection
            await pool.release(self._conn)
            self._conn = None

    async def execute_write(self, func, *args, **kwargs):
        pool = self._client.connection
        async with pool.acquire() as conn, conn.transaction():
            return await func(conn, *args, **kwargs)


class _PGTransaction(Transaction):
    def __init__(self, conn):
        self._conn = conn

    async def run(self, query: str, **kwargs: Any) -> Any:
        return await _execute_sql(self._conn, query, kwargs)


async def _execute_sql(conn, query: str, params: dict[str, Any]):
    clean_params = {k: v for k, v in params.items() if k not in ('routing_', 'database_')}

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
        self._embedding_dim = embedding_dim or EMBEDDING_DIM

        self._client: AsyncPostGraph | None = None

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

        self.graph_operations_interface = PostGraphOperationsInterface(self)

        self._init_task: asyncio.Task | None = None
        try:
            loop = asyncio.get_running_loop()
            self._init_task = loop.create_task(self._init())
        except RuntimeError:
            pass

    async def _init(self):
        await self._ensure_client()
        await self.build_indices_and_constraints()

    async def _ensure_client(self) -> AsyncPostGraph:
        if self._client is None:
            self._client = AsyncPostGraph(dsn=self._dsn)
            await self._client.connect()
        return self._client

    @property
    def client(self) -> AsyncPostGraph:
        if self._client is None:
            raise RuntimeError('PostGraphDriver not initialized. Await _ensure_client() first.')
        return self._client

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
        client = await self._ensure_client()
        pool = client.connection
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
        client = await self._ensure_client()
        return await _execute_sql(client.connection, cypher_query_, kwargs)

    def session(self, database: str | None = None) -> GraphDriverSession:
        if self._client is None:
            raise RuntimeError('PostGraphDriver not initialized. Await _ensure_client() first.')
        return PostGraphDriverSession(self._client)

    async def close(self) -> None:
        if self._init_task is not None and not self._init_task.done():
            self._init_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._init_task
        if self._client is not None:
            await self._client.close()
            self._client = None

    def delete_all_indexes(self) -> Coroutine:
        return self.execute_query('SELECT 1')

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        client = await self._ensure_client()
        await self._graph_ops.build_indices_and_constraints_pg(client, self._embedding_dim)

    def build_fulltext_query(
        self, query: str, group_ids: list[str] | None = None, max_query_length: int = 128
    ) -> str:
        words = query.strip().split()[:max_query_length]
        ts_query = ' & '.join(w for w in words if w)
        return ts_query or ''

    async def _resolve_vertex_id(
        self, table_name: str, realm: str, graphiti_uuid: str,
    ) -> int | None:
        """Get post-graph's integer id from a Graphiti UUID stored in payload."""
        rows = await self.client._fetch(
            f'SELECT id FROM "{table_name}" WHERE realm = $1 AND payload @> $2::jsonb',
            realm, json.dumps({'uuid': graphiti_uuid}),
        )
        return rows[0]['id'] if rows else None

    async def _resolve_edge_id(
        self, table_name: str, realm: str, graphiti_uuid: str,
    ) -> int | None:
        """Get post-graph's integer edge id from a Graphiti UUID stored in payload."""
        rows = await self.client._fetch(
            f'SELECT id FROM "{table_name}" WHERE realm = $1 AND payload @> $2::jsonb',
            realm, json.dumps({'uuid': graphiti_uuid}),
        )
        return rows[0]['id'] if rows else None
