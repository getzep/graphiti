"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from typing import Any

from neo4j import AsyncGraphDatabase, EagerResult
from typing_extensions import LiteralString

from graphiti_core.driver.arcadedb.operations.community_edge_ops import (
    ArcadeDBCommunityEdgeOperations,
)
from graphiti_core.driver.arcadedb.operations.community_node_ops import (
    ArcadeDBCommunityNodeOperations,
)
from graphiti_core.driver.arcadedb.operations.entity_edge_ops import ArcadeDBEntityEdgeOperations
from graphiti_core.driver.arcadedb.operations.entity_node_ops import ArcadeDBEntityNodeOperations
from graphiti_core.driver.arcadedb.operations.episode_node_ops import ArcadeDBEpisodeNodeOperations
from graphiti_core.driver.arcadedb.operations.episodic_edge_ops import (
    ArcadeDBEpisodicEdgeOperations,
)
from graphiti_core.driver.arcadedb.operations.graph_ops import ArcadeDBGraphMaintenanceOperations
from graphiti_core.driver.arcadedb.operations.has_episode_edge_ops import (
    ArcadeDBHasEpisodeEdgeOperations,
)
from graphiti_core.driver.arcadedb.operations.next_episode_edge_ops import (
    ArcadeDBNextEpisodeEdgeOperations,
)
from graphiti_core.driver.arcadedb.operations.saga_node_ops import ArcadeDBSagaNodeOperations
from graphiti_core.driver.arcadedb.operations.search_ops import ArcadeDBSearchOperations
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
from graphiti_core.driver.query_executor import Transaction
from graphiti_core.graph_queries import get_fulltext_indices, get_range_indices

logger = logging.getLogger(__name__)


class ArcadeDBDriver(GraphDriver):
    """Graph driver for ArcadeDB using the Neo4j Bolt wire protocol.

    ArcadeDB 26.2.1+ ships the Bolt protocol, allowing standard Neo4j async
    drivers to connect directly. This driver reuses the ``neo4j`` Python
    async driver for transport while adapting queries for ArcadeDB's
    OpenCypher and SQL dialect.
    """

    provider = GraphProvider.ARCADEDB
    default_group_id: str = ''

    def __init__(
        self,
        uri: str,
        user: str | None,
        password: str | None,
        database: str = 'graphiti',
    ):
        super().__init__()
        self.client = AsyncGraphDatabase.driver(
            uri=uri,
            auth=(user or '', password or ''),
        )
        self._database = database

        # Instantiate ArcadeDB operations
        self._entity_node_ops = ArcadeDBEntityNodeOperations()
        self._episode_node_ops = ArcadeDBEpisodeNodeOperations()
        self._community_node_ops = ArcadeDBCommunityNodeOperations()
        self._saga_node_ops = ArcadeDBSagaNodeOperations()
        self._entity_edge_ops = ArcadeDBEntityEdgeOperations()
        self._episodic_edge_ops = ArcadeDBEpisodicEdgeOperations()
        self._community_edge_ops = ArcadeDBCommunityEdgeOperations()
        self._has_episode_edge_ops = ArcadeDBHasEpisodeEdgeOperations()
        self._next_episode_edge_ops = ArcadeDBNextEpisodeEdgeOperations()
        self._search_ops = ArcadeDBSearchOperations()
        self._graph_ops = ArcadeDBGraphMaintenanceOperations()

        self.aoss_client = None

    # --- Operations properties ---

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
        """ArcadeDB transaction with real commit/rollback semantics via Bolt."""
        async with self.client.session(database=self._database) as session:
            tx = await session.begin_transaction()
            try:
                yield _ArcadeDBTransaction(tx)
                await tx.commit()
            except BaseException:
                await tx.rollback()
                raise

    async def execute_query(self, cypher_query_: LiteralString, **kwargs: Any) -> EagerResult:
        params = kwargs.pop('params', None)
        if params is None:
            params = {}
        params.setdefault('database_', self._database)

        try:
            result = await self.client.execute_query(cypher_query_, parameters_=params, **kwargs)
        except Exception as e:
            logger.error(f'Error executing ArcadeDB query: {e}\n{cypher_query_}\n{params}')
            raise

        return result

    def session(self, database: str | None = None) -> GraphDriverSession:
        _database = database or self._database
        return self.client.session(database=_database)  # type: ignore

    async def close(self) -> None:
        return await self.client.close()

    def delete_all_indexes(self) -> Coroutine:
        # ArcadeDB does not support Neo4j's CALL db.indexes() YIELD name DROP INDEX name.
        # Return a no-op coroutine. Indexes should be managed via ArcadeDB SQL DDL.
        async def _noop():
            logger.warning(
                'delete_all_indexes via Bolt is not supported for ArcadeDB. '
                'Use ArcadeDB SQL DDL to manage indexes.'
            )

        return _noop()

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        if delete_existing:
            await self.delete_all_indexes()

        range_indices: list[LiteralString] = get_range_indices(self.provider)
        fulltext_indices: list[LiteralString] = get_fulltext_indices(self.provider)
        index_queries: list[LiteralString] = range_indices + fulltext_indices

        for query in index_queries:
            try:
                await self.execute_query(query)
            except Exception as e:
                # Index may already exist
                logger.debug(f'Index creation skipped (may already exist): {e}')

    async def health_check(self) -> None:
        """Check ArcadeDB connectivity via the Bolt driver."""
        try:
            await self.client.verify_connectivity()
            return None
        except Exception as e:
            print(f'ArcadeDB health check failed: {e}')
            raise


class _ArcadeDBTransaction(Transaction):
    """Wraps a Neo4j AsyncTransaction for ArcadeDB Bolt protocol."""

    def __init__(self, tx: Any):
        self._tx = tx

    async def run(self, query: str, **kwargs: Any) -> Any:
        return await self._tx.run(query, **kwargs)
