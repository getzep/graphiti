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
from neo4j.exceptions import ClientError
from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.driver.neo4j.operations.community_edge_ops import Neo4jCommunityEdgeOperations
from graphiti_core.driver.neo4j.operations.community_node_ops import Neo4jCommunityNodeOperations
from graphiti_core.driver.neo4j.operations.entity_edge_ops import Neo4jEntityEdgeOperations
from graphiti_core.driver.neo4j.operations.entity_node_ops import Neo4jEntityNodeOperations
from graphiti_core.driver.neo4j.operations.episode_node_ops import Neo4jEpisodeNodeOperations
from graphiti_core.driver.neo4j.operations.episodic_edge_ops import Neo4jEpisodicEdgeOperations
from graphiti_core.driver.neo4j.operations.graph_ops import Neo4jGraphMaintenanceOperations
from graphiti_core.driver.neo4j.operations.has_episode_edge_ops import (
    Neo4jHasEpisodeEdgeOperations,
)
from graphiti_core.driver.neo4j.operations.next_episode_edge_ops import (
    Neo4jNextEpisodeEdgeOperations,
)
from graphiti_core.driver.neo4j.operations.saga_node_ops import Neo4jSagaNodeOperations
from graphiti_core.driver.neo4j.operations.search_ops import Neo4jSearchOperations
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
from graphiti_core.helpers import semaphore_gather

logger = logging.getLogger(__name__)


class Neo4jDriver(GraphDriver):
    provider = GraphProvider.NEO4J
    default_group_id: str = ''

    def __init__(
        self,
        uri: str,
        user: str | None,
        password: str | None,
        database: str = 'neo4j',
    ):
        super().__init__()
        self.client = AsyncGraphDatabase.driver(
            uri=uri,
            auth=(user or '', password or ''),
        )
        self._database = database

        # Instantiate Neo4j operations
        self._entity_node_ops = Neo4jEntityNodeOperations()
        self._episode_node_ops = Neo4jEpisodeNodeOperations()
        self._community_node_ops = Neo4jCommunityNodeOperations()
        self._saga_node_ops = Neo4jSagaNodeOperations()
        self._entity_edge_ops = Neo4jEntityEdgeOperations()
        self._episodic_edge_ops = Neo4jEpisodicEdgeOperations()
        self._community_edge_ops = Neo4jCommunityEdgeOperations()
        self._has_episode_edge_ops = Neo4jHasEpisodeEdgeOperations()
        self._next_episode_edge_ops = Neo4jNextEpisodeEdgeOperations()
        self._search_ops = Neo4jSearchOperations()
        self._graph_ops = Neo4jGraphMaintenanceOperations()

        # Schedule the indices and constraints to be built
        import asyncio

        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # Schedule the build_indices_and_constraints to run
            loop.create_task(self.build_indices_and_constraints())
        except RuntimeError:
            # No event loop running, this will be handled later
            pass

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
        """Neo4j transaction with real commit/rollback semantics."""
        async with self.client.session(database=self._database) as session:
            tx = await session.begin_transaction()
            try:
                yield _Neo4jTransaction(tx)
                await tx.commit()
            except BaseException:
                await tx.rollback()
                raise

    async def execute_query(self, cypher_query_: LiteralString, **kwargs: Any) -> EagerResult:
        # Check if database_ is provided in kwargs.
        # If not populated, set the value to retain backwards compatibility
        params = kwargs.pop('params', None)
        if params is None:
            params = {}
        params.setdefault('database_', self._database)

        try:
            result = await self.client.execute_query(cypher_query_, parameters_=params, **kwargs)
        except Exception as e:
            logger.error(f'Error executing Neo4j query: {e}\n{cypher_query_}\n{params}')
            raise

        return result

    def session(self, database: str | None = None) -> GraphDriverSession:
        _database = database or self._database
        return self.client.session(database=_database)  # type: ignore

    async def close(self) -> None:
        return await self.client.close()

    def delete_all_indexes(self) -> Coroutine:
        return self.client.execute_query(
            'CALL db.indexes() YIELD name DROP INDEX name',
        )

    async def _execute_index_query(self, query: LiteralString) -> EagerResult | None:
        """Execute an index creation query, ignoring 'index already exists' errors.

        Neo4j can raise EquivalentSchemaRuleAlreadyExists when concurrent CREATE INDEX
        IF NOT EXISTS queries race, even though the index exists. This is safe to ignore.
        """
        try:
            return await self.execute_query(query)
        except ClientError as e:
            # Ignore "equivalent index already exists" error (race condition with IF NOT EXISTS)
            if 'EquivalentSchemaRuleAlreadyExists' in str(e):
                logger.debug(f'Index already exists (concurrent creation): {query[:50]}...')
                return None
            raise

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        if delete_existing:
            await self.delete_all_indexes()

        range_indices: list[LiteralString] = get_range_indices(self.provider)

        fulltext_indices: list[LiteralString] = get_fulltext_indices(self.provider)

        index_queries: list[LiteralString] = range_indices + fulltext_indices

        await semaphore_gather(*[self._execute_index_query(query) for query in index_queries])

    async def health_check(self) -> None:
        """Check Neo4j connectivity by running the driver's verify_connectivity method."""
        try:
            await self.client.verify_connectivity()
            return None
        except Exception as e:
            print(f'Neo4j health check failed: {e}')
            raise


class _Neo4jTransaction(Transaction):
    """Wraps a Neo4j AsyncTransaction for the Transaction ABC."""

    def __init__(self, tx: Any):
        self._tx = tx

    async def run(self, query: str, **kwargs: Any) -> Any:
        return await self._tx.run(query, **kwargs)
