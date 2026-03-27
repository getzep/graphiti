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
from typing import Any

import kuzu

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.driver.kuzu.operations.community_edge_ops import KuzuCommunityEdgeOperations
from graphiti_core.driver.kuzu.operations.community_node_ops import KuzuCommunityNodeOperations
from graphiti_core.driver.kuzu.operations.entity_edge_ops import KuzuEntityEdgeOperations
from graphiti_core.driver.kuzu.operations.entity_node_ops import KuzuEntityNodeOperations
from graphiti_core.driver.kuzu.operations.episode_node_ops import KuzuEpisodeNodeOperations
from graphiti_core.driver.kuzu.operations.episodic_edge_ops import KuzuEpisodicEdgeOperations
from graphiti_core.driver.kuzu.operations.graph_ops import KuzuGraphMaintenanceOperations
from graphiti_core.driver.kuzu.operations.has_episode_edge_ops import KuzuHasEpisodeEdgeOperations
from graphiti_core.driver.kuzu.operations.next_episode_edge_ops import (
    KuzuNextEpisodeEdgeOperations,
)
from graphiti_core.driver.kuzu.operations.saga_node_ops import KuzuSagaNodeOperations
from graphiti_core.driver.kuzu.operations.search_ops import KuzuSearchOperations
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

logger = logging.getLogger(__name__)

# Kuzu requires an explicit schema.
# As Kuzu currently does not support creating full text indexes on edge properties,
# we work around this by representing (n:Entity)-[:RELATES_TO]->(m:Entity) as
# (n)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m).
SCHEMA_QUERIES = """
    CREATE NODE TABLE IF NOT EXISTS Episodic (
        uuid STRING PRIMARY KEY,
        name STRING,
        group_id STRING,
        created_at TIMESTAMP,
        source STRING,
        source_description STRING,
        content STRING,
        valid_at TIMESTAMP,
        entity_edges STRING[]
    );
    CREATE NODE TABLE IF NOT EXISTS Entity (
        uuid STRING PRIMARY KEY,
        name STRING,
        group_id STRING,
        labels STRING[],
        created_at TIMESTAMP,
        name_embedding FLOAT[],
        summary STRING,
        attributes STRING
    );
    CREATE NODE TABLE IF NOT EXISTS Community (
        uuid STRING PRIMARY KEY,
        name STRING,
        group_id STRING,
        created_at TIMESTAMP,
        name_embedding FLOAT[],
        summary STRING
    );
    CREATE NODE TABLE IF NOT EXISTS RelatesToNode_ (
        uuid STRING PRIMARY KEY,
        group_id STRING,
        created_at TIMESTAMP,
        name STRING,
        fact STRING,
        fact_embedding FLOAT[],
        episodes STRING[],
        expired_at TIMESTAMP,
        valid_at TIMESTAMP,
        invalid_at TIMESTAMP,
        attributes STRING
    );
    CREATE REL TABLE IF NOT EXISTS RELATES_TO(
        FROM Entity TO RelatesToNode_,
        FROM RelatesToNode_ TO Entity
    );
    CREATE REL TABLE IF NOT EXISTS MENTIONS(
        FROM Episodic TO Entity,
        uuid STRING PRIMARY KEY,
        group_id STRING,
        created_at TIMESTAMP
    );
    CREATE REL TABLE IF NOT EXISTS HAS_MEMBER(
        FROM Community TO Entity,
        FROM Community TO Community,
        uuid STRING,
        group_id STRING,
        created_at TIMESTAMP
    );
    CREATE NODE TABLE IF NOT EXISTS Saga (
        uuid STRING PRIMARY KEY,
        name STRING,
        group_id STRING,
        created_at TIMESTAMP
    );
    CREATE REL TABLE IF NOT EXISTS HAS_EPISODE(
        FROM Saga TO Episodic,
        uuid STRING,
        group_id STRING,
        created_at TIMESTAMP
    );
    CREATE REL TABLE IF NOT EXISTS NEXT_EPISODE(
        FROM Episodic TO Episodic,
        uuid STRING,
        group_id STRING,
        created_at TIMESTAMP
    );
"""


class KuzuDriver(GraphDriver):
    provider: GraphProvider = GraphProvider.KUZU
    aoss_client: None = None

    def __init__(
        self,
        db: str = ':memory:',
        max_concurrent_queries: int = 1,
    ):
        super().__init__()
        self.db = kuzu.Database(db)

        self.setup_schema()

        self.client = kuzu.AsyncConnection(self.db, max_concurrent_queries=max_concurrent_queries)

        # Instantiate Kuzu operations
        self._entity_node_ops = KuzuEntityNodeOperations()
        self._episode_node_ops = KuzuEpisodeNodeOperations()
        self._community_node_ops = KuzuCommunityNodeOperations()
        self._saga_node_ops = KuzuSagaNodeOperations()
        self._entity_edge_ops = KuzuEntityEdgeOperations()
        self._episodic_edge_ops = KuzuEpisodicEdgeOperations()
        self._community_edge_ops = KuzuCommunityEdgeOperations()
        self._has_episode_edge_ops = KuzuHasEpisodeEdgeOperations()
        self._next_episode_edge_ops = KuzuNextEpisodeEdgeOperations()
        self._search_ops = KuzuSearchOperations()
        self._graph_ops = KuzuGraphMaintenanceOperations()

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

    async def execute_query(
        self, cypher_query_: str, **kwargs: Any
    ) -> tuple[list[dict[str, Any]] | list[list[dict[str, Any]]], None, None]:
        params = {k: v for k, v in kwargs.items() if v is not None}
        # Kuzu does not support these parameters.
        params.pop('database_', None)
        params.pop('routing_', None)

        try:
            results = await self.client.execute(cypher_query_, parameters=params)
        except Exception as e:
            params = {k: (v[:5] if isinstance(v, list) else v) for k, v in params.items()}
            logger.error(f'Error executing Kuzu query: {e}\n{cypher_query_}\n{params}')
            raise

        if not results:
            return [], None, None

        if isinstance(results, list):
            dict_results = [list(result.rows_as_dict()) for result in results]
        else:
            dict_results = list(results.rows_as_dict())
        return dict_results, None, None  # type: ignore

    def session(self, _database: str | None = None) -> GraphDriverSession:
        return KuzuDriverSession(self)

    async def close(self):
        # Do not explicitly close the connection, instead rely on GC.
        pass

    def delete_all_indexes(self, database_: str):
        pass

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        # Kuzu doesn't support dynamic index creation like Neo4j or FalkorDB
        # Schema and indices are created during setup_schema()
        # This method is required by the abstract base class but is a no-op for Kuzu
        pass

    def setup_schema(self):
        conn = kuzu.Connection(self.db)
        conn.execute(SCHEMA_QUERIES)
        conn.close()


class KuzuDriverSession(GraphDriverSession):
    provider = GraphProvider.KUZU

    def __init__(self, driver: KuzuDriver):
        self.driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # No cleanup needed for Kuzu, but method must exist.
        pass

    async def close(self):
        # Do not close the session here, as we're reusing the driver connection.
        pass

    async def execute_write(self, func, *args, **kwargs):
        # Directly await the provided async function with `self` as the transaction/session
        return await func(self, *args, **kwargs)

    async def run(self, query: str | list, **kwargs: Any) -> Any:
        if isinstance(query, list):
            for cypher, params in query:
                await self.driver.execute_query(cypher, **params)
        else:
            await self.driver.execute_query(query, **kwargs)
        return None
