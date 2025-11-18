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
from collections.abc import Coroutine
from typing import Any

from neo4j import AsyncGraphDatabase, EagerResult
from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
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

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        if delete_existing:
            await self.delete_all_indexes()

        range_indices: list[LiteralString] = get_range_indices(self.provider)

        fulltext_indices: list[LiteralString] = get_fulltext_indices(self.provider)

        index_queries: list[LiteralString] = range_indices + fulltext_indices

        await semaphore_gather(
            *[
                self.execute_query(
                    query,
                )
                for query in index_queries
            ]
        )

    async def health_check(self) -> None:
        """Check Neo4j connectivity by running the driver's verify_connectivity method."""
        try:
            await self.client.verify_connectivity()
            return None
        except Exception as e:
            print(f'Neo4j health check failed: {e}')
            raise
