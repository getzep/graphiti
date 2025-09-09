"""
Copyright 2025, Zep Software, Inc.

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

from neo4j import AsyncGraphDatabase
from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider

logger = logging.getLogger(__name__)


class MemgraphDriver(GraphDriver):
    provider = GraphProvider.MEMGRAPH

    def __init__(
        self, uri: str, user: str | None, password: str | None, database: str = 'memgraph'
    ):
        super().__init__()
        self.client = AsyncGraphDatabase.driver(
            uri=uri,
            auth=(user or '', password or ''),
        )
        self._database = database

    async def execute_query(
        self, cypher_query_: LiteralString, **kwargs: Any
    ) -> tuple[list, Any, Any]:
        """
        Execute a Cypher query against Memgraph using implicit transactions.
        Returns a tuple of (records, summary, keys) for compatibility with the GraphDriver interface.
        """
        # Extract parameters from kwargs
        params = kwargs.pop('params', None)
        if params is None:
            # If no 'params' key, use the remaining kwargs as parameters
            # but first extract database-specific parameters
            database = kwargs.pop('database_', self._database)
            kwargs.pop('parameters_', None)  # Remove if present (Neo4j async driver param)

            # All remaining kwargs are query parameters
            params = kwargs
        else:
            # Extract database parameter if params was provided separately
            database = kwargs.pop('database_', self._database)
            kwargs.pop('parameters_', None)  # Remove if present

        async with self.client.session(database=database) as session:
            try:
                result = await session.run(cypher_query_, params)
                records = [record async for record in result]
                summary = await result.consume()
                keys = result.keys()
                return (records, summary, keys)
            except Exception as e:
                logger.error(f'Error executing Memgraph query: {e}\n{cypher_query_}\n{params}')
                raise

    def session(self, database: str | None = None) -> GraphDriverSession:
        _database = database or self._database
        return self.client.session(database=_database)  # type: ignore

    async def close(self) -> None:
        return await self.client.close()

    def delete_all_indexes(self) -> Coroutine[Any, Any, Any]:
        # TODO: Implement index deletion for Memgraph
        raise NotImplementedError('Index deletion not implemented for MemgraphDriver')
