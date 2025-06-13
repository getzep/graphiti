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

from neo4j import AsyncGraphDatabase
from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession
from graphiti_core.helpers import DEFAULT_DATABASE

logger = logging.getLogger(__name__)


class Neo4jDriver(GraphDriver):
    provider: str = 'neo4j'

    def __init__(
        self,
        uri: str,
        user: str | None,
        password: str | None,
    ):
        super().__init__()
        self.client = AsyncGraphDatabase.driver(
            uri=uri,
            auth=(user or '', password or ''),
        )

    async def execute_query(self, cypher_query_: LiteralString, **kwargs: Any) -> Coroutine:
        params = kwargs.pop('params', None)
        result = await self.client.execute_query(cypher_query_, parameters_=params, **kwargs)

        return result

    def session(self, database: str) -> GraphDriverSession:
        return self.client.session(database=database)  # type: ignore

    async def close(self) -> None:
        return await self.client.close()

    def delete_all_indexes(self, database_: str = DEFAULT_DATABASE) -> Coroutine:
        return self.client.execute_query(
            'CALL db.indexes() YIELD name DROP INDEX name',
            database_=database_,
        )
