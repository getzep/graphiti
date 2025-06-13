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
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from typing import Any

from graphiti_core.helpers import DEFAULT_DATABASE

logger = logging.getLogger(__name__)


class GraphDriverSession(ABC):
    @abstractmethod
    async def run(self, query: str, **kwargs: Any) -> Any:
        raise NotImplementedError()


class GraphDriver(ABC):
    provider: str

    @abstractmethod
    def execute_query(self, cypher_query_: str, **kwargs: Any) -> Coroutine:
        raise NotImplementedError()

    @abstractmethod
    def session(self, database: str) -> GraphDriverSession:
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def delete_all_indexes(self, database_: str = DEFAULT_DATABASE) -> Coroutine:
        raise NotImplementedError()


# class GraphDriver:
#     _driver: GraphClient
#
#     def __init__(
#             self,
#             uri: str,
#             user: str,
#             password: str,
#     ):
#         if uri.startswith('falkor'):
#             # FalkorDB
#             self._driver = FalkorClient(uri, user, password)
#             self.provider = 'falkordb'
#         else:
#             # Neo4j
#             self._driver = Neo4jClient(uri, user, password)
#             self.provider = 'neo4j'
#
#     def execute_query(self, cypher_query_, **kwargs: Any) -> Coroutine:
#         return self._driver.execute_query(cypher_query_, **kwargs)
#
#     async def close(self):
#         return await self._driver.close()
#
#     def delete_all_indexes(self, database_: str = DEFAULT_DATABASE) -> Coroutine:
#         return self._driver.delete_all_indexes(database_)
#
#     def session(self, database: str) -> GraphClientSession:
#         return self._driver.session(database)
