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
from typing import Any, Coroutine
from datetime import datetime

from neo4j import AsyncGraphDatabase, GraphDatabase
from falkordb.asyncio import FalkorDB
from falkordb import Graph as FalkorGraph

from graphiti_core.helpers import DEFAULT_DATABASE
import asyncio

logger = logging.getLogger(__name__)


class GraphClientSession(ABC):
    @abstractmethod
    async def run(self, query: str, **kwargs: Any) -> Any:
        raise NotImplementedError()


class FalkorClientSession(GraphClientSession):

    def __init__(self, graph: FalkorGraph):
        self.graph = graph

    async def execute_write(self, func, *args, **kwargs):
        # Directly await the provided async function with `self` as the transaction/session
        return await func(self, *args, **kwargs)
    async def run(self, cypher_query_: str|list, **kwargs: Any) -> Any:
        # FalkorDB does not support argument for Label Set, so its converted into array of queries
        if isinstance(cypher_query_, list):
            for cypher, params in cypher_query_:
                params = convert_datetimes_to_strings(params)
                await self.graph.query(str(cypher), params)
        else:
            params = dict(kwargs)
            params = convert_datetimes_to_strings(params)
            await self.graph.query(str(cypher_query_), params)
        # Assuming `graph.query` is async (ideal); otherwise, wrap in executor
        return None


class GraphClient(ABC):

    @abstractmethod
    def execute_query(self, cypher_query_: str, **kwargs: Any) -> Coroutine:
        raise NotImplementedError()

    @abstractmethod
    def session(self, database: str) -> GraphClientSession:
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError()

    def delete_all_indexes(self, database_: str = DEFAULT_DATABASE) -> Coroutine:
        return self._client.execute_query(
            "CALL db.indexes() YIELD name DROP INDEX name",
            database_,
        )

class Neo4jClient(GraphClient):

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
    ):
        super().__init__()
        self._client = AsyncGraphDatabase.driver(
            uri=uri,
            auth=(user, password),
        )

    def execute_query(self, cypher_query_: str, **kwargs: Any) -> Coroutine:
        params = kwargs.pop("params", None)
        result = self._client.execute_query(cypher_query_, parameters_=params, **kwargs)
        
        return result

    def session(self, database: str) -> GraphClientSession:
        return self._client.session(database=database)  # type: ignore

    def close(self) -> None:
        return self._client.close()

class FalkorClient(GraphClient):
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
    ):
        super().__init__()
        self.client = FalkorDB.from_url(
            url=uri,
            # username=user,
            # password=password,
        )

    def _get_graph(self, graph_name: str) -> FalkorGraph:
        # FalkorDB requires a non-None database name for multi-tenant graphs; the default is "DEFAULT_DATABASE"
        if graph_name is None:
            graph_name = "DEFAULT_DATABASE"
        return self.client.select_graph(graph_name)

    async def execute_query(self, cypher_query_, **kwargs: Any) -> Coroutine:
        graph_name = kwargs.pop("database_", DEFAULT_DATABASE)
        graph = self._get_graph(graph_name)

        # Convert datetime objects to ISO strings (FalkorDB does not support datetime objects directly)
        params = convert_datetimes_to_strings(dict(kwargs))

        try:
            result = await graph.query(cypher_query_, params)
        except Exception as e:
            if "already indexed" in str(e):
                # check if index already exists
                print(f"Index already exists: {e}")
                return None
            return None

        # Convert the result header to a list of strings
        header = [h[1].decode('utf-8') for h in result.header]
        return (result.result_set, header, None)

    def session(self, database: str) -> GraphClientSession:
        return FalkorClientSession(self._get_graph(database))

    def close(self) -> None:
        self.client.connection.close()


class Driver:

    _driver: GraphClient

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
    ):
        if "falkor" in uri:
            # FalkorDB
            self._driver = FalkorClient(uri, user, password)
            self.provider = "falkordb"
        else:
            # Neo4j
            self._driver = Neo4jClient(uri, user, password)
            self.provider = "neo4j"
        
    def execute_query(self, cypher_query_, **kwargs: Any) -> Coroutine:
        return self._driver.execute_query(cypher_query_, **kwargs)

    async def close(self):
        return self._driver.close()

    def delete_all_indexes(self, database_: str = DEFAULT_DATABASE) -> Coroutine:
        return self._driver.delete_all_indexes(database_)
    
    def session(self, database: str) -> GraphClientSession:
        return self._driver.session(database)

def convert_datetimes_to_strings(obj):
    if isinstance(obj, dict):
        return {k: convert_datetimes_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetimes_to_strings(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_datetimes_to_strings(item) for item in obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj