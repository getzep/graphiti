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
from asyncio import sleep, Future
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
        if isinstance(cypher_query_, list):
            for query in cypher_query_:
                params = query[1]
                query = query[0]
                params = convert_datetimes_to_strings(params)
                await self.graph.query(str(query), params)
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

    @abstractmethod
    def delete_all_indexes(self, database_: str = DEFAULT_DATABASE) -> Coroutine:
        raise NotImplementedError()

    @abstractmethod
    def create_node_index(
        self,
        label: str,
        property: str,
        index_name: str | None = None,
        database_: str = DEFAULT_DATABASE,
    ) -> Coroutine:
        raise NotImplementedError()

    @abstractmethod
    def create_relationship_index(
        self,
        label: str,
        property: str,
        index_name: str | None = None,
        database_: str = DEFAULT_DATABASE,
    ) -> Coroutine:
        raise NotImplementedError()

    @abstractmethod
    def create_node_fulltext_index(
        self,
        label: str,
        properties: list[str],
        index_name: str | None = None,
        database_: str = DEFAULT_DATABASE,
    ) -> Coroutine:
        raise NotImplementedError()

    @abstractmethod
    def create_relationship_fulltext_index(
        self,
        label: str,
        properties: list[str],
        index_name: str | None = None,
        database_: str = DEFAULT_DATABASE,
    ) -> Coroutine:
        raise NotImplementedError()


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
        try:
            result = self._client.execute_query(cypher_query_, parameters_=params, **kwargs)
        except Exception as e:
            print(f"Error executing query: {e}")
        return result

    def session(self, database: str) -> GraphClientSession:
        return self._client.session(database=database)  # type: ignore

    def close(self) -> None:
        return self._client.close()

    def delete_all_indexes(self, database_: str = DEFAULT_DATABASE) -> Coroutine:
        return self._client.execute_query(
            "CALL db.indexes() YIELD name DROP INDEX name",
            database_,
        )

    def create_node_index(
        self,
        label: str,
        property: str,
        index_name: str | None = None,
        database_: str = DEFAULT_DATABASE,
    ) -> Coroutine:
        return self._client.execute_query(
            f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{label}) ON (n.{property})",
            database_,
        )

    def create_relationship_index(
        self,
        label: str,
        property: str,
        index_name: str | None = None,
        database_: str = DEFAULT_DATABASE,
    ) -> Coroutine:
        return self._client.execute_query(
            f"CREATE INDEX {index_name} IF NOT EXISTS FOR ()-[e:{label}]-() ON (e.{property})",
            database_,
        )

    def create_node_fulltext_index(
        self,
        label: str,
        properties: list[str],
        index_name: str | None = None,
        database_: str = DEFAULT_DATABASE,
    ) -> Coroutine:
        return self._client.execute_query(
            f"CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS FOR (n:{label}) ON EACH [{', n.'.join(properties)}]",
            database_,
        )

    def create_relationship_fulltext_index(
        self,
        label: str,
        properties: list[str],
        index_name: str | None = None,
        database_: str = DEFAULT_DATABASE,
    ) -> Coroutine:
        return self._client.execute_query(
            f"CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS FOR ()-[e:{label}]-() ON EACH [{', e.'.join(properties)}]",
            database_,
        )


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
        return self.client.select_graph(graph_name)

    async def execute_query(self, cypher_query_, **kwargs: Any) -> Coroutine:
        # future = Future()
        graph_name = kwargs.pop("database_", DEFAULT_DATABASE)
        graph = self.client.select_graph(graph_name)

        params = convert_datetimes_to_strings(dict(kwargs))
        try:
            result = await graph.query(cypher_query_, params)
        except Exception as e:
            if "indexed" in str(e):
                # check if index already exists
                print(f"Index already exists: {e}")
                return None
            return None
        # pdb.set_trace()
        # future.set_result((result.result_set, None, None))
        return (result.result_set, result.header, None)

    def session(self, database: str) -> GraphClientSession:
        return FalkorClientSession(self._get_graph(database))

    def close(self) -> None:
        self.client.connection.close()

    def delete_all_indexes(self, database_: str = DEFAULT_DATABASE) -> Coroutine:
        return

    def create_node_index(
        self,
        label: str,
        property: str,
        _: str | None = None,
        database_: str = DEFAULT_DATABASE,
    ) -> Coroutine:
        future = Future()
        try:
            print(f"Creating index for {label} on {property}")
            return self._get_graph(database_).query(
                f"CREATE INDEX FOR (n:{label}) ON (n.{property})",
            )
        except Exception as e:
            # check if index already exists
            if "already indexed" in str(e):
                future.set_result(None)
                return future
            raise e

    def create_relationship_index(
        self,
        label: str,
        property: str,
        _: str | None = None,
        database_: str = DEFAULT_DATABASE,
    ) -> Coroutine:
        future = Future()
        try:
            print(f"Creating index for {label} on {property}")
            return self._get_graph(database_).query(
                f"CREATE INDEX FOR ()-[e:{label}]-() ON (e.{property})",
            )
        except Exception as e:
            # check if index already exists
            if "already indexed" in str(e):
                future.set_result(None)
                return future
            raise e

    def create_node_fulltext_index(
        self,
        label: str,
        properties: list[str],
        _: str | None = None,
        database_: str = DEFAULT_DATABASE,
    ) -> Coroutine:
        future = Future()
        try:
            print(f"Creating fulltext index for {label} on {properties}")
            return self._get_graph(database_).query(
                f"CALL db.idx.fulltext.createNodeIndex('{label}', '{'\', \''.join(properties)}')",
            )
        except Exception as e:
            # check if index already exists
            if "already indexed" in str(e):
                future.set_result(None)
                return future

    def create_relationship_fulltext_index(
        self,
        label: str,
        properties: list[str],
        _: str | None = None,
        database_: str = DEFAULT_DATABASE,
    ) -> Coroutine:
        print(f"Creating fulltext index for {label} on {properties}")
        future = Future()
        future.set_result(None)
        return future


class Driver:

    _driver: GraphClient

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
    ):
        # pdb.set_trace()
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

    def close(self):
        return self._driver.close()

    def delete_all_indexes(self, database_: str = DEFAULT_DATABASE) -> Coroutine:
        return self._driver.delete_all_indexes(database_)
    
    def session(self, database: str) -> GraphClientSession:
        return self._driver.session(database)

    def create_node_index(
        self,
        label: str,
        property: str,
        index_name: str | None = None,
        database_: str = DEFAULT_DATABASE,
    ) -> Coroutine:
        return self._driver.create_node_index(label, property, index_name, database_)

    def create_relationship_index(
        self,
        label: str,
        property: str,
        index_name: str | None = None,
        database_: str = DEFAULT_DATABASE,
    ) -> Coroutine:
        return self._driver.create_relationship_index(
            label, property, index_name, database_
        )

    def create_node_fulltext_index(
        self,
        label: str,
        properties: list[str],
        index_name: str | None = None,
        database_: str = DEFAULT_DATABASE,
    ) -> Coroutine:
        return self._driver.create_node_fulltext_index(
            label, properties, index_name, database_
        )

    def create_relationship_fulltext_index(
        self,
        label: str,
        properties: list[str],
        index_name: str | None = None,
        database_: str = DEFAULT_DATABASE,
    ) -> Coroutine:
        return self._driver.create_relationship_fulltext_index(
            label, properties, index_name, database_
        )

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