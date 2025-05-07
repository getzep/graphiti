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

from neo4j import AsyncGraphDatabase
from falkordb import FalkorDB, Graph as FalkorGraph

from graphiti_core.helpers import DEFAULT_DATABASE

logger = logging.getLogger(__name__)


class GraphClientSession(ABC):

    @abstractmethod
    def run(self, query: str, **kwargs: any) -> Any:
        raise NotImplementedError()


class FalkorClientSession(GraphClientSession):

    def __init__(self, graph: FalkorGraph):
        self.graph = graph

    def run(self, cypher_query_: str, **kwargs: any) -> Coroutine:
        return self.graph.query(str(cypher_query_), dict(kwargs))


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
        return self._client.execute_query(cypher_query_, **kwargs)

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
        self.client = FalkorDB(
            url=uri,
            username=user,
            password=password,
        )

    def _get_graph(self, graph_name: str) -> FalkorGraph:
        return self.client.select_graph(graph_name)

    def execute_query(self, cypher_query_, **kwargs: Any) -> Coroutine:
        future = Future()
        graph_name = kwargs.pop("database_", DEFAULT_DATABASE)
        final_query = str(cypher_query_)
        if "db.index.fulltext.queryRelationships" in final_query:
            final_query = "RETURN []"
        result = self._get_graph(graph_name).query(final_query, dict(kwargs))
        future.set_result((result.result_set[0], None, None))
        return future

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
        self._driver = (
            FalkorClient(uri, user, password)
            if "falkor" in uri
            else Neo4jClient(uri, user, password)
        )

    def execute_query(self, cypher_query_, **kwargs: Any) -> Coroutine:
        return self._driver.execute_query(cypher_query_, **kwargs)

    def close(self):
        return self._driver.close()

    def delete_all_indexes(self, database_: str = DEFAULT_DATABASE) -> Coroutine:
        return self._driver.delete_all_indexes(database_)

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
