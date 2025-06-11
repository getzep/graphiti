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
from datetime import datetime
from typing import Any

from falkordb import Graph as FalkorGraph
from falkordb.asyncio import FalkorDB
from neo4j import AsyncGraphDatabase

from graphiti_core.helpers import DEFAULT_DATABASE

logger = logging.getLogger(__name__)


class GraphClientSession(ABC):
    @abstractmethod
    async def run(self, cypher_query_: str | list, **kwargs: Any) -> Any:
        raise NotImplementedError()
    
    @abstractmethod
    async def __aenter__(self):
        raise NotImplementedError()
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError()
    
    @abstractmethod
    async def close(self):
        raise NotImplementedError()
    
    @abstractmethod
    async def execute_write(self, func, *args, **kwargs):
        raise NotImplementedError()


class FalkorClientSession(GraphClientSession):

    def __init__(self, graph: FalkorGraph):
        self.graph = graph

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # No cleanup needed for Falkor, but method must exist
        pass

    async def close(self):
        # No explicit close needed for FalkorDB, but method must exist
        pass

    async def execute_write(self, func, *args, **kwargs):
        # Directly await the provided async function with `self` as the transaction/session
        return await func(self, *args, **kwargs)
    
    async def run(self, cypher_query_: str | list, **kwargs: Any) -> Any:
        # FalkorDB does not support argument for Label Set, so it's converted into an array of queries
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
    async def execute_query(self, cypher_query_: str, **kwargs: Any) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def session(self, database: str) -> GraphClientSession:
        raise NotImplementedError()

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    async def delete_all_indexes(self, database_: str | None = DEFAULT_DATABASE) -> Any:
        raise NotImplementedError()

class Neo4jClient(GraphClient):

    def __init__(
        self,
        uri: str,
        user: str | None,
        password: str | None,
    ):
        super().__init__()
        self.client = AsyncGraphDatabase.driver(
            uri=uri,
            auth=(user, password) if user and password else None,
        )

    async def execute_query(self, cypher_query_: str, **kwargs: Any) -> Any:
        params = kwargs.pop("params", None)
        result = await self.client.execute_query(cypher_query_, parameters_=params, **kwargs)
        
        return result

    def session(self, database: str) -> GraphClientSession:
        return self.client.session(database=database)  # type: ignore

    async def close(self) -> None:
        await self.client.close()
    
    async def delete_all_indexes(self, database_: str | None = DEFAULT_DATABASE) -> Any:
        return await self.client.execute_query(
            "CALL db.indexes() YIELD name DROP INDEX name",
            {},
            database=database_,
        )

class FalkorClient(GraphClient):
    def __init__(
        self,
        uri: str,
        user: str | None,
        password: str | None,
    ):
        super().__init__()
        if user and password:
            uri_parts = uri.split("://", 1)
            uri = f"{uri_parts[0]}://{user}:{password}@{uri_parts[1]}"
            
        self.client = FalkorDB.from_url(
            url=uri,
        )

    def _get_graph(self, graph_name: str) -> FalkorGraph:
        # FalkorDB requires a non-None database name for multi-tenant graphs; the default is "DEFAULT_DATABASE"
        if graph_name is None:
            graph_name = "DEFAULT_DATABASE"
        return self.client.select_graph(graph_name)

    async def execute_query(self, cypher_query_: str, **kwargs: Any) -> Any:
        graph_name = kwargs.pop("database_", DEFAULT_DATABASE)
        graph = self._get_graph(graph_name)

        # Convert datetime objects to ISO strings (FalkorDB does not support datetime objects directly)
        params = convert_datetimes_to_strings(dict(kwargs))

        try:
            result = await graph.query(cypher_query_, params)
        except Exception as e:
            if "already indexed" in str(e):
                # check if index already exists
                logger.info(f"Index already exists: {e}")
                return None
            logger.error(f"Error executing FalkorDB query: {e}")
            raise

        # Convert the result header to a list of strings
        header = [h[1].decode('utf-8') for h in result.header]
        return (result.result_set, header, None)

    def session(self, database: str) -> GraphClientSession:
        return FalkorClientSession(self._get_graph(database))

    async def close(self) -> None:
        await self.client.connection.close()
    
    async def delete_all_indexes(self, database_: str | None = DEFAULT_DATABASE) -> Any:
        return await self.client.execute_query(
            "CALL db.indexes() YIELD name DROP INDEX name",
            database_,
        )


class Driver:

    _driver: GraphClient

    def __init__(
        self,
        uri: str,
        user: str | None,
        password: str | None,
    ):
        if uri.startswith("falkor"):
            # FalkorDB
            self._driver = FalkorClient(uri, user, password)
            self.provider = "falkordb"
        else:
            # Neo4j
            self._driver = Neo4jClient(uri, user, password)
            self.provider = "neo4j"
        
    async def execute_query(self, cypher_query_, **kwargs: Any) -> Any:
        return await self._driver.execute_query(cypher_query_, **kwargs)

    async def close(self):
        return await self._driver.close()

    async def delete_all_indexes(self, database_: str | None = DEFAULT_DATABASE) -> Any:
        return await self._driver.delete_all_indexes(database_)
    
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