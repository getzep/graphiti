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
from datetime import datetime
from typing import TYPE_CHECKING, Any

from typing_extensions import LiteralString

if TYPE_CHECKING:
    from falkordb import Graph as FalkorGraph
    from falkordb.asyncio import FalkorDB
else:
    try:
        from falkordb import Graph as FalkorGraph
        from falkordb.asyncio import FalkorDB
    except ImportError:
        # If falkordb is not installed, raise an ImportError
        raise ImportError(
            'falkordb is required for FalkorDriver. '
            'Install it with: pip install graphiti-core[falkordb]'
        ) from None

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.graph_queries import get_fulltext_indices, get_range_indices
from graphiti_core.helpers import semaphore_gather

logger = logging.getLogger(__name__)


class FalkorDriverSession(GraphDriverSession):
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

    async def run(self, query: str | list, **kwargs: Any) -> Any:
        # FalkorDB does not support argument for Label Set, so it's converted into an array of queries
        if isinstance(query, list):
            for cypher, params in query:
                params = convert_datetimes_to_strings(params)
                await self.graph.query(str(cypher), params)  # type: ignore[reportUnknownArgumentType]
        else:
            params = dict(kwargs)
            params = convert_datetimes_to_strings(params)
            await self.graph.query(str(query), params)  # type: ignore[reportUnknownArgumentType]
        # Assuming `graph.query` is async (ideal); otherwise, wrap in executor
        return None


class FalkorDriver(GraphDriver):
    provider = GraphProvider.FALKORDB
    default_group_id: str = '\\_'
    fulltext_syntax: str = '@'  # FalkorDB uses a redisearch-like syntax for fulltext queries

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        username: str | None = None,
        password: str | None = None,
        falkor_db: FalkorDB | None = None,
        database: str = '\\_',
    ):
        """
        Initialize the FalkorDB driver.

        FalkorDB is a multi-tenant graph database.
        To connect, provide the host and port.
        The default parameters assume a local (on-premises) FalkorDB instance.
        """
        super().__init__()

        self._database = database
        if falkor_db is not None:
            # If a FalkorDB instance is provided, use it directly
            self.client = falkor_db
        else:
            self.client = FalkorDB(host=host, port=port, username=username, password=password)

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

    def _get_graph(self, graph_name: str | None) -> FalkorGraph:
        # FalkorDB requires a non-None database name for multi-tenant graphs; the default is "default_db"
        if graph_name is None:
            graph_name = self._database
        return self.client.select_graph(graph_name)

    async def execute_query(self, cypher_query_, **kwargs: Any):
        graph = self._get_graph(self._database)

        # Convert datetime objects to ISO strings (FalkorDB does not support datetime objects directly)
        params = convert_datetimes_to_strings(dict(kwargs))

        try:
            result = await graph.query(cypher_query_, params)  # type: ignore[reportUnknownArgumentType]
        except Exception as e:
            if 'already indexed' in str(e):
                # check if index already exists
                logger.info(f'Index already exists: {e}')
                return None
            logger.error(f'Error executing FalkorDB query: {e}\n{cypher_query_}\n{params}')
            raise

        # Convert the result header to a list of strings
        header = [h[1] for h in result.header]

        # Convert FalkorDB's result format (list of lists) to the format expected by Graphiti (list of dicts)
        records = []
        for row in result.result_set:
            record = {}
            for i, field_name in enumerate(header):
                if i < len(row):
                    record[field_name] = row[i]
                else:
                    # If there are more fields in header than values in row, set to None
                    record[field_name] = None
            records.append(record)

        return records, header, None

    def session(self, database: str | None = None) -> GraphDriverSession:
        return FalkorDriverSession(self._get_graph(database))

    async def close(self) -> None:
        """Close the driver connection."""
        if hasattr(self.client, 'aclose'):
            await self.client.aclose()  # type: ignore[reportUnknownMemberType]
        elif hasattr(self.client.connection, 'aclose'):
            await self.client.connection.aclose()
        elif hasattr(self.client.connection, 'close'):
            await self.client.connection.close()

    async def delete_all_indexes(self) -> None:
        from collections import defaultdict
        
        result = await self.execute_query('CALL db.indexes()')
        if result is None:
            return
        
        records, _, _ = result
        
        # Organize indexes by type and label
        range_indexes = defaultdict(list)
        fulltext_indexes = defaultdict(list)
        entity_types = {}
        
        for record in records:
            label = record['label']
            entity_types[label] = record['entitytype']
            
            for field_name, index_type in record['types'].items():
                if 'RANGE' in index_type:
                    range_indexes[label].append(field_name)
                if 'FULLTEXT' in index_type:
                    fulltext_indexes[label].append(field_name)
        
        # Drop all range indexes
        for label, fields in range_indexes.items():
            for field in fields:
                await self.execute_query(f'DROP INDEX ON :{label}({field})')

        # Drop all fulltext indexes
        for label, fields in fulltext_indexes.items():
            entity_type = entity_types[label]
            for field in fields:
                if entity_type == 'NODE':
                    await self.execute_query(
                        f'DROP FULLTEXT INDEX FOR (n:{label}) ON (n.{field})'
                    )
                elif entity_type == 'RELATIONSHIP':
                    await self.execute_query(
                        f'DROP FULLTEXT INDEX FOR ()-[e:{label}]-() ON (e.{field})'
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

    def clone(self, database: str) -> 'GraphDriver':
        """
        Returns a shallow copy of this driver with a different default database.
        Reuses the same connection (e.g. FalkorDB, Neo4j).
        """
        if database == self._database:
            cloned = self
        else:
            # Create a new instance of FalkorDriver with the same connection but a different database
            cloned = FalkorDriver(falkor_db=self.client, database=database)
        
        return cloned


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
