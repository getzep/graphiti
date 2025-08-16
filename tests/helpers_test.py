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

import os

import pytest
from dotenv import load_dotenv

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.helpers import lucene_sanitize

load_dotenv()

drivers: list[GraphProvider] = []
if os.getenv('DISABLE_NEO4J') is None:
    try:
        from graphiti_core.driver.neo4j_driver import Neo4jDriver

        drivers.append(GraphProvider.NEO4J)
    except ImportError:
        pass

if os.getenv('DISABLE_FALKORDB') is None:
    try:
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        drivers.append(GraphProvider.FALKORDB)
    except ImportError:
        pass

if os.getenv('DISABLE_KUZU') is None:
    try:
        from graphiti_core.driver.kuzu_driver import KuzuDriver

        drivers.append(GraphProvider.KUZU)
    except ImportError:
        raise


NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'test')

FALKORDB_HOST = os.getenv('FALKORDB_HOST', 'localhost')
FALKORDB_PORT = os.getenv('FALKORDB_PORT', '6379')
FALKORDB_USER = os.getenv('FALKORDB_USER', None)
FALKORDB_PASSWORD = os.getenv('FALKORDB_PASSWORD', None)

KUZU_DB = os.getenv('KUZU_DB', ':memory:')


def get_driver(provider: GraphProvider) -> GraphDriver:
    if provider == GraphProvider.NEO4J:
        return Neo4jDriver(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
        )
    elif provider == GraphProvider.FALKORDB:
        return FalkorDriver(
            host=FALKORDB_HOST,
            port=int(FALKORDB_PORT),
            username=FALKORDB_USER,
            password=FALKORDB_PASSWORD,
        )
    elif provider == GraphProvider.KUZU:
        driver = KuzuDriver(
            db=KUZU_DB,
        )
        return driver
    else:
        raise ValueError(f'Driver {provider} not available')


def test_lucene_sanitize():
    # Call the function with test data
    queries = [
        (
            'This has every escape character + - && || ! ( ) { } [ ] ^ " ~ * ? : \\ /',
            '\\This has every escape character \\+ \\- \\&\\& \\|\\| \\! \\( \\) \\{ \\} \\[ \\] \\^ \\" \\~ \\* \\? \\: \\\\ \\/',
        ),
        ('this has no escape characters', 'this has no escape characters'),
    ]

    for query, assert_result in queries:
        result = lucene_sanitize(query)
        assert assert_result == result


async def get_node_count(driver: GraphDriver, uuids: list[str]) -> int:
    results, _, _ = await driver.execute_query(
        """
        MATCH (n)
        WHERE n.uuid IN $uuids
        RETURN COUNT(n) as count
        """,
        uuids=uuids,
    )
    return int(results[0]['count'])


async def get_edge_count(driver: GraphDriver, uuids: list[str]) -> int:
    results, _, _ = await driver.execute_query(
        """
        MATCH (n)-[e]->(m)
        WHERE e.uuid IN $uuids
        RETURN COUNT(e) as count
        UNION ALL
        MATCH (n)-[:RELATES_TO]->(e)-[:RELATES_TO]->(m)
        WHERE e.uuid IN $uuids
        RETURN COUNT(e) as count
        """,
        uuids=uuids,
    )
    return sum(int(result['count']) for result in results)


if __name__ == '__main__':
    pytest.main([__file__])
