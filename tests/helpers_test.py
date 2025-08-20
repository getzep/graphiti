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
from graphiti_core.driver.neptune_driver import NeptuneDriver
from graphiti_core.helpers import lucene_sanitize
from graphiti_core.nodes import EntityNode

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

if os.getenv('DISABLE_NEPTUNE') is None:
    try:
        from graphiti_core.driver.neptune_driver import NeptuneDriver

        drivers.append(GraphProvider.NEPTUNE)
    except ImportError:
        pass

NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'test')

FALKORDB_HOST = os.getenv('FALKORDB_HOST', 'localhost')
FALKORDB_PORT = os.getenv('FALKORDB_PORT', '6379')
FALKORDB_USER = os.getenv('FALKORDB_USER', None)
FALKORDB_PASSWORD = os.getenv('FALKORDB_PASSWORD', None)

NEPTUNE_HOST = os.getenv('NEPTUNE_HOST', 'localhost')
NEPTUNE_PORT = os.getenv('NEPTUNE_PORT', 8182)
AOSS_HOST = os.getenv('AOSS_HOST', None)

KUZU_DB = os.getenv('KUZU_DB', ':memory:')

group_id = 'graphiti_test_group'
group_id_2 = 'graphiti_test_group_2'


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
    elif provider == 'neptune':
        return NeptuneDriver(
            host=NEPTUNE_HOST,
            port=int(NEPTUNE_PORT),
            aoss_host=AOSS_HOST,
        )
    else:
        raise ValueError(f'Driver {provider} not available')


async def clean_up(graph_driver):
    await EntityNode.delete_by_group_id(graph_driver, group_id)
    await EntityNode.delete_by_group_id(graph_driver, group_id_2)


@pytest.fixture(params=drivers)
async def graph_driver(request):
    driver = request.param
    graph_driver = get_driver(driver)
    await clean_up(graph_driver)
    try:
        yield graph_driver  # provide driver to the test
    finally:
        # always called, even if the test fails or raises
        # await clean_up(graph_driver)
        await graph_driver.close()


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
