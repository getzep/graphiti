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
from unittest.mock import Mock

import numpy as np
import pytest
from dotenv import load_dotenv

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.helpers import lucene_sanitize
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

load_dotenv()

drivers: list[GraphProvider] = []
if os.getenv('DISABLE_NEO4J') is None:
    try:
        from graphiti_core.driver.neo4j_driver import Neo4jDriver

        drivers.append(GraphProvider.NEO4J)
    except ImportError:
        raise

if os.getenv('DISABLE_FALKORDB') is None:
    try:
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        drivers.append(GraphProvider.FALKORDB)
    except ImportError:
        raise

if os.getenv('DISABLE_KUZU') is None:
    try:
        from graphiti_core.driver.kuzu_driver import KuzuDriver

        drivers.append(GraphProvider.KUZU)
    except ImportError:
        raise

# Disable Neptune for now
os.environ['DISABLE_NEPTUNE'] = 'True'
if os.getenv('DISABLE_NEPTUNE') is None:
    try:
        from graphiti_core.driver.neptune_driver import NeptuneDriver

        drivers.append(GraphProvider.NEPTUNE)
    except ImportError:
        raise

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
    elif provider == GraphProvider.NEPTUNE:
        return NeptuneDriver(
            host=NEPTUNE_HOST,
            port=int(NEPTUNE_PORT),
            aoss_host=AOSS_HOST,
        )
    else:
        raise ValueError(f'Driver {provider} not available')


@pytest.fixture(params=drivers)
async def graph_driver(request):
    driver = request.param
    graph_driver = get_driver(driver)
    await clear_data(graph_driver, [group_id, group_id_2])
    try:
        yield graph_driver  # provide driver to the test
    finally:
        # always called, even if the test fails or raises
        # await clean_up(graph_driver)
        await graph_driver.close()


embedding_dim = 384
embeddings = {
    key: np.random.uniform(0.0, 0.9, embedding_dim).tolist()
    for key in [
        'Alice',
        'Bob',
        'Alice likes Bob',
        'test_entity_1',
        'test_entity_2',
        'test_entity_3',
        'test_entity_4',
        'test_entity_alice',
        'test_entity_bob',
        'test_entity_1 is a duplicate of test_entity_2',
        'test_entity_3 is a duplicate of test_entity_4',
        'test_entity_1 relates to test_entity_2',
        'test_entity_1 relates to test_entity_3',
        'test_entity_2 relates to test_entity_3',
        'test_entity_1 relates to test_entity_4',
        'test_entity_2 relates to test_entity_4',
        'test_entity_3 relates to test_entity_4',
        'test_entity_1 relates to test_entity_2',
        'test_entity_3 relates to test_entity_4',
        'test_entity_2 relates to test_entity_3',
        'test_community_1',
        'test_community_2',
    ]
}
embeddings['Alice Smith'] = embeddings['Alice']


@pytest.fixture
def mock_embedder():
    mock_model = Mock(spec=EmbedderClient)

    def mock_embed(input_data):
        if isinstance(input_data, str):
            return embeddings[input_data]
        elif isinstance(input_data, list):
            combined_input = ' '.join(input_data)
            return embeddings[combined_input]
        else:
            raise ValueError(f'Unsupported input type: {type(input_data)}')

    mock_model.create.side_effect = mock_embed
    return mock_model


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
        MATCH (e:RelatesToNode_)
        WHERE e.uuid IN $uuids
        RETURN COUNT(e) as count
        """,
        uuids=uuids,
    )
    return sum(int(result['count']) for result in results)


async def print_graph(graph_driver: GraphDriver):
    nodes, _, _ = await graph_driver.execute_query(
        """
        MATCH (n)
        RETURN n.uuid, n.name
        """,
    )
    print('Nodes:')
    for node in nodes:
        print('  ', node)
    edges, _, _ = await graph_driver.execute_query(
        """
        MATCH (n)-[e]->(m)
        RETURN n.name, e.uuid, m.name
        """,
    )
    print('Edges:')
    for edge in edges:
        print('  ', edge)


async def assert_episodic_node_equals(retrieved: EpisodicNode, sample: EpisodicNode):
    assert retrieved.uuid == sample.uuid
    assert retrieved.name == sample.name
    assert retrieved.group_id == group_id
    assert retrieved.created_at == sample.created_at
    assert retrieved.source == sample.source
    assert retrieved.source_description == sample.source_description
    assert retrieved.content == sample.content
    assert retrieved.valid_at == sample.valid_at
    assert set(retrieved.entity_edges) == set(sample.entity_edges)


async def assert_entity_node_equals(
    graph_driver: GraphDriver, retrieved: EntityNode, sample: EntityNode
):
    await retrieved.load_name_embedding(graph_driver)
    assert retrieved.uuid == sample.uuid
    assert retrieved.name == sample.name
    assert retrieved.group_id == sample.group_id
    assert set(retrieved.labels) == set(sample.labels)
    assert retrieved.created_at == sample.created_at
    assert retrieved.name_embedding is not None
    assert sample.name_embedding is not None
    assert np.allclose(retrieved.name_embedding, sample.name_embedding)
    assert retrieved.summary == sample.summary
    assert retrieved.attributes == sample.attributes


async def assert_community_node_equals(
    graph_driver: GraphDriver, retrieved: CommunityNode, sample: CommunityNode
):
    await retrieved.load_name_embedding(graph_driver)
    assert retrieved.uuid == sample.uuid
    assert retrieved.name == sample.name
    assert retrieved.group_id == group_id
    assert retrieved.created_at == sample.created_at
    assert retrieved.name_embedding is not None
    assert sample.name_embedding is not None
    assert np.allclose(retrieved.name_embedding, sample.name_embedding)
    assert retrieved.summary == sample.summary


async def assert_episodic_edge_equals(retrieved: EpisodicEdge, sample: EpisodicEdge):
    assert retrieved.uuid == sample.uuid
    assert retrieved.group_id == sample.group_id
    assert retrieved.created_at == sample.created_at
    assert retrieved.source_node_uuid == sample.source_node_uuid
    assert retrieved.target_node_uuid == sample.target_node_uuid


async def assert_entity_edge_equals(
    graph_driver: GraphDriver, retrieved: EntityEdge, sample: EntityEdge
):
    await retrieved.load_fact_embedding(graph_driver)
    assert retrieved.uuid == sample.uuid
    assert retrieved.group_id == sample.group_id
    assert retrieved.created_at == sample.created_at
    assert retrieved.source_node_uuid == sample.source_node_uuid
    assert retrieved.target_node_uuid == sample.target_node_uuid
    assert retrieved.name == sample.name
    assert retrieved.fact == sample.fact
    assert retrieved.fact_embedding is not None
    assert sample.fact_embedding is not None
    assert np.allclose(retrieved.fact_embedding, sample.fact_embedding)
    assert retrieved.episodes == sample.episodes
    assert retrieved.expired_at == sample.expired_at
    assert retrieved.valid_at == sample.valid_at
    assert retrieved.invalid_at == sample.invalid_at
    assert retrieved.attributes == sample.attributes


if __name__ == '__main__':
    pytest.main([__file__])
