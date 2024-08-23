import os
from datetime import datetime, timedelta

import pytest
from dotenv import load_dotenv

from core.edges import EntityEdge
from core.llm_client import LLMConfig, OpenAIClient
from core.nodes import EntityNode, EpisodicNode
from core.utils.maintenance.temporal_operations import (
    invalidate_edges,
)

load_dotenv()


def setup_llm_client():
    return OpenAIClient(
        LLMConfig(
            api_key=os.getenv('TEST_OPENAI_API_KEY'),
            model=os.getenv('TEST_OPENAI_MODEL'),
            base_url='https://api.openai.com/v1',
        )
    )


def create_test_data():
    now = datetime.now()

    # Create nodes
    node1 = EntityNode(uuid='1', name='Alice', labels=['Person'], created_at=now)
    node2 = EntityNode(uuid='2', name='Bob', labels=['Person'], created_at=now)

    # Create edges
    edge1 = EntityEdge(
        uuid='e1',
        source_node_uuid='1',
        target_node_uuid='2',
        name='LIKES',
        fact='Alice likes Bob',
        created_at=now - timedelta(days=1),
    )
    edge2 = EntityEdge(
        uuid='e2',
        source_node_uuid='1',
        target_node_uuid='2',
        name='DISLIKES',
        fact='Alice dislikes Bob',
        created_at=now,
    )

    existing_edge = (node1, edge1, node2)
    new_edge = (node1, edge2, node2)

    # Create current episode
    current_episode = EpisodicNode(
        name='Current Episode',
        content='Alice now dislikes Bob',
        created_at=now,
        valid_at=now,
        source='test',
        source_description='Test episode for unit testing',
    )

    # Create previous episodes
    previous_episodes = [
        EpisodicNode(
            name='Previous Episode',
            content='Alice liked Bob',
            created_at=now - timedelta(days=1),
            valid_at=now - timedelta(days=1),
            source='test',
            source_description='Test previous episode for unit testing',
        )
    ]

    return existing_edge, new_edge, current_episode, previous_episodes


@pytest.mark.asyncio
@pytest.mark.integration
async def test_invalidate_edges():
    existing_edge, new_edge, current_episode, previous_episodes = create_test_data()

    invalidated_edges = await invalidate_edges(
        setup_llm_client(), [existing_edge], [new_edge], current_episode, previous_episodes
    )

    assert len(invalidated_edges) == 1
    assert invalidated_edges[0].uuid == existing_edge[1].uuid
    assert invalidated_edges[0].expired_at is not None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_invalidate_edges_no_invalidation():
    existing_edge, _, current_episode, previous_episodes = create_test_data()

    invalidated_edges = await invalidate_edges(
        setup_llm_client(), [existing_edge], [], current_episode, previous_episodes
    )

    assert len(invalidated_edges) == 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_invalidate_edges_multiple_existing():
    existing_edge1, new_edge = create_test_data()
    existing_edge2, _ = create_test_data()
    existing_edge2[1].uuid = 'e3'
    existing_edge2[1].name = 'KNOWS'
    existing_edge2[1].fact = 'Alice knows Bob'

    invalidated_edges = await invalidate_edges(
        setup_llm_client(), [existing_edge1, existing_edge2], [new_edge]
    )

    assert len(invalidated_edges) == 1
    assert invalidated_edges[0].uuid == existing_edge1[1].uuid
    assert invalidated_edges[0].expired_at is not None


# Helper function to create more complex test data
def create_complex_test_data():
    now = datetime.now()

    # Create nodes
    node1 = EntityNode(uuid='1', name='Alice', labels=['Person'], created_at=now)
    node2 = EntityNode(uuid='2', name='Bob', labels=['Person'], created_at=now)
    node3 = EntityNode(uuid='3', name='Charlie', labels=['Person'], created_at=now)
    node4 = EntityNode(uuid='4', name='Company XYZ', labels=['Organization'], created_at=now)

    # Create edges
    edge1 = EntityEdge(
        uuid='e1',
        source_node_uuid='1',
        target_node_uuid='2',
        name='LIKES',
        fact='Alice likes Bob',
        created_at=now - timedelta(days=5),
    )
    edge2 = EntityEdge(
        uuid='e2',
        source_node_uuid='1',
        target_node_uuid='3',
        name='FRIENDS_WITH',
        fact='Alice is friends with Charlie',
        created_at=now - timedelta(days=3),
    )
    edge3 = EntityEdge(
        uuid='e3',
        source_node_uuid='2',
        target_node_uuid='4',
        name='WORKS_FOR',
        fact='Bob works for Company XYZ',
        created_at=now - timedelta(days=2),
    )

    existing_edge1 = (node1, edge1, node2)
    existing_edge2 = (node1, edge2, node3)
    existing_edge3 = (node2, edge3, node4)

    return [existing_edge1, existing_edge2, existing_edge3], [
        node1,
        node2,
        node3,
        node4,
    ]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_invalidate_edges_complex():
    existing_edges, nodes = create_complex_test_data()

    # Create a new edge that contradicts an existing one
    new_edge = (
        nodes[0],
        EntityEdge(
            uuid='e4',
            source_node_uuid='1',
            target_node_uuid='2',
            name='DISLIKES',
            fact='Alice dislikes Bob',
            created_at=datetime.now(),
        ),
        nodes[1],
    )

    invalidated_edges = await invalidate_edges(setup_llm_client(), existing_edges, [new_edge])

    assert len(invalidated_edges) == 1
    assert invalidated_edges[0].uuid == 'e1'
    assert invalidated_edges[0].expired_at is not None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_invalidate_edges_temporal_update():
    existing_edges, nodes = create_complex_test_data()

    # Create a new edge that updates an existing one with new information
    new_edge = (
        nodes[1],
        EntityEdge(
            uuid='e5',
            source_node_uuid='2',
            target_node_uuid='4',
            name='LEFT_JOB',
            fact='Bob left his job at Company XYZ',
            created_at=datetime.now(),
        ),
        nodes[3],
    )

    invalidated_edges = await invalidate_edges(setup_llm_client(), existing_edges, [new_edge])

    assert len(invalidated_edges) == 1
    assert invalidated_edges[0].uuid == 'e3'
    assert invalidated_edges[0].expired_at is not None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_invalidate_edges_multiple_invalidations():
    existing_edges, nodes = create_complex_test_data()

    # Create new edges that invalidate multiple existing edges
    new_edge1 = (
        nodes[0],
        EntityEdge(
            uuid='e6',
            source_node_uuid='1',
            target_node_uuid='2',
            name='ENEMIES_WITH',
            fact='Alice and Bob are now enemies',
            created_at=datetime.now(),
        ),
        nodes[1],
    )
    new_edge2 = (
        nodes[0],
        EntityEdge(
            uuid='e7',
            source_node_uuid='1',
            target_node_uuid='3',
            name='ENDED_FRIENDSHIP',
            fact='Alice ended her friendship with Charlie',
            created_at=datetime.now(),
        ),
        nodes[2],
    )

    invalidated_edges = await invalidate_edges(
        setup_llm_client(), existing_edges, [new_edge1, new_edge2]
    )

    assert len(invalidated_edges) == 2
    assert set(edge.uuid for edge in invalidated_edges) == {'e1', 'e2'}
    for edge in invalidated_edges:
        assert edge.expired_at is not None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_invalidate_edges_no_effect():
    existing_edges, nodes = create_complex_test_data()

    # Create a new edge that doesn't invalidate any existing edges
    new_edge = (
        nodes[2],
        EntityEdge(
            uuid='e8',
            source_node_uuid='3',
            target_node_uuid='4',
            name='APPLIED_TO',
            fact='Charlie applied to Company XYZ',
            created_at=datetime.now(),
        ),
        nodes[3],
    )

    invalidated_edges = await invalidate_edges(setup_llm_client(), existing_edges, [new_edge])

    assert len(invalidated_edges) == 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_invalidate_edges_partial_update():
    existing_edges, nodes = create_complex_test_data()

    # Create a new edge that partially updates an existing one
    new_edge = (
        nodes[1],
        EntityEdge(
            uuid='e9',
            source_node_uuid='2',
            target_node_uuid='4',
            name='CHANGED_POSITION',
            fact='Bob changed his position at Company XYZ',
            created_at=datetime.now(),
        ),
        nodes[3],
    )

    invalidated_edges = await invalidate_edges(setup_llm_client(), existing_edges, [new_edge])

    assert len(invalidated_edges) == 0  # The existing edge is not invalidated, just updated


@pytest.mark.asyncio
@pytest.mark.integration
async def test_invalidate_edges_empty_inputs():
    invalidated_edges = await invalidate_edges(setup_llm_client(), [], [])

    assert len(invalidated_edges) == 0
