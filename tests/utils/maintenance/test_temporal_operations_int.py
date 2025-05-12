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
from datetime import timedelta

import pytest
from dotenv import load_dotenv

from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.temporal_operations import (
    get_edge_contradictions,
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
    now = utc_now()

    # Create edges
    existing_edge = EntityEdge(
        uuid='e1',
        source_node_uuid='1',
        target_node_uuid='2',
        name='LIKES',
        fact='Alice likes Bob',
        created_at=now - timedelta(days=1),
        group_id='1',
    )
    new_edge = EntityEdge(
        uuid='e2',
        source_node_uuid='1',
        target_node_uuid='2',
        name='DISLIKES',
        fact='Alice dislikes Bob',
        created_at=now,
        group_id='1',
    )

    # Create current episode
    current_episode = EpisodicNode(
        name='Current Episode',
        content='Alice now dislikes Bob',
        created_at=now,
        valid_at=now,
        source=EpisodeType.message,
        source_description='Test episode for unit testing',
        group_id='1',
    )

    # Create previous episodes
    previous_episodes = [
        EpisodicNode(
            name='Previous Episode',
            content='Alice liked Bob',
            created_at=now - timedelta(days=1),
            valid_at=now - timedelta(days=1),
            source=EpisodeType.message,
            source_description='Test previous episode for unit testing',
            group_id='1',
        )
    ]

    return existing_edge, new_edge, current_episode, previous_episodes


@pytest.mark.asyncio
@pytest.mark.integration
async def test_get_edge_contradictions():
    existing_edge, new_edge, current_episode, previous_episodes = create_test_data()

    invalidated_edges = await get_edge_contradictions(setup_llm_client(), new_edge, [existing_edge])

    assert len(invalidated_edges) == 1
    assert invalidated_edges[0].uuid == existing_edge.uuid


@pytest.mark.asyncio
@pytest.mark.integration
async def test_get_edge_contradictions_no_contradictions():
    _, new_edge, current_episode, previous_episodes = create_test_data()

    invalidated_edges = await get_edge_contradictions(setup_llm_client(), new_edge, [])

    assert len(invalidated_edges) == 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_get_edge_contradictions_multiple_existing():
    existing_edge1, new_edge, _, _ = create_test_data()
    existing_edge2, _, _, _ = create_test_data()
    existing_edge2.uuid = 'e3'
    existing_edge2.name = 'KNOWS'
    existing_edge2.fact = 'Alice knows Bob'

    invalidated_edges = await get_edge_contradictions(
        setup_llm_client(), new_edge, [existing_edge1, existing_edge2]
    )

    assert len(invalidated_edges) == 1
    assert invalidated_edges[0].uuid == existing_edge1.uuid


# Helper function to create more complex test data
def create_complex_test_data():
    now = utc_now()

    # Create nodes
    node1 = EntityNode(uuid='1', name='Alice', labels=['Person'], created_at=now, group_id='1')
    node2 = EntityNode(uuid='2', name='Bob', labels=['Person'], created_at=now, group_id='1')
    node3 = EntityNode(uuid='3', name='Charlie', labels=['Person'], created_at=now, group_id='1')
    node4 = EntityNode(
        uuid='4', name='Company XYZ', labels=['Organization'], created_at=now, group_id='1'
    )

    # Create edges
    existing_edge1 = EntityEdge(
        uuid='e1',
        source_node_uuid='1',
        target_node_uuid='2',
        name='LIKES',
        fact='Alice likes Bob',
        group_id='1',
        created_at=now - timedelta(days=5),
    )
    existing_edge2 = EntityEdge(
        uuid='e2',
        source_node_uuid='1',
        target_node_uuid='3',
        name='FRIENDS_WITH',
        fact='Alice is friends with Charlie',
        group_id='1',
        created_at=now - timedelta(days=3),
    )
    existing_edge3 = EntityEdge(
        uuid='e3',
        source_node_uuid='2',
        target_node_uuid='4',
        name='WORKS_FOR',
        fact='Bob works for Company XYZ',
        group_id='1',
        created_at=now - timedelta(days=2),
    )

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
    new_edge = EntityEdge(
        uuid='e4',
        source_node_uuid='1',
        target_node_uuid='2',
        name='DISLIKES',
        fact='Alice dislikes Bob',
        group_id='1',
        created_at=utc_now(),
    )

    invalidated_edges = await get_edge_contradictions(setup_llm_client(), new_edge, existing_edges)

    assert len(invalidated_edges) == 1
    assert invalidated_edges[0].uuid == 'e1'


@pytest.mark.asyncio
@pytest.mark.integration
async def test_get_edge_contradictions_temporal_update():
    existing_edges, nodes = create_complex_test_data()

    # Create a new edge that updates an existing one with new information
    new_edge = EntityEdge(
        uuid='e5',
        source_node_uuid='2',
        target_node_uuid='4',
        name='LEFT_JOB',
        fact='Bob no longer works at at Company XYZ',
        group_id='1',
        created_at=utc_now(),
    )

    invalidated_edges = await get_edge_contradictions(setup_llm_client(), new_edge, existing_edges)

    assert len(invalidated_edges) == 1
    assert invalidated_edges[0].uuid == 'e3'


@pytest.mark.asyncio
@pytest.mark.integration
async def test_get_edge_contradictions_no_effect():
    existing_edges, nodes = create_complex_test_data()

    # Create a new edge that doesn't invalidate any existing edges
    new_edge = EntityEdge(
        uuid='e8',
        source_node_uuid='3',
        target_node_uuid='4',
        name='APPLIED_TO',
        fact='Charlie applied to Company XYZ',
        group_id='1',
        created_at=utc_now(),
    )

    invalidated_edges = await get_edge_contradictions(setup_llm_client(), new_edge, existing_edges)

    assert len(invalidated_edges) == 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_invalidate_edges_partial_update():
    existing_edges, nodes = create_complex_test_data()

    # Create a new edge that partially updates an existing one
    new_edge = EntityEdge(
        uuid='e9',
        source_node_uuid='2',
        target_node_uuid='4',
        name='CHANGED_POSITION',
        fact='Bob changed his position at Company XYZ',
        group_id='1',
        created_at=utc_now(),
    )

    invalidated_edges = await get_edge_contradictions(setup_llm_client(), new_edge, existing_edges)

    assert len(invalidated_edges) == 0  # The existing edge is not invalidated, just updated


# Run the tests
if __name__ == '__main__':
    pytest.main([__file__])
