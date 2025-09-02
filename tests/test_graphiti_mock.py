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

from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np
import pytest

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.edges import CommunityEdge, EntityEdge, EpisodicEdge
from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters
from graphiti_core.search.search_utils import (
    community_fulltext_search,
    community_similarity_search,
    edge_bfs_search,
    edge_fulltext_search,
    edge_similarity_search,
    episode_fulltext_search,
    episode_mentions_reranker,
    get_communities_by_nodes,
    get_edge_invalidation_candidates,
    get_embeddings_for_communities,
    get_embeddings_for_edges,
    get_embeddings_for_nodes,
    get_mentioned_nodes,
    get_relevant_edges,
    get_relevant_nodes,
    node_bfs_search,
    node_distance_reranker,
    node_fulltext_search,
    node_similarity_search,
)
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from graphiti_core.utils.maintenance.community_operations import (
    determine_entity_community,
    get_community_clusters,
    remove_communities,
)
from graphiti_core.utils.maintenance.edge_operations import filter_existing_duplicate_of_edges
from tests.helpers_test import (
    GraphProvider,
    assert_entity_edge_equals,
    assert_entity_node_equals,
    assert_episodic_edge_equals,
    assert_episodic_node_equals,
    get_edge_count,
    get_node_count,
    group_id,
    group_id_2,
)

pytest_plugins = ('pytest_asyncio',)


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM"""
    mock_llm = Mock(spec=LLMClient)
    mock_llm.config = Mock()
    mock_llm.model = 'test-model'
    mock_llm.small_model = 'test-small-model'
    mock_llm.temperature = 0.0
    mock_llm.max_tokens = 1000
    mock_llm.cache_enabled = False
    mock_llm.cache_dir = None

    # Mock the public method that's actually called
    mock_llm.generate_response = Mock()
    mock_llm.generate_response.return_value = {
        'tool_calls': [
            {
                'name': 'extract_entities',
                'arguments': {'entities': [{'entity': 'test_entity', 'entity_type': 'test_type'}]},
            }
        ]
    }

    return mock_llm


@pytest.fixture
def mock_cross_encoder_client():
    """Create a mock LLM"""
    mock_llm = Mock(spec=CrossEncoderClient)
    mock_llm.config = Mock()

    # Mock the public method that's actually called
    mock_llm.rerank = Mock()
    mock_llm.rerank.return_value = {
        'tool_calls': [
            {
                'name': 'extract_entities',
                'arguments': {'entities': [{'entity': 'test_entity', 'entity_type': 'test_type'}]},
            }
        ]
    }

    return mock_llm


@pytest.mark.asyncio
async def test_add_bulk(graph_driver, mock_llm_client, mock_embedder, mock_cross_encoder_client):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as test fails on FalkorDB')

    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )

    await graphiti.build_indices_and_constraints()

    now = datetime.now()

    # Create episodic nodes
    episode_node_1 = EpisodicNode(
        name='test_episode',
        group_id=group_id,
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='conversation message',
        content='Alice likes Bob',
        valid_at=now,
        entity_edges=[],  # Filled in later
    )
    episode_node_2 = EpisodicNode(
        name='test_episode_2',
        group_id=group_id,
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='conversation message',
        content='Bob adores Alice',
        valid_at=now,
        entity_edges=[],  # Filled in later
    )

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        group_id=group_id,
        labels=['Entity', 'Person'],
        created_at=now,
        summary='test_entity_1 summary',
        attributes={'age': 30, 'location': 'New York'},
    )
    await entity_node_1.generate_name_embedding(mock_embedder)

    entity_node_2 = EntityNode(
        name='test_entity_2',
        group_id=group_id,
        labels=['Entity', 'Person2'],
        created_at=now,
        summary='test_entity_2 summary',
        attributes={'age': 25, 'location': 'Los Angeles'},
    )
    await entity_node_2.generate_name_embedding(mock_embedder)

    entity_node_3 = EntityNode(
        name='test_entity_3',
        group_id=group_id,
        labels=['Entity', 'City', 'Location'],
        created_at=now,
        summary='test_entity_3 summary',
        attributes={'age': 25, 'location': 'Los Angeles'},
    )
    await entity_node_3.generate_name_embedding(mock_embedder)

    entity_node_4 = EntityNode(
        name='test_entity_4',
        group_id=group_id,
        labels=['Entity'],
        created_at=now,
        summary='test_entity_4 summary',
        attributes={'age': 25, 'location': 'Los Angeles'},
    )
    await entity_node_4.generate_name_embedding(mock_embedder)

    # Create entity edges
    entity_edge_1 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        created_at=now,
        name='likes',
        fact='test_entity_1 relates to test_entity_2',
        episodes=[],
        expired_at=now,
        valid_at=now,
        invalid_at=now,
        group_id=group_id,
    )
    await entity_edge_1.generate_embedding(mock_embedder)

    entity_edge_2 = EntityEdge(
        source_node_uuid=entity_node_3.uuid,
        target_node_uuid=entity_node_4.uuid,
        created_at=now,
        name='relates_to',
        fact='test_entity_3 relates to test_entity_4',
        episodes=[],
        expired_at=now,
        valid_at=now,
        invalid_at=now,
        group_id=group_id,
    )
    await entity_edge_2.generate_embedding(mock_embedder)

    # Create episodic to entity edges
    episodic_edge_1 = EpisodicEdge(
        source_node_uuid=episode_node_1.uuid,
        target_node_uuid=entity_node_1.uuid,
        created_at=now,
        group_id=group_id,
    )
    episodic_edge_2 = EpisodicEdge(
        source_node_uuid=episode_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        created_at=now,
        group_id=group_id,
    )
    episodic_edge_3 = EpisodicEdge(
        source_node_uuid=episode_node_2.uuid,
        target_node_uuid=entity_node_3.uuid,
        created_at=now,
        group_id=group_id,
    )
    episodic_edge_4 = EpisodicEdge(
        source_node_uuid=episode_node_2.uuid,
        target_node_uuid=entity_node_4.uuid,
        created_at=now,
        group_id=group_id,
    )

    # Cross reference the ids
    episode_node_1.entity_edges = [entity_edge_1.uuid]
    episode_node_2.entity_edges = [entity_edge_2.uuid]
    entity_edge_1.episodes = [episode_node_1.uuid, episode_node_2.uuid]
    entity_edge_2.episodes = [episode_node_2.uuid]

    # Test add bulk
    await add_nodes_and_edges_bulk(
        graph_driver,
        [episode_node_1, episode_node_2],
        [episodic_edge_1, episodic_edge_2, episodic_edge_3, episodic_edge_4],
        [entity_node_1, entity_node_2, entity_node_3, entity_node_4],
        [entity_edge_1, entity_edge_2],
        mock_embedder,
    )

    node_ids = [
        episode_node_1.uuid,
        episode_node_2.uuid,
        entity_node_1.uuid,
        entity_node_2.uuid,
        entity_node_3.uuid,
        entity_node_4.uuid,
    ]
    edge_ids = [
        episodic_edge_1.uuid,
        episodic_edge_2.uuid,
        episodic_edge_3.uuid,
        episodic_edge_4.uuid,
        entity_edge_1.uuid,
        entity_edge_2.uuid,
    ]
    node_count = await get_node_count(graph_driver, node_ids)
    assert node_count == len(node_ids)
    edge_count = await get_edge_count(graph_driver, edge_ids)
    assert edge_count == len(edge_ids)

    # Test episodic nodes
    retrieved_episode = await EpisodicNode.get_by_uuid(graph_driver, episode_node_1.uuid)
    await assert_episodic_node_equals(retrieved_episode, episode_node_1)

    retrieved_episode = await EpisodicNode.get_by_uuid(graph_driver, episode_node_2.uuid)
    await assert_episodic_node_equals(retrieved_episode, episode_node_2)

    # Test entity nodes
    retrieved_entity_node = await EntityNode.get_by_uuid(graph_driver, entity_node_1.uuid)
    await assert_entity_node_equals(graph_driver, retrieved_entity_node, entity_node_1)

    retrieved_entity_node = await EntityNode.get_by_uuid(graph_driver, entity_node_2.uuid)
    await assert_entity_node_equals(graph_driver, retrieved_entity_node, entity_node_2)

    retrieved_entity_node = await EntityNode.get_by_uuid(graph_driver, entity_node_3.uuid)
    await assert_entity_node_equals(graph_driver, retrieved_entity_node, entity_node_3)

    retrieved_entity_node = await EntityNode.get_by_uuid(graph_driver, entity_node_4.uuid)
    await assert_entity_node_equals(graph_driver, retrieved_entity_node, entity_node_4)

    # Test episodic edges
    retrieved_episode_edge = await EpisodicEdge.get_by_uuid(graph_driver, episodic_edge_1.uuid)
    await assert_episodic_edge_equals(retrieved_episode_edge, episodic_edge_1)

    retrieved_episode_edge = await EpisodicEdge.get_by_uuid(graph_driver, episodic_edge_2.uuid)
    await assert_episodic_edge_equals(retrieved_episode_edge, episodic_edge_2)

    retrieved_episode_edge = await EpisodicEdge.get_by_uuid(graph_driver, episodic_edge_3.uuid)
    await assert_episodic_edge_equals(retrieved_episode_edge, episodic_edge_3)

    retrieved_episode_edge = await EpisodicEdge.get_by_uuid(graph_driver, episodic_edge_4.uuid)
    await assert_episodic_edge_equals(retrieved_episode_edge, episodic_edge_4)

    # Test entity edges
    retrieved_entity_edge = await EntityEdge.get_by_uuid(graph_driver, entity_edge_1.uuid)
    await assert_entity_edge_equals(graph_driver, retrieved_entity_edge, entity_edge_1)

    retrieved_entity_edge = await EntityEdge.get_by_uuid(graph_driver, entity_edge_2.uuid)
    await assert_entity_edge_equals(graph_driver, retrieved_entity_edge, entity_edge_2)


@pytest.mark.asyncio
async def test_remove_episode(
    graph_driver, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )

    await graphiti.build_indices_and_constraints()

    now = datetime.now()

    # Create episodic nodes
    episode_node = EpisodicNode(
        name='test_episode',
        group_id=group_id,
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='conversation message',
        content='Alice likes Bob',
        valid_at=now,
        entity_edges=[],  # Filled in later
    )

    # Create entity nodes
    alice_node = EntityNode(
        name='Alice',
        group_id=group_id,
        labels=['Entity', 'Person'],
        created_at=now,
        summary='Alice summary',
        attributes={'age': 30, 'location': 'New York'},
    )
    await alice_node.generate_name_embedding(mock_embedder)

    bob_node = EntityNode(
        name='Bob',
        group_id=group_id,
        labels=['Entity', 'Person2'],
        created_at=now,
        summary='Bob summary',
        attributes={'age': 25, 'location': 'Los Angeles'},
    )
    await bob_node.generate_name_embedding(mock_embedder)

    # Create entity to entity edge
    entity_edge = EntityEdge(
        source_node_uuid=alice_node.uuid,
        target_node_uuid=bob_node.uuid,
        created_at=now,
        name='likes',
        fact='Alice likes Bob',
        episodes=[],
        expired_at=now,
        valid_at=now,
        invalid_at=now,
        group_id=group_id,
    )
    await entity_edge.generate_embedding(mock_embedder)

    # Create episodic to entity edges
    episodic_alice_edge = EpisodicEdge(
        source_node_uuid=episode_node.uuid,
        target_node_uuid=alice_node.uuid,
        created_at=now,
        group_id=group_id,
    )
    episodic_bob_edge = EpisodicEdge(
        source_node_uuid=episode_node.uuid,
        target_node_uuid=bob_node.uuid,
        created_at=now,
        group_id=group_id,
    )

    # Cross reference the ids
    episode_node.entity_edges = [entity_edge.uuid]
    entity_edge.episodes = [episode_node.uuid]

    # Test add bulk
    await add_nodes_and_edges_bulk(
        graph_driver,
        [episode_node],
        [episodic_alice_edge, episodic_bob_edge],
        [alice_node, bob_node],
        [entity_edge],
        mock_embedder,
    )

    node_ids = [episode_node.uuid, alice_node.uuid, bob_node.uuid]
    edge_ids = [episodic_alice_edge.uuid, episodic_bob_edge.uuid, entity_edge.uuid]
    node_count = await get_node_count(graph_driver, node_ids)
    assert node_count == 3
    edge_count = await get_edge_count(graph_driver, edge_ids)
    assert edge_count == 3

    # Test remove episode
    await graphiti.remove_episode(episode_node.uuid)
    node_count = await get_node_count(graph_driver, node_ids)
    assert node_count == 0
    edge_count = await get_edge_count(graph_driver, edge_ids)
    assert edge_count == 0

    # Test add bulk again
    await add_nodes_and_edges_bulk(
        graph_driver,
        [episode_node],
        [episodic_alice_edge, episodic_bob_edge],
        [alice_node, bob_node],
        [entity_edge],
        mock_embedder,
    )
    node_count = await get_node_count(graph_driver, node_ids)
    assert node_count == 3
    edge_count = await get_edge_count(graph_driver, edge_ids)
    assert edge_count == 3


@pytest.mark.asyncio
async def test_graphiti_retrieve_episodes(
    graph_driver, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as test fails on FalkorDB')

    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )

    await graphiti.build_indices_and_constraints()

    now = datetime.now()
    valid_at_1 = now - timedelta(days=2)
    valid_at_2 = now - timedelta(days=4)
    valid_at_3 = now - timedelta(days=6)

    # Create episodic nodes
    episode_node_1 = EpisodicNode(
        name='test_episode_1',
        labels=[],
        created_at=now,
        valid_at=valid_at_1,
        source=EpisodeType.message,
        source_description='conversation message',
        content='Test message 1',
        entity_edges=[],
        group_id=group_id,
    )
    episode_node_2 = EpisodicNode(
        name='test_episode_2',
        labels=[],
        created_at=now,
        valid_at=valid_at_2,
        source=EpisodeType.message,
        source_description='conversation message',
        content='Test message 2',
        entity_edges=[],
        group_id=group_id,
    )
    episode_node_3 = EpisodicNode(
        name='test_episode_3',
        labels=[],
        created_at=now,
        valid_at=valid_at_3,
        source=EpisodeType.message,
        source_description='conversation message',
        content='Test message 3',
        entity_edges=[],
        group_id=group_id,
    )

    # Save the nodes
    await episode_node_1.save(graph_driver)
    await episode_node_2.save(graph_driver)
    await episode_node_3.save(graph_driver)

    node_ids = [episode_node_1.uuid, episode_node_2.uuid, episode_node_3.uuid]
    node_count = await get_node_count(graph_driver, node_ids)
    assert node_count == 3

    # Retrieve episodes
    query_time = now - timedelta(days=3)
    episodes = await graphiti.retrieve_episodes(
        query_time, last_n=5, group_ids=[group_id], source=EpisodeType.message
    )
    assert len(episodes) == 2
    assert episodes[0].name == episode_node_3.name
    assert episodes[1].name == episode_node_2.name


@pytest.mark.asyncio
async def test_filter_existing_duplicate_of_edges(graph_driver, mock_embedder):
    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)
    entity_node_3 = EntityNode(
        name='test_entity_3',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_3.generate_name_embedding(mock_embedder)
    entity_node_4 = EntityNode(
        name='test_entity_4',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_4.generate_name_embedding(mock_embedder)

    # Save the nodes
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_node_3.save(graph_driver)
    await entity_node_4.save(graph_driver)

    node_ids = [entity_node_1.uuid, entity_node_2.uuid, entity_node_3.uuid, entity_node_4.uuid]
    node_count = await get_node_count(graph_driver, node_ids)
    assert node_count == 4

    # Create duplicate entity edge
    entity_edge = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        name='IS_DUPLICATE_OF',
        fact='test_entity_1 is a duplicate of test_entity_2',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_edge.generate_embedding(mock_embedder)
    await entity_edge.save(graph_driver)

    # Filter duplicate entity edges
    duplicate_node_tuples = [
        (entity_node_1, entity_node_2),
        (entity_node_3, entity_node_4),
    ]
    node_tuples = await filter_existing_duplicate_of_edges(graph_driver, duplicate_node_tuples)
    assert len(node_tuples) == 1
    assert [node.name for node in node_tuples[0]] == [entity_node_3.name, entity_node_4.name]


@pytest.mark.asyncio
async def test_determine_entity_community(graph_driver, mock_embedder):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as test fails on FalkorDB')

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)
    entity_node_3 = EntityNode(
        name='test_entity_3',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_3.generate_name_embedding(mock_embedder)
    entity_node_4 = EntityNode(
        name='test_entity_4',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_4.generate_name_embedding(mock_embedder)

    # Create entity edges
    entity_edge_1 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_4.uuid,
        name='RELATES_TO',
        fact='test_entity_1 relates to test_entity_4',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_edge_1.generate_embedding(mock_embedder)
    entity_edge_2 = EntityEdge(
        source_node_uuid=entity_node_2.uuid,
        target_node_uuid=entity_node_4.uuid,
        name='RELATES_TO',
        fact='test_entity_2 relates to test_entity_4',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_edge_2.generate_embedding(mock_embedder)
    entity_edge_3 = EntityEdge(
        source_node_uuid=entity_node_3.uuid,
        target_node_uuid=entity_node_4.uuid,
        name='RELATES_TO',
        fact='test_entity_3 relates to test_entity_4',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_edge_3.generate_embedding(mock_embedder)

    # Create community nodes
    community_node_1 = CommunityNode(
        name='test_community_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await community_node_1.generate_name_embedding(mock_embedder)
    community_node_2 = CommunityNode(
        name='test_community_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await community_node_2.generate_name_embedding(mock_embedder)

    # Create community to entity edges
    community_edge_1 = CommunityEdge(
        source_node_uuid=community_node_1.uuid,
        target_node_uuid=entity_node_1.uuid,
        created_at=datetime.now(),
        group_id=group_id,
    )
    community_edge_2 = CommunityEdge(
        source_node_uuid=community_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        created_at=datetime.now(),
        group_id=group_id,
    )
    community_edge_3 = CommunityEdge(
        source_node_uuid=community_node_2.uuid,
        target_node_uuid=entity_node_3.uuid,
        created_at=datetime.now(),
        group_id=group_id,
    )

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_node_3.save(graph_driver)
    await entity_node_4.save(graph_driver)
    await community_node_1.save(graph_driver)
    await community_node_2.save(graph_driver)

    await entity_edge_1.save(graph_driver)
    await entity_edge_2.save(graph_driver)
    await entity_edge_3.save(graph_driver)
    await community_edge_1.save(graph_driver)
    await community_edge_2.save(graph_driver)
    await community_edge_3.save(graph_driver)

    node_ids = [
        entity_node_1.uuid,
        entity_node_2.uuid,
        entity_node_3.uuid,
        entity_node_4.uuid,
        community_node_1.uuid,
        community_node_2.uuid,
    ]
    edge_ids = [
        entity_edge_1.uuid,
        entity_edge_2.uuid,
        entity_edge_3.uuid,
        community_edge_1.uuid,
        community_edge_2.uuid,
        community_edge_3.uuid,
    ]
    node_count = await get_node_count(graph_driver, node_ids)
    assert node_count == 6
    edge_count = await get_edge_count(graph_driver, edge_ids)
    assert edge_count == 6

    # Determine entity community
    community, is_new = await determine_entity_community(graph_driver, entity_node_4)
    assert community.name == community_node_1.name
    assert is_new

    # Add entity to community edge
    community_edge_4 = CommunityEdge(
        source_node_uuid=community_node_1.uuid,
        target_node_uuid=entity_node_4.uuid,
        created_at=datetime.now(),
        group_id=group_id,
    )
    await community_edge_4.save(graph_driver)

    # Determine entity community again
    community, is_new = await determine_entity_community(graph_driver, entity_node_4)
    assert community.name == community_node_1.name
    assert not is_new

    await remove_communities(graph_driver)
    node_count = await get_node_count(graph_driver, [community_node_1.uuid, community_node_2.uuid])
    assert node_count == 0


@pytest.mark.asyncio
async def test_get_community_clusters(graph_driver, mock_embedder):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as test fails on FalkorDB')

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)
    entity_node_3 = EntityNode(
        name='test_entity_3',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id_2,
    )
    await entity_node_3.generate_name_embedding(mock_embedder)
    entity_node_4 = EntityNode(
        name='test_entity_4',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id_2,
    )
    await entity_node_4.generate_name_embedding(mock_embedder)

    # Create entity edges
    entity_edge_1 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        name='RELATES_TO',
        fact='test_entity_1 relates to test_entity_2',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_edge_1.generate_embedding(mock_embedder)
    entity_edge_2 = EntityEdge(
        source_node_uuid=entity_node_3.uuid,
        target_node_uuid=entity_node_4.uuid,
        name='RELATES_TO',
        fact='test_entity_3 relates to test_entity_4',
        created_at=datetime.now(),
        group_id=group_id_2,
    )
    await entity_edge_2.generate_embedding(mock_embedder)

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_node_3.save(graph_driver)
    await entity_node_4.save(graph_driver)
    await entity_edge_1.save(graph_driver)
    await entity_edge_2.save(graph_driver)

    node_ids = [entity_node_1.uuid, entity_node_2.uuid, entity_node_3.uuid, entity_node_4.uuid]
    edge_ids = [entity_edge_1.uuid, entity_edge_2.uuid]
    node_count = await get_node_count(graph_driver, node_ids)
    assert node_count == 4
    edge_count = await get_edge_count(graph_driver, edge_ids)
    assert edge_count == 2

    # Get community clusters
    clusters = await get_community_clusters(graph_driver, group_ids=None)
    assert len(clusters) == 2
    assert len(clusters[0]) == 2
    assert len(clusters[1]) == 2
    entities_1 = set([node.name for node in clusters[0]])
    entities_2 = set([node.name for node in clusters[1]])
    assert entities_1 == set(['test_entity_1', 'test_entity_2']) or entities_2 == set(
        ['test_entity_1', 'test_entity_2']
    )
    assert entities_1 == set(['test_entity_3', 'test_entity_4']) or entities_2 == set(
        ['test_entity_3', 'test_entity_4']
    )


@pytest.mark.asyncio
async def test_get_mentioned_nodes(graph_driver, mock_embedder):
    # Create episodic nodes
    episodic_node_1 = EpisodicNode(
        name='test_episodic_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
        source=EpisodeType.message,
        source_description='test_source_description',
        content='test_content',
        valid_at=datetime.now(),
    )
    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)

    # Create episodic to entity edges
    episodic_edge_1 = EpisodicEdge(
        source_node_uuid=episodic_node_1.uuid,
        target_node_uuid=entity_node_1.uuid,
        created_at=datetime.now(),
        group_id=group_id,
    )

    # Save the graph
    await episodic_node_1.save(graph_driver)
    await entity_node_1.save(graph_driver)
    await episodic_edge_1.save(graph_driver)

    # Get mentioned nodes
    mentioned_nodes = await get_mentioned_nodes(graph_driver, [episodic_node_1])
    assert len(mentioned_nodes) == 1
    assert mentioned_nodes[0].name == entity_node_1.name


@pytest.mark.asyncio
async def test_get_communities_by_nodes(graph_driver, mock_embedder):
    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)

    # Create community nodes
    community_node_1 = CommunityNode(
        name='test_community_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await community_node_1.generate_name_embedding(mock_embedder)

    # Create community to entity edges
    community_edge_1 = CommunityEdge(
        source_node_uuid=community_node_1.uuid,
        target_node_uuid=entity_node_1.uuid,
        created_at=datetime.now(),
        group_id=group_id,
    )

    # Save the graph
    await entity_node_1.save(graph_driver)
    await community_node_1.save(graph_driver)
    await community_edge_1.save(graph_driver)

    # Get communities by nodes
    communities = await get_communities_by_nodes(graph_driver, [entity_node_1])
    assert len(communities) == 1
    assert communities[0].name == community_node_1.name


@pytest.mark.asyncio
async def test_edge_fulltext_search(
    graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder_client
):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )
    await graphiti.build_indices_and_constraints()

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)

    now = datetime.now()
    created_at = now
    expired_at = now + timedelta(days=6)
    valid_at = now + timedelta(days=2)
    invalid_at = now + timedelta(days=4)

    # Create entity edges
    entity_edge_1 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        name='RELATES_TO',
        fact='test_entity_1 relates to test_entity_2',
        created_at=created_at,
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
        group_id=group_id,
    )
    await entity_edge_1.generate_embedding(mock_embedder)

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_edge_1.save(graph_driver)

    # Search for entity edges
    search_filters = SearchFilters(
        node_labels=['Entity'],
        edge_types=['RELATES_TO'],
        created_at=[
            [DateFilter(date=created_at, comparison_operator=ComparisonOperator.equals)],
        ],
        expired_at=[
            [DateFilter(date=now, comparison_operator=ComparisonOperator.not_equals)],
        ],
        valid_at=[
            [
                DateFilter(
                    date=now + timedelta(days=1),
                    comparison_operator=ComparisonOperator.greater_than_equal,
                )
            ],
            [
                DateFilter(
                    date=now + timedelta(days=3),
                    comparison_operator=ComparisonOperator.less_than_equal,
                )
            ],
        ],
        invalid_at=[
            [
                DateFilter(
                    date=now + timedelta(days=3),
                    comparison_operator=ComparisonOperator.greater_than,
                )
            ],
            [
                DateFilter(
                    date=now + timedelta(days=5), comparison_operator=ComparisonOperator.less_than
                )
            ],
        ],
    )
    edges = await edge_fulltext_search(
        graph_driver, 'test_entity_1 relates to test_entity_2', search_filters, group_ids=[group_id]
    )
    assert len(edges) == 1
    assert edges[0].name == entity_edge_1.name


@pytest.mark.asyncio
async def test_edge_similarity_search(graph_driver, mock_embedder):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)

    now = datetime.now()
    created_at = now
    expired_at = now + timedelta(days=6)
    valid_at = now + timedelta(days=2)
    invalid_at = now + timedelta(days=4)

    # Create entity edges
    entity_edge_1 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        name='RELATES_TO',
        fact='test_entity_1 relates to test_entity_2',
        created_at=created_at,
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
        group_id=group_id,
    )
    await entity_edge_1.generate_embedding(mock_embedder)

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_edge_1.save(graph_driver)

    # Search for entity edges
    search_filters = SearchFilters(
        node_labels=['Entity'],
        edge_types=['RELATES_TO'],
        created_at=[
            [DateFilter(date=created_at, comparison_operator=ComparisonOperator.equals)],
        ],
        expired_at=[
            [DateFilter(date=now, comparison_operator=ComparisonOperator.not_equals)],
        ],
        valid_at=[
            [
                DateFilter(
                    date=now + timedelta(days=1),
                    comparison_operator=ComparisonOperator.greater_than_equal,
                )
            ],
            [
                DateFilter(
                    date=now + timedelta(days=3),
                    comparison_operator=ComparisonOperator.less_than_equal,
                )
            ],
        ],
        invalid_at=[
            [
                DateFilter(
                    date=now + timedelta(days=3),
                    comparison_operator=ComparisonOperator.greater_than,
                )
            ],
            [
                DateFilter(
                    date=now + timedelta(days=5), comparison_operator=ComparisonOperator.less_than
                )
            ],
        ],
    )
    edges = await edge_similarity_search(
        graph_driver,
        entity_edge_1.fact_embedding,
        entity_node_1.uuid,
        entity_node_2.uuid,
        search_filters,
        group_ids=[group_id],
    )
    assert len(edges) == 1
    assert edges[0].name == entity_edge_1.name


@pytest.mark.asyncio
async def test_edge_bfs_search(graph_driver, mock_embedder):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    # Create episodic nodes
    episodic_node_1 = EpisodicNode(
        name='test_episodic_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
        source=EpisodeType.message,
        source_description='test_source_description',
        content='test_content',
        valid_at=datetime.now(),
    )

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)
    entity_node_3 = EntityNode(
        name='test_entity_3',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_3.generate_name_embedding(mock_embedder)

    now = datetime.now()
    created_at = now
    expired_at = now + timedelta(days=6)
    valid_at = now + timedelta(days=2)
    invalid_at = now + timedelta(days=4)

    # Create entity edges
    entity_edge_1 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        name='RELATES_TO',
        fact='test_entity_1 relates to test_entity_2',
        created_at=created_at,
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
        group_id=group_id,
    )
    await entity_edge_1.generate_embedding(mock_embedder)
    entity_edge_2 = EntityEdge(
        source_node_uuid=entity_node_2.uuid,
        target_node_uuid=entity_node_3.uuid,
        name='RELATES_TO',
        fact='test_entity_2 relates to test_entity_3',
        created_at=created_at,
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
        group_id=group_id,
    )
    await entity_edge_2.generate_embedding(mock_embedder)

    # Create episodic to entity edges
    episodic_edge_1 = EpisodicEdge(
        source_node_uuid=episodic_node_1.uuid,
        target_node_uuid=entity_node_1.uuid,
        created_at=datetime.now(),
        group_id=group_id,
    )

    # Save the graph
    await episodic_node_1.save(graph_driver)
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_node_3.save(graph_driver)
    await entity_edge_1.save(graph_driver)
    await entity_edge_2.save(graph_driver)
    await episodic_edge_1.save(graph_driver)

    # Search for entity edges
    search_filters = SearchFilters(
        node_labels=['Entity'],
        edge_types=['RELATES_TO'],
        created_at=[
            [DateFilter(date=created_at, comparison_operator=ComparisonOperator.equals)],
        ],
        expired_at=[
            [DateFilter(date=now, comparison_operator=ComparisonOperator.not_equals)],
        ],
        valid_at=[
            [
                DateFilter(
                    date=now + timedelta(days=1),
                    comparison_operator=ComparisonOperator.greater_than_equal,
                )
            ],
            [
                DateFilter(
                    date=now + timedelta(days=3),
                    comparison_operator=ComparisonOperator.less_than_equal,
                )
            ],
        ],
        invalid_at=[
            [
                DateFilter(
                    date=now + timedelta(days=3),
                    comparison_operator=ComparisonOperator.greater_than,
                )
            ],
            [
                DateFilter(
                    date=now + timedelta(days=5), comparison_operator=ComparisonOperator.less_than
                )
            ],
        ],
    )

    # Test bfs from episodic node

    edges = await edge_bfs_search(
        graph_driver,
        [episodic_node_1.uuid],
        1,
        search_filters,
        group_ids=[group_id],
    )
    assert len(edges) == 0

    edges = await edge_bfs_search(
        graph_driver,
        [episodic_node_1.uuid],
        2,
        search_filters,
        group_ids=[group_id],
    )
    edges_deduplicated = set({edge.uuid: edge.fact for edge in edges}.values())
    assert len(edges_deduplicated) == 1
    assert edges_deduplicated == {'test_entity_1 relates to test_entity_2'}

    edges = await edge_bfs_search(
        graph_driver,
        [episodic_node_1.uuid],
        3,
        search_filters,
        group_ids=[group_id],
    )
    edges_deduplicated = set({edge.uuid: edge.fact for edge in edges}.values())
    assert len(edges_deduplicated) == 2
    assert edges_deduplicated == {
        'test_entity_1 relates to test_entity_2',
        'test_entity_2 relates to test_entity_3',
    }

    # Test bfs from entity node

    edges = await edge_bfs_search(
        graph_driver,
        [entity_node_1.uuid],
        1,
        search_filters,
        group_ids=[group_id],
    )
    edges_deduplicated = set({edge.uuid: edge.fact for edge in edges}.values())
    assert len(edges_deduplicated) == 1
    assert edges_deduplicated == {'test_entity_1 relates to test_entity_2'}

    edges = await edge_bfs_search(
        graph_driver,
        [entity_node_1.uuid],
        2,
        search_filters,
        group_ids=[group_id],
    )
    edges_deduplicated = set({edge.uuid: edge.fact for edge in edges}.values())
    assert len(edges_deduplicated) == 2
    assert edges_deduplicated == {
        'test_entity_1 relates to test_entity_2',
        'test_entity_2 relates to test_entity_3',
    }


@pytest.mark.asyncio
async def test_node_fulltext_search(
    graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder_client
):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )
    await graphiti.build_indices_and_constraints()

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        summary='Summary about Alice',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        summary='Summary about Bob',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)

    # Search for entity edges
    search_filters = SearchFilters(node_labels=['Entity'])
    nodes = await node_fulltext_search(
        graph_driver,
        'Alice',
        search_filters,
        group_ids=[group_id],
    )
    assert len(nodes) == 1
    assert nodes[0].name == entity_node_1.name


@pytest.mark.asyncio
async def test_node_similarity_search(graph_driver, mock_embedder):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_alice',
        summary='Summary about Alice',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_bob',
        summary='Summary about Bob',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)

    # Search for entity edges
    search_filters = SearchFilters(node_labels=['Entity'])
    nodes = await node_similarity_search(
        graph_driver,
        entity_node_1.name_embedding,
        search_filters,
        group_ids=[group_id],
        min_score=0.9,
    )
    assert len(nodes) == 1
    assert nodes[0].name == entity_node_1.name


@pytest.mark.asyncio
async def test_node_bfs_search(graph_driver, mock_embedder):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    # Create episodic nodes
    episodic_node_1 = EpisodicNode(
        name='test_episodic_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
        source=EpisodeType.message,
        source_description='test_source_description',
        content='test_content',
        valid_at=datetime.now(),
    )

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)
    entity_node_3 = EntityNode(
        name='test_entity_3',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_3.generate_name_embedding(mock_embedder)

    # Create entity edges
    entity_edge_1 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        name='RELATES_TO',
        fact='test_entity_1 relates to test_entity_2',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_edge_1.generate_embedding(mock_embedder)
    entity_edge_2 = EntityEdge(
        source_node_uuid=entity_node_2.uuid,
        target_node_uuid=entity_node_3.uuid,
        name='RELATES_TO',
        fact='test_entity_2 relates to test_entity_3',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_edge_2.generate_embedding(mock_embedder)

    # Create episodic to entity edges
    episodic_edge_1 = EpisodicEdge(
        source_node_uuid=episodic_node_1.uuid,
        target_node_uuid=entity_node_1.uuid,
        created_at=datetime.now(),
        group_id=group_id,
    )

    # Save the graph
    await episodic_node_1.save(graph_driver)
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_node_3.save(graph_driver)
    await entity_edge_1.save(graph_driver)
    await entity_edge_2.save(graph_driver)
    await episodic_edge_1.save(graph_driver)

    # Search for entity nodes
    search_filters = SearchFilters(
        node_labels=['Entity'],
    )

    # Test bfs from episodic node

    nodes = await node_bfs_search(
        graph_driver,
        [episodic_node_1.uuid],
        search_filters,
        1,
        group_ids=[group_id],
    )
    nodes_deduplicated = set({node.uuid: node.name for node in nodes}.values())
    assert len(nodes_deduplicated) == 1
    assert nodes_deduplicated == {'test_entity_1'}

    nodes = await node_bfs_search(
        graph_driver,
        [episodic_node_1.uuid],
        search_filters,
        2,
        group_ids=[group_id],
    )
    nodes_deduplicated = set({node.uuid: node.name for node in nodes}.values())
    assert len(nodes_deduplicated) == 2
    assert nodes_deduplicated == {'test_entity_1', 'test_entity_2'}

    # Test bfs from entity node

    nodes = await node_bfs_search(
        graph_driver,
        [entity_node_1.uuid],
        search_filters,
        1,
        group_ids=[group_id],
    )
    nodes_deduplicated = set({node.uuid: node.name for node in nodes}.values())
    assert len(nodes_deduplicated) == 1
    assert nodes_deduplicated == {'test_entity_2'}


@pytest.mark.asyncio
async def test_episode_fulltext_search(
    graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder_client
):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )
    await graphiti.build_indices_and_constraints()

    # Create episodic nodes
    episodic_node_1 = EpisodicNode(
        name='test_episodic_1',
        content='test_content',
        created_at=datetime.now(),
        valid_at=datetime.now(),
        group_id=group_id,
        source=EpisodeType.message,
        source_description='Description about Alice',
    )
    episodic_node_2 = EpisodicNode(
        name='test_episodic_2',
        content='test_content_2',
        created_at=datetime.now(),
        valid_at=datetime.now(),
        group_id=group_id,
        source=EpisodeType.message,
        source_description='Description about Bob',
    )

    # Save the graph
    await episodic_node_1.save(graph_driver)
    await episodic_node_2.save(graph_driver)

    # Search for episodic nodes
    search_filters = SearchFilters(node_labels=['Episodic'])
    nodes = await episode_fulltext_search(
        graph_driver,
        'Alice',
        search_filters,
        group_ids=[group_id],
    )
    assert len(nodes) == 1
    assert nodes[0].name == episodic_node_1.name


@pytest.mark.asyncio
async def test_community_fulltext_search(
    graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder_client
):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )
    await graphiti.build_indices_and_constraints()

    # Create community nodes
    community_node_1 = CommunityNode(
        name='Alice',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await community_node_1.generate_name_embedding(mock_embedder)
    community_node_2 = CommunityNode(
        name='Bob',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await community_node_2.generate_name_embedding(mock_embedder)

    # Save the graph
    await community_node_1.save(graph_driver)
    await community_node_2.save(graph_driver)

    # Search for community nodes
    nodes = await community_fulltext_search(
        graph_driver,
        'Alice',
        group_ids=[group_id],
    )
    assert len(nodes) == 1
    assert nodes[0].name == community_node_1.name


@pytest.mark.asyncio
async def test_community_similarity_search(
    graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder_client
):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )
    await graphiti.build_indices_and_constraints()

    # Create community nodes
    community_node_1 = CommunityNode(
        name='Alice',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await community_node_1.generate_name_embedding(mock_embedder)
    community_node_2 = CommunityNode(
        name='Bob',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await community_node_2.generate_name_embedding(mock_embedder)

    # Save the graph
    await community_node_1.save(graph_driver)
    await community_node_2.save(graph_driver)

    # Search for community nodes
    nodes = await community_similarity_search(
        graph_driver,
        community_node_1.name_embedding,
        group_ids=[group_id],
        min_score=0.9,
    )
    assert len(nodes) == 1
    assert nodes[0].name == community_node_1.name


@pytest.mark.asyncio
async def test_get_relevant_nodes(
    graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder_client
):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    if graph_driver.provider == GraphProvider.KUZU:
        pytest.skip('Skipping as tests fail on Kuzu')

    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )
    await graphiti.build_indices_and_constraints()

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='Alice',
        summary='Alice',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='Bob',
        summary='Bob',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)
    entity_node_3 = EntityNode(
        name='Alice Smith',
        summary='Alice Smith',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_3.generate_name_embedding(mock_embedder)

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_node_3.save(graph_driver)

    # Search for entity nodes
    search_filters = SearchFilters(node_labels=['Entity'])
    nodes = (
        await get_relevant_nodes(
            graph_driver,
            [entity_node_1],
            search_filters,
            min_score=0.9,
        )
    )[0]
    assert len(nodes) == 2
    assert set({node.name for node in nodes}) == {entity_node_1.name, entity_node_3.name}


@pytest.mark.asyncio
async def test_get_relevant_edges_and_invalidation_candidates(
    graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder_client
):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )
    await graphiti.build_indices_and_constraints()

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        summary='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        summary='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)
    entity_node_3 = EntityNode(
        name='test_entity_3',
        summary='test_entity_3',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_3.generate_name_embedding(mock_embedder)

    now = datetime.now()
    created_at = now
    expired_at = now + timedelta(days=6)
    valid_at = now + timedelta(days=2)
    invalid_at = now + timedelta(days=4)

    # Create entity edges
    entity_edge_1 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        name='RELATES_TO',
        fact='Alice',
        created_at=created_at,
        expired_at=expired_at,
        valid_at=valid_at,
        invalid_at=invalid_at,
        group_id=group_id,
    )
    await entity_edge_1.generate_embedding(mock_embedder)
    entity_edge_2 = EntityEdge(
        source_node_uuid=entity_node_2.uuid,
        target_node_uuid=entity_node_3.uuid,
        name='RELATES_TO',
        fact='Bob',
        created_at=created_at,
        expired_at=expired_at,
        valid_at=valid_at,
        invalid_at=invalid_at,
        group_id=group_id,
    )
    await entity_edge_2.generate_embedding(mock_embedder)
    entity_edge_3 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_3.uuid,
        name='RELATES_TO',
        fact='Alice',
        created_at=created_at,
        expired_at=expired_at,
        valid_at=valid_at,
        invalid_at=invalid_at,
        group_id=group_id,
    )
    await entity_edge_3.generate_embedding(mock_embedder)

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_node_3.save(graph_driver)
    await entity_edge_1.save(graph_driver)
    await entity_edge_2.save(graph_driver)
    await entity_edge_3.save(graph_driver)

    # Search for entity nodes
    search_filters = SearchFilters(
        node_labels=['Entity'],
        edge_types=['RELATES_TO'],
        created_at=[
            [DateFilter(date=created_at, comparison_operator=ComparisonOperator.equals)],
        ],
        expired_at=[
            [DateFilter(date=now, comparison_operator=ComparisonOperator.not_equals)],
        ],
        valid_at=[
            [
                DateFilter(
                    date=now + timedelta(days=1),
                    comparison_operator=ComparisonOperator.greater_than_equal,
                )
            ],
            [
                DateFilter(
                    date=now + timedelta(days=3),
                    comparison_operator=ComparisonOperator.less_than_equal,
                )
            ],
        ],
        invalid_at=[
            [
                DateFilter(
                    date=now + timedelta(days=3),
                    comparison_operator=ComparisonOperator.greater_than,
                )
            ],
            [
                DateFilter(
                    date=now + timedelta(days=5), comparison_operator=ComparisonOperator.less_than
                )
            ],
        ],
    )
    edges = (
        await get_relevant_edges(
            graph_driver,
            [entity_edge_1],
            search_filters,
            min_score=0.9,
        )
    )[0]
    assert len(edges) == 1
    assert set({edge.name for edge in edges}) == {entity_edge_1.name}

    edges = (
        await get_edge_invalidation_candidates(
            graph_driver,
            [entity_edge_1],
            search_filters,
            min_score=0.9,
        )
    )[0]
    assert len(edges) == 2
    assert set({edge.name for edge in edges}) == {entity_edge_1.name, entity_edge_3.name}


@pytest.mark.asyncio
async def test_node_distance_reranker(graph_driver, mock_embedder):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)
    entity_node_3 = EntityNode(
        name='test_entity_3',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_3.generate_name_embedding(mock_embedder)

    # Create entity edges
    entity_edge_1 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        name='RELATES_TO',
        fact='test_entity_1 relates to test_entity_2',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_edge_1.generate_embedding(mock_embedder)

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_node_3.save(graph_driver)
    await entity_edge_1.save(graph_driver)

    # Test reranker
    reranked_uuids, reranked_scores = await node_distance_reranker(
        graph_driver,
        [entity_node_2.uuid, entity_node_3.uuid],
        entity_node_1.uuid,
    )
    uuid_to_name = {
        entity_node_1.uuid: entity_node_1.name,
        entity_node_2.uuid: entity_node_2.name,
        entity_node_3.uuid: entity_node_3.name,
    }
    names = [uuid_to_name[uuid] for uuid in reranked_uuids]
    assert names == [entity_node_2.name, entity_node_3.name]
    assert np.allclose(reranked_scores, [1.0, 0.0])


@pytest.mark.asyncio
async def test_episode_mentions_reranker(graph_driver, mock_embedder):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    # Create episodic nodes
    episodic_node_1 = EpisodicNode(
        name='test_episodic_1',
        content='test_content',
        created_at=datetime.now(),
        valid_at=datetime.now(),
        group_id=group_id,
        source=EpisodeType.message,
        source_description='Description about Alice',
    )

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)

    # Create entity edges
    episodic_edge_1 = EpisodicEdge(
        source_node_uuid=episodic_node_1.uuid,
        target_node_uuid=entity_node_1.uuid,
        created_at=datetime.now(),
        group_id=group_id,
    )

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await episodic_node_1.save(graph_driver)
    await episodic_edge_1.save(graph_driver)

    # Test reranker
    reranked_uuids, reranked_scores = await episode_mentions_reranker(
        graph_driver,
        [[entity_node_1.uuid, entity_node_2.uuid]],
    )
    uuid_to_name = {entity_node_1.uuid: entity_node_1.name, entity_node_2.uuid: entity_node_2.name}
    names = [uuid_to_name[uuid] for uuid in reranked_uuids]
    assert names == [entity_node_1.name, entity_node_2.name]
    assert np.allclose(reranked_scores, [1.0, float('inf')])


@pytest.mark.asyncio
async def test_get_embeddings_for_edges(graph_driver, mock_embedder):
    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)

    # Create entity edges
    entity_edge_1 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        name='RELATES_TO',
        fact='test_entity_1 relates to test_entity_2',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_edge_1.generate_embedding(mock_embedder)

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_edge_1.save(graph_driver)

    # Get embeddings for edges
    embeddings = await get_embeddings_for_edges(graph_driver, [entity_edge_1])
    assert len(embeddings) == 1
    assert entity_edge_1.uuid in embeddings
    assert np.allclose(embeddings[entity_edge_1.uuid], entity_edge_1.fact_embedding)


@pytest.mark.asyncio
async def test_get_embeddings_for_nodes(graph_driver, mock_embedder):
    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)

    # Save the graph
    await entity_node_1.save(graph_driver)

    # Get embeddings for edges
    embeddings = await get_embeddings_for_nodes(graph_driver, [entity_node_1])
    assert len(embeddings) == 1
    assert entity_node_1.uuid in embeddings
    assert np.allclose(embeddings[entity_node_1.uuid], entity_node_1.name_embedding)


@pytest.mark.asyncio
async def test_get_embeddings_for_communities(graph_driver, mock_embedder):
    # Create community nodes
    community_node_1 = CommunityNode(
        name='test_community_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await community_node_1.generate_name_embedding(mock_embedder)

    # Save the graph
    await community_node_1.save(graph_driver)

    # Get embeddings for communities
    embeddings = await get_embeddings_for_communities(graph_driver, [community_node_1])
    assert len(embeddings) == 1
    assert community_node_1.uuid in embeddings
    assert np.allclose(embeddings[community_node_1.uuid], community_node_1.name_embedding)
