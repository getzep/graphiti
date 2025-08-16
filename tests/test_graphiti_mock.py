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

from datetime import datetime
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import pytest

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from tests.helpers_test import drivers, get_driver, get_edge_count, get_node_count

pytest_plugins = ('pytest_asyncio',)


group_id = f'test_group_{str(uuid4())}'
embedding_dim = 384
embeddings = {
    'Alice': np.random.uniform(0.0, 0.9, embedding_dim).tolist(),
    'Bob': np.random.uniform(0.0, 0.9, embedding_dim).tolist(),
    'Alice likes Bob': np.random.uniform(0.0, 0.9, embedding_dim).tolist(),
}


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
@pytest.mark.parametrize(
    'driver',
    drivers,
)
async def test_graphiti(driver, mock_llm_client, mock_embedder, mock_cross_encoder_client):
    graph_driver = get_driver(driver)
    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )

    await graphiti.build_indices_and_constraints()

    now = datetime.now()

    # Create episodic node
    episode_node = EpisodicNode(
        name='test_episode',
        labels=[],
        created_at=now,
        valid_at=now,
        source=EpisodeType.message,
        source_description='conversation message',
        content='Alice likes Bob',
        entity_edges=[],
        group_id=group_id,
    )

    # Create entity node
    alice_node = EntityNode(
        name='Alice',
        labels=[],
        created_at=now,
        summary='Alice summary',
        group_id=group_id,
    )
    await alice_node.generate_name_embedding(mock_embedder)

    # Create entity node
    bob_node = EntityNode(
        name='Bob',
        labels=[],
        created_at=now,
        summary='Bob summary',
        group_id=group_id,
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

    # Create episodic to entity edge
    episodic_alice_edge = EpisodicEdge(
        source_node_uuid=episode_node.uuid,
        target_node_uuid=alice_node.uuid,
        created_at=now,
        group_id=group_id,
    )

    # Create episodic to entity edge
    episodic_bob_edge = EpisodicEdge(
        source_node_uuid=episode_node.uuid,
        target_node_uuid=bob_node.uuid,
        created_at=now,
        group_id=group_id,
    )

    # Cross reference the ids
    episode_node.entity_edges = [entity_edge.uuid]
    entity_edge.episodes = [episode_node.uuid]

    # Save the nodes and edges
    await episode_node.save(graph_driver)
    await alice_node.save(graph_driver)
    await bob_node.save(graph_driver)
    await entity_edge.save(graph_driver)
    await episodic_alice_edge.save(graph_driver)
    await episodic_bob_edge.save(graph_driver)

    node_ids = [episode_node.uuid, alice_node.uuid, bob_node.uuid]
    edge_ids = [episodic_alice_edge.uuid, episodic_bob_edge.uuid, entity_edge.uuid]
    node_count = await get_node_count(graph_driver, node_ids)
    assert node_count == 3
    edge_count = await get_edge_count(graph_driver, edge_ids)
    assert edge_count == 3

    await graphiti.remove_episode(episode_node.uuid)
    node_count = await get_node_count(graph_driver, node_ids)
    assert node_count == 0
    edge_count = await get_edge_count(graph_driver, edge_ids)
    assert edge_count == 0

    await graphiti.close()
