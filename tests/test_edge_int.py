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
import sys
from datetime import datetime

import numpy as np
import pytest

from graphiti_core.edges import CommunityEdge, EntityEdge, EpisodicEdge
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode
from tests.helpers_test import get_edge_count, get_node_count, group_id

pytest_plugins = ('pytest_asyncio',)
pytestmark = pytest.mark.integration


def setup_logging():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the logging level to INFO

    # Create console handler and set level to INFO
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add formatter to console handler
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    return logger


@pytest.mark.asyncio
async def test_episodic_edge(graph_driver, mock_embedder):
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
    node_count = await get_node_count(graph_driver, [episode_node.uuid])
    assert node_count == 0
    await episode_node.save(graph_driver)
    node_count = await get_node_count(graph_driver, [episode_node.uuid])
    assert node_count == 1

    # Create entity node
    alice_node = EntityNode(
        name='Alice',
        labels=[],
        created_at=now,
        summary='Alice summary',
        group_id=group_id,
    )
    await alice_node.generate_name_embedding(mock_embedder)
    node_count = await get_node_count(graph_driver, [alice_node.uuid])
    assert node_count == 0
    await alice_node.save(graph_driver)
    node_count = await get_node_count(graph_driver, [alice_node.uuid])
    assert node_count == 1

    # Create episodic to entity edge
    episodic_edge = EpisodicEdge(
        source_node_uuid=episode_node.uuid,
        target_node_uuid=alice_node.uuid,
        created_at=now,
        group_id=group_id,
    )
    edge_count = await get_edge_count(graph_driver, [episodic_edge.uuid])
    assert edge_count == 0
    await episodic_edge.save(graph_driver)
    edge_count = await get_edge_count(graph_driver, [episodic_edge.uuid])
    assert edge_count == 1

    # Get edge by uuid
    retrieved = await EpisodicEdge.get_by_uuid(graph_driver, episodic_edge.uuid)
    assert retrieved.uuid == episodic_edge.uuid
    assert retrieved.source_node_uuid == episode_node.uuid
    assert retrieved.target_node_uuid == alice_node.uuid
    assert retrieved.created_at == now
    assert retrieved.group_id == group_id

    # Get edge by uuids
    retrieved = await EpisodicEdge.get_by_uuids(graph_driver, [episodic_edge.uuid])
    assert len(retrieved) == 1
    assert retrieved[0].uuid == episodic_edge.uuid
    assert retrieved[0].source_node_uuid == episode_node.uuid
    assert retrieved[0].target_node_uuid == alice_node.uuid
    assert retrieved[0].created_at == now
    assert retrieved[0].group_id == group_id

    # Get edge by group ids
    retrieved = await EpisodicEdge.get_by_group_ids(graph_driver, [group_id], limit=2)
    assert len(retrieved) == 1
    assert retrieved[0].uuid == episodic_edge.uuid
    assert retrieved[0].source_node_uuid == episode_node.uuid
    assert retrieved[0].target_node_uuid == alice_node.uuid
    assert retrieved[0].created_at == now
    assert retrieved[0].group_id == group_id

    # Get episodic node by entity node uuid
    retrieved = await EpisodicNode.get_by_entity_node_uuid(graph_driver, alice_node.uuid)
    assert len(retrieved) == 1
    assert retrieved[0].uuid == episode_node.uuid
    assert retrieved[0].name == 'test_episode'
    assert retrieved[0].created_at == now
    assert retrieved[0].group_id == group_id

    # Delete edge by uuid
    await episodic_edge.delete(graph_driver)
    edge_count = await get_edge_count(graph_driver, [episodic_edge.uuid])
    assert edge_count == 0

    # Delete edge by uuids
    await episodic_edge.save(graph_driver)
    await episodic_edge.delete_by_uuids(graph_driver, [episodic_edge.uuid])
    edge_count = await get_edge_count(graph_driver, [episodic_edge.uuid])
    assert edge_count == 0

    # Cleanup nodes
    await episode_node.delete(graph_driver)
    node_count = await get_node_count(graph_driver, [episode_node.uuid])
    assert node_count == 0
    await alice_node.delete(graph_driver)
    node_count = await get_node_count(graph_driver, [alice_node.uuid])
    assert node_count == 0

    await graph_driver.close()


@pytest.mark.asyncio
async def test_entity_edge(graph_driver, mock_embedder):
    now = datetime.now()

    # Create entity node
    alice_node = EntityNode(
        name='Alice',
        labels=[],
        created_at=now,
        summary='Alice summary',
        group_id=group_id,
    )
    await alice_node.generate_name_embedding(mock_embedder)
    node_count = await get_node_count(graph_driver, [alice_node.uuid])
    assert node_count == 0
    await alice_node.save(graph_driver)
    node_count = await get_node_count(graph_driver, [alice_node.uuid])
    assert node_count == 1

    # Create entity node
    bob_node = EntityNode(
        name='Bob', labels=[], created_at=now, summary='Bob summary', group_id=group_id
    )
    await bob_node.generate_name_embedding(mock_embedder)
    node_count = await get_node_count(graph_driver, [bob_node.uuid])
    assert node_count == 0
    await bob_node.save(graph_driver)
    node_count = await get_node_count(graph_driver, [bob_node.uuid])
    assert node_count == 1

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
    edge_embedding = await entity_edge.generate_embedding(mock_embedder)
    edge_count = await get_edge_count(graph_driver, [entity_edge.uuid])
    assert edge_count == 0
    await entity_edge.save(graph_driver)
    edge_count = await get_edge_count(graph_driver, [entity_edge.uuid])
    assert edge_count == 1

    # Get edge by uuid
    retrieved = await EntityEdge.get_by_uuid(graph_driver, entity_edge.uuid)
    assert retrieved.uuid == entity_edge.uuid
    assert retrieved.source_node_uuid == alice_node.uuid
    assert retrieved.target_node_uuid == bob_node.uuid
    assert retrieved.created_at == now
    assert retrieved.group_id == group_id

    # Get edge by uuids
    retrieved = await EntityEdge.get_by_uuids(graph_driver, [entity_edge.uuid])
    assert len(retrieved) == 1
    assert retrieved[0].uuid == entity_edge.uuid
    assert retrieved[0].source_node_uuid == alice_node.uuid
    assert retrieved[0].target_node_uuid == bob_node.uuid
    assert retrieved[0].created_at == now
    assert retrieved[0].group_id == group_id

    # Get edge by group ids
    retrieved = await EntityEdge.get_by_group_ids(graph_driver, [group_id], limit=2)
    assert len(retrieved) == 1
    assert retrieved[0].uuid == entity_edge.uuid
    assert retrieved[0].source_node_uuid == alice_node.uuid
    assert retrieved[0].target_node_uuid == bob_node.uuid
    assert retrieved[0].created_at == now
    assert retrieved[0].group_id == group_id

    # Get edge by node uuid
    retrieved = await EntityEdge.get_by_node_uuid(graph_driver, alice_node.uuid)
    assert len(retrieved) == 1
    assert retrieved[0].uuid == entity_edge.uuid
    assert retrieved[0].source_node_uuid == alice_node.uuid
    assert retrieved[0].target_node_uuid == bob_node.uuid
    assert retrieved[0].created_at == now
    assert retrieved[0].group_id == group_id

    # Get fact embedding
    await entity_edge.load_fact_embedding(graph_driver)
    assert np.allclose(entity_edge.fact_embedding, edge_embedding)

    # Delete edge by uuid
    await entity_edge.delete(graph_driver)
    edge_count = await get_edge_count(graph_driver, [entity_edge.uuid])
    assert edge_count == 0

    # Delete edge by uuids
    await entity_edge.save(graph_driver)
    await entity_edge.delete_by_uuids(graph_driver, [entity_edge.uuid])
    edge_count = await get_edge_count(graph_driver, [entity_edge.uuid])
    assert edge_count == 0

    # Deleting node should delete the edge
    await entity_edge.save(graph_driver)
    await alice_node.delete(graph_driver)
    node_count = await get_node_count(graph_driver, [alice_node.uuid])
    assert node_count == 0
    edge_count = await get_edge_count(graph_driver, [entity_edge.uuid])
    assert edge_count == 0

    # Deleting node by uuids should delete the edge
    await alice_node.save(graph_driver)
    await entity_edge.save(graph_driver)
    await alice_node.delete_by_uuids(graph_driver, [alice_node.uuid])
    node_count = await get_node_count(graph_driver, [alice_node.uuid])
    assert node_count == 0
    edge_count = await get_edge_count(graph_driver, [entity_edge.uuid])
    assert edge_count == 0

    # Deleting node by group id should delete the edge
    await alice_node.save(graph_driver)
    await entity_edge.save(graph_driver)
    await alice_node.delete_by_group_id(graph_driver, alice_node.group_id)
    node_count = await get_node_count(graph_driver, [alice_node.uuid])
    assert node_count == 0
    edge_count = await get_edge_count(graph_driver, [entity_edge.uuid])
    assert edge_count == 0

    # Cleanup nodes
    await alice_node.delete(graph_driver)
    node_count = await get_node_count(graph_driver, [alice_node.uuid])
    assert node_count == 0
    await bob_node.delete(graph_driver)
    node_count = await get_node_count(graph_driver, [bob_node.uuid])
    assert node_count == 0

    await graph_driver.close()


@pytest.mark.asyncio
async def test_community_edge(graph_driver, mock_embedder):
    now = datetime.now()

    # Create community node
    community_node_1 = CommunityNode(
        name='test_community_1',
        group_id=group_id,
        summary='Community A summary',
    )
    await community_node_1.generate_name_embedding(mock_embedder)
    node_count = await get_node_count(graph_driver, [community_node_1.uuid])
    assert node_count == 0
    await community_node_1.save(graph_driver)
    node_count = await get_node_count(graph_driver, [community_node_1.uuid])
    assert node_count == 1

    # Create community node
    community_node_2 = CommunityNode(
        name='test_community_2',
        group_id=group_id,
        summary='Community B summary',
    )
    await community_node_2.generate_name_embedding(mock_embedder)
    node_count = await get_node_count(graph_driver, [community_node_2.uuid])
    assert node_count == 0
    await community_node_2.save(graph_driver)
    node_count = await get_node_count(graph_driver, [community_node_2.uuid])
    assert node_count == 1

    # Create entity node
    alice_node = EntityNode(
        name='Alice', labels=[], created_at=now, summary='Alice summary', group_id=group_id
    )
    await alice_node.generate_name_embedding(mock_embedder)
    node_count = await get_node_count(graph_driver, [alice_node.uuid])
    assert node_count == 0
    await alice_node.save(graph_driver)
    node_count = await get_node_count(graph_driver, [alice_node.uuid])
    assert node_count == 1

    # Create community to community edge
    community_edge = CommunityEdge(
        source_node_uuid=community_node_1.uuid,
        target_node_uuid=community_node_2.uuid,
        created_at=now,
        group_id=group_id,
    )
    edge_count = await get_edge_count(graph_driver, [community_edge.uuid])
    assert edge_count == 0
    await community_edge.save(graph_driver)
    edge_count = await get_edge_count(graph_driver, [community_edge.uuid])
    assert edge_count == 1

    # Get edge by uuid
    retrieved = await CommunityEdge.get_by_uuid(graph_driver, community_edge.uuid)
    assert retrieved.uuid == community_edge.uuid
    assert retrieved.source_node_uuid == community_node_1.uuid
    assert retrieved.target_node_uuid == community_node_2.uuid
    assert retrieved.created_at == now
    assert retrieved.group_id == group_id

    # Get edge by uuids
    retrieved = await CommunityEdge.get_by_uuids(graph_driver, [community_edge.uuid])
    assert len(retrieved) == 1
    assert retrieved[0].uuid == community_edge.uuid
    assert retrieved[0].source_node_uuid == community_node_1.uuid
    assert retrieved[0].target_node_uuid == community_node_2.uuid
    assert retrieved[0].created_at == now
    assert retrieved[0].group_id == group_id

    # Get edge by group ids
    retrieved = await CommunityEdge.get_by_group_ids(graph_driver, [group_id], limit=1)
    assert len(retrieved) == 1
    assert retrieved[0].uuid == community_edge.uuid
    assert retrieved[0].source_node_uuid == community_node_1.uuid
    assert retrieved[0].target_node_uuid == community_node_2.uuid
    assert retrieved[0].created_at == now
    assert retrieved[0].group_id == group_id

    # Delete edge by uuid
    await community_edge.delete(graph_driver)
    edge_count = await get_edge_count(graph_driver, [community_edge.uuid])
    assert edge_count == 0

    # Delete edge by uuids
    await community_edge.save(graph_driver)
    await community_edge.delete_by_uuids(graph_driver, [community_edge.uuid])
    edge_count = await get_edge_count(graph_driver, [community_edge.uuid])
    assert edge_count == 0

    # Cleanup nodes
    await alice_node.delete(graph_driver)
    node_count = await get_node_count(graph_driver, [alice_node.uuid])
    assert node_count == 0
    await community_node_1.delete(graph_driver)
    node_count = await get_node_count(graph_driver, [community_node_1.uuid])
    assert node_count == 0
    await community_node_2.delete(graph_driver)
    node_count = await get_node_count(graph_driver, [community_node_2.uuid])
    assert node_count == 0

    await graph_driver.close()
