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
from uuid import uuid4

import pytest

from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    EpisodeType,
    EpisodicNode,
)
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from graphiti_core.utils.maintenance.graph_data_operations import retrieve_episodes
from tests.helpers_test import (
    assert_community_node_equals,
    assert_entity_node_equals,
    assert_episodic_node_equals,
    get_node_count,
    group_id,
)

created_at = datetime.now()
deleted_at = created_at + timedelta(days=3)
valid_at = created_at + timedelta(days=1)
invalid_at = created_at + timedelta(days=2)


@pytest.fixture
def sample_entity_node():
    return EntityNode(
        uuid=str(uuid4()),
        name='Test Entity',
        group_id=group_id,
        labels=['Entity', 'Person'],
        created_at=created_at,
        name_embedding=[0.5] * 1024,
        summary='Entity Summary',
        attributes={
            'age': 30,
            'location': 'New York',
        },
    )


@pytest.fixture
def sample_episodic_node():
    return EpisodicNode(
        uuid=str(uuid4()),
        name='Episode 1',
        group_id=group_id,
        created_at=created_at,
        source=EpisodeType.text,
        source_description='Test source',
        content='Some content here',
        valid_at=valid_at,
        entity_edges=[],
    )


EPISODE_METADATA = {'agent': 'planner', 'step': 3, 'score': 0.5, 'approved': True}


def episodic_node_with_metadata() -> EpisodicNode:
    return EpisodicNode(
        uuid=str(uuid4()),
        name='Episode with metadata',
        group_id=group_id,
        created_at=created_at,
        source=EpisodeType.text,
        source_description='Test source',
        content='Some content here',
        valid_at=valid_at,
        entity_edges=[],
        episode_metadata=EPISODE_METADATA,
    )


async def assert_reads_metadata(graph_driver, uuid: str, expected: dict | None):
    reads = [
        await EpisodicNode.get_by_uuid(graph_driver, uuid),
        (await EpisodicNode.get_by_uuids(graph_driver, [uuid]))[0],
        (await EpisodicNode.get_by_group_ids(graph_driver, [group_id], limit=1))[0],
        (await retrieve_episodes(graph_driver, datetime.now() + timedelta(days=2), 1, [group_id]))[
            0
        ],
        await graph_driver.episode_node_ops.get_by_uuid(graph_driver, uuid),
    ]
    assert [episode.episode_metadata for episode in reads] == [expected] * len(reads)


@pytest.fixture
def sample_community_node():
    return CommunityNode(
        uuid=str(uuid4()),
        name='Community A',
        group_id=group_id,
        created_at=created_at,
        name_embedding=[0.5] * 1024,
        summary='Community summary',
    )


@pytest.mark.asyncio
async def test_entity_node(sample_entity_node, graph_driver):
    uuid = sample_entity_node.uuid

    # Create node
    node_count = await get_node_count(graph_driver, [uuid])
    assert node_count == 0
    await sample_entity_node.save(graph_driver)
    node_count = await get_node_count(graph_driver, [uuid])
    assert node_count == 1

    # Get node by uuid
    retrieved = await EntityNode.get_by_uuid(graph_driver, sample_entity_node.uuid)
    await assert_entity_node_equals(graph_driver, retrieved, sample_entity_node)

    # Get node by uuids
    retrieved = await EntityNode.get_by_uuids(graph_driver, [sample_entity_node.uuid])
    await assert_entity_node_equals(graph_driver, retrieved[0], sample_entity_node)

    # Get node by group ids
    retrieved = await EntityNode.get_by_group_ids(
        graph_driver, [group_id], limit=2, with_embeddings=True
    )
    assert len(retrieved) == 1
    await assert_entity_node_equals(graph_driver, retrieved[0], sample_entity_node)

    # Delete node by uuid
    await sample_entity_node.delete(graph_driver)
    node_count = await get_node_count(graph_driver, [uuid])
    assert node_count == 0

    # Delete node by uuids
    await sample_entity_node.save(graph_driver)
    node_count = await get_node_count(graph_driver, [uuid])
    assert node_count == 1
    await sample_entity_node.delete_by_uuids(graph_driver, [uuid])
    node_count = await get_node_count(graph_driver, [uuid])
    assert node_count == 0

    # Delete node by group id
    await sample_entity_node.save(graph_driver)
    node_count = await get_node_count(graph_driver, [uuid])
    assert node_count == 1
    await sample_entity_node.delete_by_group_id(graph_driver, group_id)
    node_count = await get_node_count(graph_driver, [uuid])
    assert node_count == 0

    await graph_driver.close()


@pytest.mark.asyncio
async def test_community_node(sample_community_node, graph_driver):
    uuid = sample_community_node.uuid

    # Create node
    node_count = await get_node_count(graph_driver, [uuid])
    assert node_count == 0
    await sample_community_node.save(graph_driver)
    node_count = await get_node_count(graph_driver, [uuid])
    assert node_count == 1

    # Get node by uuid
    retrieved = await CommunityNode.get_by_uuid(graph_driver, sample_community_node.uuid)
    await assert_community_node_equals(graph_driver, retrieved, sample_community_node)

    # Get node by uuids
    retrieved = await CommunityNode.get_by_uuids(graph_driver, [sample_community_node.uuid])
    await assert_community_node_equals(graph_driver, retrieved[0], sample_community_node)

    # Get node by group ids
    retrieved = await CommunityNode.get_by_group_ids(graph_driver, [group_id], limit=2)
    assert len(retrieved) == 1
    await assert_community_node_equals(graph_driver, retrieved[0], sample_community_node)

    # Delete node by uuid
    await sample_community_node.delete(graph_driver)
    node_count = await get_node_count(graph_driver, [uuid])
    assert node_count == 0

    # Delete node by uuids
    await sample_community_node.save(graph_driver)
    node_count = await get_node_count(graph_driver, [uuid])
    assert node_count == 1
    await sample_community_node.delete_by_uuids(graph_driver, [uuid])
    node_count = await get_node_count(graph_driver, [uuid])
    assert node_count == 0

    # Delete node by group id
    await sample_community_node.save(graph_driver)
    node_count = await get_node_count(graph_driver, [uuid])
    assert node_count == 1
    await sample_community_node.delete_by_group_id(graph_driver, group_id)
    node_count = await get_node_count(graph_driver, [uuid])
    assert node_count == 0

    await graph_driver.close()


@pytest.mark.asyncio
async def test_episodic_node(sample_episodic_node, graph_driver):
    uuid = sample_episodic_node.uuid

    # Create node
    node_count = await get_node_count(graph_driver, [uuid])
    assert node_count == 0
    await sample_episodic_node.save(graph_driver)
    node_count = await get_node_count(graph_driver, [uuid])
    assert node_count == 1

    # Get node by uuid
    retrieved = await EpisodicNode.get_by_uuid(graph_driver, sample_episodic_node.uuid)
    await assert_episodic_node_equals(retrieved, sample_episodic_node)

    # Get node by uuids
    retrieved = await EpisodicNode.get_by_uuids(graph_driver, [sample_episodic_node.uuid])
    await assert_episodic_node_equals(retrieved[0], sample_episodic_node)

    # Get node by group ids
    retrieved = await EpisodicNode.get_by_group_ids(graph_driver, [group_id], limit=2)
    assert len(retrieved) == 1
    await assert_episodic_node_equals(retrieved[0], sample_episodic_node)

    # Delete node by uuid
    await sample_episodic_node.delete(graph_driver)
    node_count = await get_node_count(graph_driver, [uuid])
    assert node_count == 0

    # Delete node by uuids
    await sample_episodic_node.save(graph_driver)
    node_count = await get_node_count(graph_driver, [uuid])
    assert node_count == 1
    await sample_episodic_node.delete_by_uuids(graph_driver, [uuid])
    node_count = await get_node_count(graph_driver, [uuid])
    assert node_count == 0

    # Delete node by group id
    await sample_episodic_node.save(graph_driver)
    node_count = await get_node_count(graph_driver, [uuid])
    assert node_count == 1
    await sample_episodic_node.delete_by_group_id(graph_driver, group_id)
    node_count = await get_node_count(graph_driver, [uuid])
    assert node_count == 0

    await graph_driver.close()


@pytest.mark.asyncio
async def test_episodic_node_metadata_round_trip(graph_driver):
    episode = episodic_node_with_metadata()

    await episode.save(graph_driver)
    await assert_reads_metadata(graph_driver, episode.uuid, EPISODE_METADATA)

    reloaded = await EpisodicNode.get_by_uuid(graph_driver, episode.uuid)
    await reloaded.save(graph_driver)
    await assert_reads_metadata(graph_driver, episode.uuid, EPISODE_METADATA)

    await graph_driver.close()


@pytest.mark.asyncio
async def test_episodic_node_metadata_driver_ops_round_trip(graph_driver):
    episode = episodic_node_with_metadata()

    await graph_driver.episode_node_ops.save(graph_driver, episode)
    await assert_reads_metadata(graph_driver, episode.uuid, EPISODE_METADATA)

    await graph_driver.close()


@pytest.mark.asyncio
async def test_episodic_node_metadata_bulk_round_trip(graph_driver, mock_embedder):
    episode = episodic_node_with_metadata()

    await add_nodes_and_edges_bulk(graph_driver, [episode], [], [], [], mock_embedder)
    await assert_reads_metadata(graph_driver, episode.uuid, EPISODE_METADATA)

    await graph_driver.close()


@pytest.mark.asyncio
async def test_episodic_node_without_metadata_reads_none(sample_episodic_node, graph_driver):
    await sample_episodic_node.save(graph_driver)
    await assert_reads_metadata(graph_driver, sample_episodic_node.uuid, None)

    await graph_driver.close()
