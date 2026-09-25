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
async def test_episodic_node_recency_ordering(graph_driver):
    """Three episodes whose uuid, created_at, and valid_at orders all disagree, so each
    ordering is distinguishable from the other two.

    The middle episode is a backfill: ingested most recently, but describing the oldest
    events. That is the case that makes created_at and valid_at differ, and the reason
    valid_at is the default.
    """
    episodes = [
        # uuid rank 3 (highest), created_at rank 2, valid_at rank 1 (newest events)
        EpisodicNode(
            uuid=f'3-{uuid4()}',
            name='Most recent events',
            group_id=group_id,
            created_at=created_at - timedelta(days=2),
            source=EpisodeType.text,
            source_description='Test source',
            content='Happened last, ingested second.',
            valid_at=created_at,
            entity_edges=[],
        ),
        # uuid rank 2, created_at rank 1 (newest ingest), valid_at rank 3 (oldest events)
        EpisodicNode(
            uuid=f'2-{uuid4()}',
            name='Backfilled old events',
            group_id=group_id,
            created_at=created_at,
            source=EpisodeType.text,
            source_description='Test source',
            content='Happened first, ingested last.',
            valid_at=created_at - timedelta(days=30),
            entity_edges=[],
        ),
        # uuid rank 1 (lowest), created_at rank 3 (oldest ingest), valid_at rank 2
        EpisodicNode(
            uuid=f'1-{uuid4()}',
            name='Middling events',
            group_id=group_id,
            created_at=created_at - timedelta(days=4),
            source=EpisodeType.text,
            source_description='Test source',
            content='Happened second, ingested first.',
            valid_at=created_at - timedelta(days=15),
            entity_edges=[],
        ),
    ]
    for episode in episodes:
        await episode.save(graph_driver)

    by_name = {episode.name: episode.uuid for episode in episodes}

    # Default: event time, newest first.
    retrieved = await EpisodicNode.get_recent_by_group_ids(graph_driver, [group_id])
    assert [episode.uuid for episode in retrieved] == [
        by_name['Most recent events'],
        by_name['Middling events'],
        by_name['Backfilled old events'],
    ]

    # Ingest time puts the backfill first — the ordering that answers "did my write land?"
    retrieved = await EpisodicNode.get_recent_by_group_ids(
        graph_driver, [group_id], order_by='created_at'
    )
    assert [episode.uuid for episode in retrieved] == [
        by_name['Backfilled old events'],
        by_name['Most recent events'],
        by_name['Middling events'],
    ]

    # The uuid ordering is still reachable, and still matches the pagination helper's.
    retrieved = await EpisodicNode.get_recent_by_group_ids(
        graph_driver, [group_id], order_by='uuid'
    )
    assert [episode.uuid for episode in retrieved] == sorted(by_name.values(), reverse=True)
    paginated = await EpisodicNode.get_by_group_ids(graph_driver, [group_id])
    assert [episode.uuid for episode in retrieved] == [episode.uuid for episode in paginated]

    # The limit is applied after ordering, so a truncated page is the newest episodes —
    # not an arbitrary subset that a caller could re-sort into the right answer.
    retrieved = await EpisodicNode.get_recent_by_group_ids(graph_driver, [group_id], limit=2)
    assert [episode.uuid for episode in retrieved] == [
        by_name['Most recent events'],
        by_name['Middling events'],
    ]

    await EpisodicNode.delete_by_group_id(graph_driver, group_id)
    await graph_driver.close()
