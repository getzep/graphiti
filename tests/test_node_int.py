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

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    EpisodeType,
    EpisodicNode,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import node_similarity_search
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
@pytest.mark.parametrize('bulk', [False, True], ids=['direct', 'bulk'])
async def test_neo4j_entity_save_with_optional_embedding(sample_entity_node, graph_driver, bulk):
    if graph_driver.provider != GraphProvider.NEO4J:
        pytest.skip('This regression exercises Neo4j vector writes')

    async def save(*nodes: EntityNode):
        if bulk:
            ops = graph_driver.entity_node_ops
            assert ops is not None
            await ops.save_bulk(graph_driver, list(nodes))
        else:
            for node in nodes:
                await node.save(graph_driver)

    unembedded = EntityNode(name='Unembedded entity', group_id=group_id)
    await save(unembedded, sample_entity_node)

    retrieved = await EntityNode.get_by_uuid(graph_driver, unembedded.uuid)
    await retrieved.load_name_embedding(graph_driver)
    assert retrieved.name_embedding is None

    # Default reads omit embeddings. Saving an edited node must keep its stored vector.
    retrieved = await EntityNode.get_by_uuid(graph_driver, sample_entity_node.uuid)
    assert retrieved.name_embedding is None
    retrieved.summary = 'Updated summary'
    await save(retrieved)

    updated = await EntityNode.get_by_uuid(graph_driver, sample_entity_node.uuid)
    assert updated.summary == 'Updated summary'
    await updated.load_name_embedding(graph_driver)
    assert updated.name_embedding == sample_entity_node.name_embedding

    updated.name_embedding = [0.25] * 1024
    await save(updated)
    reloaded = await EntityNode.get_by_uuid(graph_driver, updated.uuid)
    await reloaded.load_name_embedding(graph_driver)
    assert reloaded.name_embedding == updated.name_embedding

    matches = await node_similarity_search(
        graph_driver, updated.name_embedding, SearchFilters(), group_ids=[group_id]
    )
    assert [node.uuid for node in matches] == [updated.uuid]


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
