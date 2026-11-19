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
from graphiti_core.namespaces.nodes import NodeNamespace
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
    group_id_2,
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
@pytest.mark.parametrize(
    ('namespace_name', 'node_label'),
    [
        ('entity', 'Entity'),
        ('episode', 'Episodic'),
        ('community', 'Community'),
        ('saga', 'Saga'),
    ],
)
@pytest.mark.parametrize('delete_by', ['group_id', 'uuids'])
async def test_namespace_batch_delete(
    graph_driver, mock_embedder, namespace_name, node_label, delete_by
):
    if graph_driver.provider != GraphProvider.NEO4J:
        pytest.skip('Neo4j namespace batch deletion regression')

    target_uuids = [str(uuid4()), str(uuid4())]
    same_group_uuid, other_group_uuid, other_label_uuid = (str(uuid4()) for _ in range(3))
    all_uuids = [*target_uuids, same_group_uuid, other_group_uuid, other_label_uuid]
    other_label = 'Episodic' if node_label == 'Entity' else 'Entity'
    nodes = [
        {'uuid': uuid, 'group_id': group_id, 'name': 'Batch delete target'}
        for uuid in [*target_uuids, same_group_uuid]
    ]
    nodes.append({'uuid': other_group_uuid, 'group_id': group_id_2, 'name': 'Other group control'})

    try:
        await graph_driver.execute_query(
            f'UNWIND $nodes AS node CREATE (n:{node_label}) SET n = node', nodes=nodes
        )
        await graph_driver.execute_query(
            f'CREATE (n:{other_label} {{uuid: $uuid, group_id: $group_id}})',
            uuid=other_label_uuid,
            group_id=group_id,
        )

        namespace = getattr(NodeNamespace(graph_driver, mock_embedder), namespace_name)
        if delete_by == 'group_id':
            await namespace.delete_by_group_id(group_id, batch_size=1)
            expected_uuids = {other_group_uuid, other_label_uuid}
        else:
            await namespace.delete_by_uuids(target_uuids, batch_size=1)
            expected_uuids = {same_group_uuid, other_group_uuid, other_label_uuid}

        records, _, _ = await graph_driver.execute_query(
            'MATCH (n) WHERE n.uuid IN $uuids RETURN n.uuid AS uuid', uuids=all_uuids
        )
        assert {record['uuid'] for record in records} == expected_uuids
    finally:
        await graph_driver.execute_query(
            'MATCH (n) WHERE n.uuid IN $uuids DETACH DELETE n', uuids=all_uuids
        )


@pytest.mark.asyncio
async def test_namespace_batch_delete_rollback(graph_driver, mock_embedder):
    if graph_driver.provider != GraphProvider.NEO4J:
        pytest.skip('Neo4j namespace batch deletion regression')

    uuid = str(uuid4())
    try:
        await graph_driver.execute_query(
            'CREATE (:Entity {uuid: $uuid, group_id: $group_id})', uuid=uuid, group_id=group_id
        )
        nodes = NodeNamespace(graph_driver, mock_embedder)
        with pytest.raises(RuntimeError, match='rollback namespace deletion'):
            async with graph_driver.transaction() as tx:
                await nodes.entity.delete_by_uuids([uuid], tx=tx, batch_size=1)
                result = await tx.run(
                    'MATCH (n:Entity {uuid: $uuid}) RETURN count(n) AS count', uuid=uuid
                )
                record = await result.single()
                assert record['count'] == 0
                raise RuntimeError('rollback namespace deletion')

        assert await get_node_count(graph_driver, [uuid]) == 1
    finally:
        await graph_driver.execute_query(
            'MATCH (n) WHERE n.uuid = $uuid DETACH DELETE n', uuid=uuid
        )
