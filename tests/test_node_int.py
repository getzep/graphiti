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
from uuid import uuid4

import numpy as np
import pytest

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    EpisodeType,
    EpisodicNode,
)
from tests.helpers_test import drivers, get_driver

group_id = f'test_group_{str(uuid4())}'


@pytest.fixture
def sample_entity_node():
    return EntityNode(
        uuid=str(uuid4()),
        name='Test Entity',
        group_id=group_id,
        labels=[],
        name_embedding=[0.5] * 1024,
        summary='Entity Summary',
    )


@pytest.fixture
def sample_episodic_node():
    return EpisodicNode(
        uuid=str(uuid4()),
        name='Episode 1',
        group_id=group_id,
        source=EpisodeType.text,
        source_description='Test source',
        content='Some content here',
        valid_at=datetime.now(),
    )


@pytest.fixture
def sample_community_node():
    return CommunityNode(
        uuid=str(uuid4()),
        name='Community A',
        name_embedding=[0.5] * 1024,
        group_id=group_id,
        summary='Community summary',
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'driver',
    drivers,
    ids=drivers,
)
async def test_entity_node(sample_entity_node, driver):
    driver = get_driver(driver)
    uuid = sample_entity_node.uuid

    # Create node
    node_count = await get_node_count(driver, uuid)
    assert node_count == 0
    await sample_entity_node.save(driver)
    node_count = await get_node_count(driver, uuid)
    assert node_count == 1

    retrieved = await EntityNode.get_by_uuid(driver, sample_entity_node.uuid)
    assert retrieved.uuid == sample_entity_node.uuid
    assert retrieved.name == 'Test Entity'
    assert retrieved.group_id == group_id

    retrieved = await EntityNode.get_by_uuids(driver, [sample_entity_node.uuid])
    assert retrieved[0].uuid == sample_entity_node.uuid
    assert retrieved[0].name == 'Test Entity'
    assert retrieved[0].group_id == group_id

    retrieved = await EntityNode.get_by_group_ids(driver, [group_id], limit=2)
    assert len(retrieved) == 1
    assert retrieved[0].uuid == sample_entity_node.uuid
    assert retrieved[0].name == 'Test Entity'
    assert retrieved[0].group_id == group_id

    await sample_entity_node.load_name_embedding(driver)
    assert np.allclose(sample_entity_node.name_embedding, [0.5] * 1024)

    # Delete node by uuid
    await sample_entity_node.delete(driver)
    node_count = await get_node_count(driver, uuid)
    assert node_count == 0

    # Delete node by group id
    await sample_entity_node.save(driver)
    node_count = await get_node_count(driver, uuid)
    assert node_count == 1
    await sample_entity_node.delete_by_group_id(driver, group_id)
    node_count = await get_node_count(driver, uuid)
    assert node_count == 0

    await driver.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'driver',
    drivers,
    ids=drivers,
)
async def test_community_node(sample_community_node, driver):
    driver = get_driver(driver)
    uuid = sample_community_node.uuid

    # Create node
    node_count = await get_node_count(driver, uuid)
    assert node_count == 0
    await sample_community_node.save(driver)
    node_count = await get_node_count(driver, uuid)
    assert node_count == 1

    retrieved = await CommunityNode.get_by_uuid(driver, sample_community_node.uuid)
    assert retrieved.uuid == sample_community_node.uuid
    assert retrieved.name == 'Community A'
    assert retrieved.group_id == group_id
    assert retrieved.summary == 'Community summary'

    retrieved = await CommunityNode.get_by_uuids(driver, [sample_community_node.uuid])
    assert retrieved[0].uuid == sample_community_node.uuid
    assert retrieved[0].name == 'Community A'
    assert retrieved[0].group_id == group_id
    assert retrieved[0].summary == 'Community summary'

    retrieved = await CommunityNode.get_by_group_ids(driver, [group_id], limit=2)
    assert len(retrieved) == 1
    assert retrieved[0].uuid == sample_community_node.uuid
    assert retrieved[0].name == 'Community A'
    assert retrieved[0].group_id == group_id

    # Delete node by uuid
    await sample_community_node.delete(driver)
    node_count = await get_node_count(driver, uuid)
    assert node_count == 0

    # Delete node by group id
    await sample_community_node.save(driver)
    node_count = await get_node_count(driver, uuid)
    assert node_count == 1
    await sample_community_node.delete_by_group_id(driver, group_id)
    node_count = await get_node_count(driver, uuid)
    assert node_count == 0

    await driver.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'driver',
    drivers,
    ids=drivers,
)
async def test_episodic_node(sample_episodic_node, driver):
    driver = get_driver(driver)
    uuid = sample_episodic_node.uuid

    # Create node
    node_count = await get_node_count(driver, uuid)
    assert node_count == 0
    await sample_episodic_node.save(driver)
    node_count = await get_node_count(driver, uuid)
    assert node_count == 1

    retrieved = await EpisodicNode.get_by_uuid(driver, sample_episodic_node.uuid)
    assert retrieved.uuid == sample_episodic_node.uuid
    assert retrieved.name == 'Episode 1'
    assert retrieved.group_id == group_id
    assert retrieved.source == EpisodeType.text
    assert retrieved.source_description == 'Test source'
    assert retrieved.content == 'Some content here'
    assert retrieved.valid_at == sample_episodic_node.valid_at

    retrieved = await EpisodicNode.get_by_uuids(driver, [sample_episodic_node.uuid])
    assert retrieved[0].uuid == sample_episodic_node.uuid
    assert retrieved[0].name == 'Episode 1'
    assert retrieved[0].group_id == group_id
    assert retrieved[0].source == EpisodeType.text
    assert retrieved[0].source_description == 'Test source'
    assert retrieved[0].content == 'Some content here'
    assert retrieved[0].valid_at == sample_episodic_node.valid_at

    retrieved = await EpisodicNode.get_by_group_ids(driver, [group_id], limit=2)
    assert len(retrieved) == 1
    assert retrieved[0].uuid == sample_episodic_node.uuid
    assert retrieved[0].name == 'Episode 1'
    assert retrieved[0].group_id == group_id
    assert retrieved[0].source == EpisodeType.text
    assert retrieved[0].source_description == 'Test source'
    assert retrieved[0].content == 'Some content here'
    assert retrieved[0].valid_at == sample_episodic_node.valid_at

    # Delete node by uuid
    await sample_episodic_node.delete(driver)
    node_count = await get_node_count(driver, uuid)
    assert node_count == 0

    # Delete node by group id
    await sample_episodic_node.save(driver)
    node_count = await get_node_count(driver, uuid)
    assert node_count == 1
    await sample_episodic_node.delete_by_group_id(driver, group_id)
    node_count = await get_node_count(driver, uuid)
    assert node_count == 0

    await driver.close()


async def get_node_count(driver: GraphDriver, uuid: str):
    result, _, _ = await driver.execute_query(
        """
        MATCH (n {uuid: $uuid})
        RETURN COUNT(n) as count
        """,
        uuid=uuid,
    )
    return int(result[0]['count'])
