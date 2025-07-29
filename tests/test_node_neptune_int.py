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
import unittest
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    EpisodeType,
    EpisodicNode,
)

NEPTUNE_HOST = os.getenv('NEPTUNE_HOST', 'localhost')
AOSS_HOST = os.getenv('AOSS_HOST', None)

try:
    from graphiti_core.driver.neptune_driver import NeptuneDriver

    HAS_NEPTUNE = True
except ImportError:
    NeptuneDriver = None
    HAS_NEPTUNE = False


@pytest.fixture
def sample_entity_node():
    return EntityNode(
        uuid=str(uuid4()),
        name='Test Entity',
        group_id='test_group',
        labels=['Entity'],
        name_embedding=[0.5] * 1024,
        summary='Entity Summary',
    )


@pytest.fixture
def sample_episodic_node():
    return EpisodicNode(
        uuid=str(uuid4()),
        name='Episode 1',
        group_id='test_group',
        source=EpisodeType.text,
        source_description='Test source',
        content='Some content here',
        valid_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_community_node():
    return CommunityNode(
        uuid=str(uuid4()),
        name='Community A',
        name_embedding=[0.5] * 1024,
        group_id='test_group',
        summary='Community summary',
    )


@pytest.mark.asyncio
@pytest.mark.integration
@unittest.skipIf(not HAS_NEPTUNE, 'Neptune is not installed')
async def test_entity_node_save_get_and_delete(sample_entity_node):
    neptune_driver = NeptuneDriver(host=NEPTUNE_HOST, aoss_host=AOSS_HOST)

    await sample_entity_node.save(neptune_driver)

    retrieved = await EntityNode.get_by_uuid(neptune_driver, sample_entity_node.uuid)
    assert retrieved.uuid == sample_entity_node.uuid
    assert retrieved.name == 'Test Entity'
    assert retrieved.group_id == 'test_group'

    await sample_entity_node.delete(neptune_driver)
    await neptune_driver.close()


@pytest.mark.asyncio
@pytest.mark.integration
@unittest.skipIf(not HAS_NEPTUNE, 'Neptune is not installed')
async def test_community_node_save_get_and_delete(sample_community_node):
    neptune_driver = NeptuneDriver(host=NEPTUNE_HOST, aoss_host=AOSS_HOST)

    await sample_community_node.save(neptune_driver)

    retrieved = await CommunityNode.get_by_uuid(neptune_driver, sample_community_node.uuid)
    assert retrieved.uuid == sample_community_node.uuid
    assert retrieved.name == 'Community A'
    assert retrieved.group_id == 'test_group'
    assert retrieved.summary == 'Community summary'

    await sample_community_node.delete(neptune_driver)
    await neptune_driver.close()


@pytest.mark.asyncio
@pytest.mark.integration
@unittest.skipIf(not HAS_NEPTUNE, 'Neptune is not installed')
async def test_episodic_node_save_get_and_delete(sample_episodic_node):
    neptune_driver = NeptuneDriver(host=NEPTUNE_HOST, aoss_host=AOSS_HOST)

    await sample_episodic_node.save(neptune_driver)

    retrieved = await EpisodicNode.get_by_uuid(neptune_driver, sample_episodic_node.uuid)
    assert retrieved.uuid == sample_episodic_node.uuid
    assert retrieved.name == 'Episode 1'
    assert retrieved.group_id == 'test_group'
    assert retrieved.source == EpisodeType.text
    assert retrieved.source_description == 'Test source'
    assert retrieved.content == 'Some content here'

    await sample_episodic_node.delete(neptune_driver)
    await neptune_driver.close()
