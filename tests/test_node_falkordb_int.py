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

FALKORDB_HOST = os.getenv('FALKORDB_HOST', 'localhost')
FALKORDB_PORT = os.getenv('FALKORDB_PORT', '6379')
FALKORDB_USER = os.getenv('FALKORDB_USER', None)
FALKORDB_PASSWORD = os.getenv('FALKORDB_PASSWORD', None)

try:
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    HAS_FALKORDB = True
except ImportError:
    FalkorDriver = None
    HAS_FALKORDB = False


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
@unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
async def test_entity_node_save_get_and_delete(sample_entity_node):
    falkor_driver = FalkorDriver(
        host=FALKORDB_HOST, port=FALKORDB_PORT, username=FALKORDB_USER, password=FALKORDB_PASSWORD
    )

    await sample_entity_node.save(falkor_driver)

    retrieved = await EntityNode.get_by_uuid(falkor_driver, sample_entity_node.uuid)
    assert retrieved.uuid == sample_entity_node.uuid
    assert retrieved.name == 'Test Entity'
    assert retrieved.group_id == 'test_group'

    await sample_entity_node.delete(falkor_driver)
    await falkor_driver.close()


@pytest.mark.asyncio
@pytest.mark.integration
@unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
async def test_community_node_save_get_and_delete(sample_community_node):
    falkor_driver = FalkorDriver(
        host=FALKORDB_HOST, port=FALKORDB_PORT, username=FALKORDB_USER, password=FALKORDB_PASSWORD
    )

    await sample_community_node.save(falkor_driver)

    retrieved = await CommunityNode.get_by_uuid(falkor_driver, sample_community_node.uuid)
    assert retrieved.uuid == sample_community_node.uuid
    assert retrieved.name == 'Community A'
    assert retrieved.group_id == 'test_group'
    assert retrieved.summary == 'Community summary'

    await sample_community_node.delete(falkor_driver)
    await falkor_driver.close()


@pytest.mark.asyncio
@pytest.mark.integration
@unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
async def test_episodic_node_save_get_and_delete(sample_episodic_node):
    falkor_driver = FalkorDriver(
        host=FALKORDB_HOST, port=FALKORDB_PORT, username=FALKORDB_USER, password=FALKORDB_PASSWORD
    )

    await sample_episodic_node.save(falkor_driver)

    retrieved = await EpisodicNode.get_by_uuid(falkor_driver, sample_episodic_node.uuid)
    assert retrieved.uuid == sample_episodic_node.uuid
    assert retrieved.name == 'Episode 1'
    assert retrieved.group_id == 'test_group'
    assert retrieved.source == EpisodeType.text
    assert retrieved.source_description == 'Test source'
    assert retrieved.content == 'Some content here'

    await sample_episodic_node.delete(falkor_driver)
    await falkor_driver.close()
