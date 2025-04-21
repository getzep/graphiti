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
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from neo4j import AsyncGraphDatabase

from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    EpisodeType,
    EpisodicNode,
)

NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'test')


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
async def test_entity_node_save_get_and_delete(sample_entity_node):
    neo4j_driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    await sample_entity_node.save(neo4j_driver)
    retrieved = await EntityNode.get_by_uuid(neo4j_driver, sample_entity_node.uuid)
    assert retrieved.uuid == sample_entity_node.uuid
    assert retrieved.name == 'Test Entity'
    assert retrieved.group_id == 'test_group'

    await sample_entity_node.delete(neo4j_driver)

    await neo4j_driver.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_community_node_save_get_and_delete(sample_community_node):
    neo4j_driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    await sample_community_node.save(neo4j_driver)

    retrieved = await CommunityNode.get_by_uuid(neo4j_driver, sample_community_node.uuid)
    assert retrieved.uuid == sample_community_node.uuid
    assert retrieved.name == 'Community A'
    assert retrieved.group_id == 'test_group'
    assert retrieved.summary == 'Community summary'

    await sample_community_node.delete(neo4j_driver)

    await neo4j_driver.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_episodic_node_save_get_and_delete(sample_episodic_node):
    neo4j_driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    await sample_episodic_node.save(neo4j_driver)

    retrieved = await EpisodicNode.get_by_uuid(neo4j_driver, sample_episodic_node.uuid)
    assert retrieved.uuid == sample_episodic_node.uuid
    assert retrieved.name == 'Episode 1'
    assert retrieved.group_id == 'test_group'
    assert retrieved.source == EpisodeType.text
    assert retrieved.source_description == 'Test source'
    assert retrieved.content == 'Some content here'

    await sample_episodic_node.delete(neo4j_driver)

    await neo4j_driver.close()
