import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from neo4j import AsyncGraphDatabase

from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    EpisodeType,
    EpisodicNode,
)

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')


@pytest.fixture(scope='module')
async def neo4j_driver():
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    yield driver
    await driver.close()


@pytest.fixture
def sample_entity_node():
    return EntityNode(
        name='Test Entity', group_id='group1', labels=['Entity'], summary='Entity Summary'
    )


@pytest.fixture
def sample_episodic_node():
    return EpisodicNode(
        name='Episode 1',
        group_id='group1',
        source=EpisodeType.text,
        source_description='Test source',
        content='Some content here',
        valid_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_community_node():
    return CommunityNode(
        name='Community A',
        group_id='group1',
        summary='Community summary',
    )


@pytest.mark.asyncio
async def test_entity_node_save(neo4j_driver, sample_entity_node):
    neo4j_driver.execute_query.return_value = 'OK'

    result = await sample_entity_node.save(neo4j_driver)

    neo4j_driver.execute_query.assert_awaited_once()
    assert result == 'OK'


@pytest.mark.asyncio
async def test_episodic_node_save(neo4j_driver, sample_episodic_node):
    neo4j_driver.execute_query.return_value = 'OK'

    result = await sample_episodic_node.save(neo4j_driver)

    neo4j_driver.execute_query.assert_awaited_once()
    assert result == 'OK'


@pytest.mark.asyncio
async def test_community_node_save(neo4j_driver, sample_community_node):
    neo4j_driver.execute_query.return_value = 'OK'

    result = await sample_community_node.save(neo4j_driver)

    neo4j_driver.execute_query.assert_awaited_once()
    assert result == 'OK'


@pytest.mark.asyncio
async def test_node_delete(neo4j_driver, sample_entity_node):
    neo4j_driver.execute_query.return_value = 'DELETED'

    result = await sample_entity_node.delete(neo4j_driver)

    assert result == 'DELETED'
    neo4j_driver.execute_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_node_delete_by_group_id(neo4j_driver):
    result = await EntityNode.delete_by_group_id(neo4j_driver, 'group1')

    assert result == 'SUCCESS'
    neo4j_driver.execute_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_name_embedding(sample_entity_node):
    mock_embedder = AsyncMock()
    mock_embedder.create = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    embedding = await sample_entity_node.generate_name_embedding(mock_embedder)

    assert embedding == [[0.1, 0.2, 0.3]]
    assert sample_entity_node.name_embedding == [[0.1, 0.2, 0.3]]


@pytest.mark.asyncio
async def test_entity_node_get_by_uuid_parses_correctly(neo4j_driver):
    now = datetime.now(timezone.utc)
    mock_record = {
        'uuid': str(uuid4()),
        'name': 'Entity Test',
        'name_embedding': [0.1, 0.2],
        'group_id': 'group1',
        'created_at': now,
        'summary': 'A summary',
        'labels': ['Entity'],
        'attributes': {'extra': 'info'},
        'episodes': ['ep1', 'ep2'],
    }
    neo4j_driver.execute_query.return_value = ([mock_record], None, None)

    node = await EntityNode.get_by_uuid(neo4j_driver, mock_record['uuid'])
    assert node.name == 'Entity Test'
    assert node.group_id == 'group1'
    assert 'extra' in node.attributes


@pytest.mark.asyncio
async def test_episodic_node_get_by_uuid_parses_correctly(neo4j_driver):
    now = datetime.now(timezone.utc)
    mock_record = {
        'uuid': str(uuid4()),
        'name': 'Episodic Test',
        'group_id': 'group1',
        'source': 'text',
        'source_description': 'desc',
        'content': 'Episode content',
        'created_at': now,
        'valid_at': now,
        'entity_edges': ['ent1', 'ent2'],
    }
    neo4j_driver.execute_query.return_value = ([mock_record], None, None)

    node = await EpisodicNode.get_by_uuid(neo4j_driver, mock_record['uuid'])
    assert node.name == 'Episodic Test'
    assert node.source == EpisodeType.text
    assert 'ent1' in node.entity_edges


@pytest.mark.asyncio
async def test_community_node_get_by_uuid_parses_correctly(neo4j_driver):
    now = datetime.now(timezone.utc)
    mock_record = {
        'uuid': str(uuid4()),
        'name': 'Community A',
        'group_id': 'group1',
        'name_embedding': [0.5],
        'created_at': now,
        'summary': 'Test summary',
    }
    neo4j_driver.execute_query.return_value = ([mock_record], None, None)

    node = await CommunityNode.get_by_uuid(neo4j_driver, mock_record['uuid'])
    assert node.name == 'Community A'
    assert node.group_id == 'group1'
