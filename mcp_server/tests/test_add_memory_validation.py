"""Regression tests for entity configuration validation before episode queueing."""

import logging
from unittest.mock import AsyncMock, Mock

import pytest
from graphiti_core import Graphiti
from graphiti_core.errors import EntityTypeValidationError
from graphiti_core.nodes import EpisodeType
from pydantic import BaseModel

import graphiti_mcp_server as server
from config.schema import DatabaseConfig, EntityTypeConfig, GraphitiConfig
from models.entity_types import ENTITY_TYPES
from models.response_types import ErrorResponse, SuccessResponse
from services.queue_service import QueueService


class DataObject(BaseModel):
    attributes: str


@pytest.fixture
def client():
    return AsyncMock(spec=Graphiti)


@pytest.fixture
async def queue_service(client):
    service = QueueService()
    await service.initialize(client)
    return service


@pytest.fixture
def memory_service(monkeypatch, queue_service):
    config = GraphitiConfig()
    config.graphiti.group_id = 'default'
    service = server.GraphitiService(config)
    monkeypatch.setattr(server, 'config', config, raising=False)
    monkeypatch.setattr(server, 'graphiti_service', service)
    monkeypatch.setattr(server, 'queue_service', queue_service)
    return service


async def test_reserved_entity_field_is_rejected_before_queueing(
    monkeypatch, queue_service, client
):
    enqueue = AsyncMock()
    monkeypatch.setattr(queue_service, 'add_episode_task', enqueue)

    with pytest.raises(EntityTypeValidationError, match='attributes.*DataObject.*protected'):
        await queue_service.add_episode(
            group_id='default',
            name='Invalid configuration',
            content='An ordinary episode body.',
            source_description='',
            episode_type=EpisodeType.text,
            entity_types={'DataObject': DataObject},
            uuid=None,
        )

    enqueue.assert_not_called()
    assert queue_service._episode_queues == {}
    client.add_episode.assert_not_called()


@pytest.mark.parametrize('entity_types', [None, {'Preference': ENTITY_TYPES['Preference']}])
async def test_invalid_excluded_types_are_rejected_before_queueing(
    monkeypatch, queue_service, client, entity_types
):
    enqueue = AsyncMock()
    monkeypatch.setattr(queue_service, 'add_episode_task', enqueue)

    with pytest.raises(ValueError, match='Invalid excluded entity types:.*Missing'):
        await queue_service.add_episode(
            group_id='default',
            name='Invalid exclusions',
            content='An ordinary episode body.',
            source_description='',
            episode_type=EpisodeType.text,
            entity_types=entity_types,
            excluded_entity_types=['Missing'],
            uuid=None,
        )

    enqueue.assert_not_called()
    assert queue_service._episode_queues == {}
    client.add_episode.assert_not_called()


async def test_add_memory_returns_reserved_field_error(
    monkeypatch, memory_service, queue_service, client
):
    memory_service.entity_types = {'DataObject': DataObject}
    enqueue = AsyncMock()
    monkeypatch.setattr(queue_service, 'add_episode_task', enqueue)

    response = await server.add_memory(
        name='Invalid configuration', episode_body='An ordinary episode body.'
    )

    expected_error = EntityTypeValidationError('DataObject', 'attributes')
    assert response == ErrorResponse(error=f'Error queuing episode: {expected_error}')
    assert 'message' not in response
    enqueue.assert_not_called()
    assert queue_service._episode_queues == {}
    client.add_episode.assert_not_called()


@pytest.mark.parametrize('entity_types', [None, {'Preference': ENTITY_TYPES['Preference']}])
async def test_add_memory_accepts_attributes_in_body(
    memory_service, queue_service, client, entity_types
):
    memory_service.entity_types = entity_types
    body = 'The object has attributes including color and size.'

    response = await server.add_memory(name='Object details', episode_body=body)

    assert response == SuccessResponse(
        message="Episode 'Object details' queued for processing in group 'default'"
    )
    await queue_service._episode_queues['default'].join()
    client.add_episode.assert_awaited_once()
    assert client.add_episode.await_args.kwargs['episode_body'] == body


@pytest.mark.parametrize('uuid', [None, 'episode-uuid'])
@pytest.mark.parametrize('fails', [False, True])
async def test_worker_logs_episode_name_and_optional_uuid(
    queue_service, client, caplog, uuid, fails
):
    caplog.set_level(logging.INFO, logger='services.queue_service')
    if fails:
        client.add_episode.side_effect = RuntimeError('Processing failed')

    await queue_service.add_episode(
        group_id='default',
        name='Object details',
        content='An ordinary episode body.',
        source_description='',
        episode_type=EpisodeType.text,
        entity_types=None,
        uuid=uuid,
    )
    await queue_service._episode_queues['default'].join()

    episode_label = "'Object details'" + (f' (uuid: {uuid})' if uuid else '')
    assert f'Processing episode {episode_label} for group default' in caplog.text
    if fails:
        assert (
            f'Failed to process episode {episode_label} for group default: Processing failed'
            in caplog.text
        )
    else:
        assert f'Successfully processed episode {episode_label} for group default' in caplog.text
    assert 'episode None' not in caplog.text


async def test_initialize_logs_invalid_entity_types_without_aborting(monkeypatch, client, caplog):
    from graphiti_core.driver import neo4j_driver

    config = GraphitiConfig(database=DatabaseConfig(provider='neo4j'))
    config.graphiti.entity_types = [
        EntityTypeConfig(name='DataObject', description='An object with custom fields')
    ]
    monkeypatch.setitem(ENTITY_TYPES, 'DataObject', DataObject)
    for factory in (server.LLMClientFactory, server.EmbedderFactory, server.CrossEncoderFactory):
        monkeypatch.setattr(factory, 'create', Mock())
    monkeypatch.setattr(
        server.DatabaseDriverFactory,
        'create_config',
        Mock(
            return_value={
                'uri': 'bolt://unused',
                'user': 'test',
                'password': '',
                'database': 'test',
            }
        ),
    )
    monkeypatch.setattr(neo4j_driver, 'Neo4jDriver', Mock())
    monkeypatch.setattr(server, 'Graphiti', Mock(return_value=client))
    service = server.GraphitiService(config)

    await service.initialize()

    assert service.client is client
    assert service.entity_types == {'DataObject': DataObject}
    client.build_indices_and_constraints.assert_awaited_once()
    assert 'Invalid entity type configuration' in caplog.text
    assert str(EntityTypeValidationError('DataObject', 'attributes')) in caplog.text
