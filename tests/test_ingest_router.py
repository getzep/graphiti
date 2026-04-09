"""
Patches are applied before TestClient starts its event loop so the lifespan
also sees the fake settings.
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from server.graph_service.main import app
from server.graph_service.config import Settings


@pytest.fixture
def mock_settings():
    return Settings(
        openai_api_key='test-key',
        openai_base_url='http://localhost:11434/v1',
        model_name='gemma4:26b',
        embedding_model_name='nomic-embed-text',
        neo4j_uri='bolt://localhost:7687',
        neo4j_user='neo4j',
        neo4j_password='test',
    )


@pytest.fixture
def mock_zep_client():
    mock = AsyncMock()
    mock.add_episode = AsyncMock(return_value=None)
    mock.close = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def client(mock_settings, mock_zep_client):
    with patch('server.graph_service.main.get_settings', return_value=mock_settings):
        with patch('server.graph_service.main.initialize_graphiti', new_callable=AsyncMock):
            with patch('server.graph_service.routers.ingest._build_graphiti_client', return_value=mock_zep_client):
                with patch('server.graph_service.routers.ingest.asyncio.create_task', side_effect=lambda t, **kw: t):
                    with TestClient(app) as c:
                        yield c, mock_zep_client


class TestAddMessagesTimestampDefault:
    def test_add_messages_defaults_timestamp_to_real_datetime(self, client):
        c, mock = client
        response = c.post(
            '/messages',
            json={
                'group_id': 'test-group',
                'messages': [
                    {'content': 'hello world', 'role': 'henry', 'role_type': 'user'},
                ],
            },
        )
        assert response.status_code == 202
        mock.add_episode.assert_called_once()
        call_kwargs = mock.add_episode.call_args.kwargs
        assert call_kwargs['reference_time'] is not None
        assert isinstance(call_kwargs['reference_time'], datetime)

    def test_add_messages_preserves_explicit_timestamp(self, client):
        c, mock = client
        explicit_ts = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        response = c.post(
            '/messages',
            json={
                'group_id': 'test-group',
                'messages': [
                    {
                        'content': 'hello world',
                        'role': 'henry',
                        'role_type': 'user',
                        'timestamp': explicit_ts.isoformat(),
                    },
                ],
            },
        )
        assert response.status_code == 202
        mock.add_episode.assert_called_once()
        assert mock.add_episode.call_args.kwargs['reference_time'] == explicit_ts


class TestAsyncWorkerClientLifetime:
    def test_worker_builds_fresh_client_per_job(self, client):
        c, mock = client
        with patch('server.graph_service.routers.ingest._build_graphiti_client', return_value=mock) as mock_build:
            response = c.post(
                '/messages',
                json={
                    'group_id': 'test-group',
                    'messages': [
                        {'content': 'hello', 'role': 'henry', 'role_type': 'user'},
                        {'content': 'world', 'role': 'henry', 'role_type': 'user'},
                    ],
                },
            )
        assert response.status_code == 202
        assert mock_build.call_count == 2
        assert mock.close.call_count == 2

    def test_worker_exception_caught_not_propagated(self, client):
        c, mock = client
        mock.add_episode.side_effect = RuntimeError('boom')
        with patch('server.graph_service.routers.ingest._build_graphiti_client', return_value=mock):
            response = c.post(
                '/messages',
                json={
                    'group_id': 'test-group',
                    'messages': [
                        {'content': 'hello', 'role': 'henry', 'role_type': 'user'},
                    ],
                },
            )
        assert response.status_code == 202
        mock.add_episode.assert_called_once()
