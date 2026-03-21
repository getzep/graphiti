import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

server = importlib.import_module('graphiti_mcp_server')
GraphitiConfig = importlib.import_module('config.schema').GraphitiConfig
EntityNode = importlib.import_module('graphiti_core.nodes').EntityNode
GraphProvider = importlib.import_module('graphiti_core.driver.driver').GraphProvider


class _QueueServiceStub:
    def __init__(self, queue_position=1, status=None):
        self.queue_position = queue_position
        self.status = status
        self.calls = []

    async def add_episode(self, **kwargs):
        self.calls.append(kwargs)
        return self.queue_position

    def get_episode_status(self, episode_uuid):
        if self.status and self.status.episode_uuid == episode_uuid:
            return self.status
        return None

    def get_queue_position(self, episode_uuid):
        if self.status and self.status.episode_uuid == episode_uuid:
            return self.status.queue_position
        return None

    def get_queue_size(self, group_id):
        return 0


class _AsyncResult:
    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        async def _iterator():
            for item in self._items:
                yield item

        return _iterator()


@pytest.mark.asyncio
async def test_add_memory_returns_episode_tracking_metadata(monkeypatch):
    config = GraphitiConfig()
    queue_service = _QueueServiceStub(queue_position=3)

    monkeypatch.setattr(server, 'config', config, raising=False)
    monkeypatch.setattr(server, 'graphiti_service', SimpleNamespace(entity_types=None), raising=False)
    monkeypatch.setattr(server, 'queue_service', queue_service, raising=False)

    response = await server.add_memory(
        name='Episode',
        episode_body='content',
        group_id='group-1',
        source='text',
        source_description='test',
    )

    assert response['group_id'] == 'group-1'
    assert response['queue_position'] == 3
    UUID(response['episode_uuid'])
    assert queue_service.calls[0]['uuid'] == response['episode_uuid']


@pytest.mark.asyncio
async def test_get_ingest_status_returns_current_episode_state(monkeypatch):
    status = SimpleNamespace(
        episode_uuid='episode-1',
        group_id='group-1',
        state='processing',
        queue_position=None,
        queue_depth=0,
        queued_at='2026-03-20T00:00:00+00:00',
        started_at='2026-03-20T00:00:01+00:00',
        processed_at=None,
        last_error=None,
    )
    queue_service = _QueueServiceStub(status=status)

    monkeypatch.setattr(server, 'queue_service', queue_service, raising=False)

    response = await server.get_ingest_status('episode-1', group_id='group-1')

    assert response['state'] == 'processing'
    assert response['episode_uuid'] == 'episode-1'
    assert response['group_id'] == 'group-1'
    assert response['started_at'] == '2026-03-20T00:00:01+00:00'


@pytest.mark.asyncio
async def test_search_nodes_uses_name_fallback_when_hybrid_search_is_empty(monkeypatch):
    config = GraphitiConfig()
    fallback_node = EntityNode(
        uuid='node-1',
        name='Codex Smoke Tester',
        labels=['Entity'],
        group_id='group-1',
        summary='summary',
    )

    class _Client:
        driver = SimpleNamespace()

        async def search_(self, **kwargs):
            return SimpleNamespace(nodes=[])

    class _GraphitiService:
        async def get_client(self):
            return _Client()

    fallback_calls = []

    async def _fallback(**kwargs):
        fallback_calls.append(kwargs)
        return [fallback_node]

    monkeypatch.setattr(server, 'config', config, raising=False)
    monkeypatch.setattr(server, 'graphiti_service', _GraphitiService(), raising=False)
    monkeypatch.setattr(server, 'search_nodes_by_name_fallback', _fallback, raising=False)

    response = await server.search_nodes('Codex Smoke Tester', group_ids=['group-1'])

    assert response['nodes'][0]['name'] == 'Codex Smoke Tester'
    assert fallback_calls


@pytest.mark.asyncio
async def test_get_status_reports_index_drift_in_details(monkeypatch):
    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def run(self, query):
            return _AsyncResult([{'count': 0}])

    class _Driver:
        provider = GraphProvider.NEO4J

        def session(self):
            return _Session()

        async def execute_query(self, query, **kwargs):
            return (
                [
                    {
                        'name': 'node_name_and_summary',
                        'properties': ['name', 'summary'],
                        'state': 'ONLINE',
                    },
                    {
                        'name': 'edge_name_and_fact',
                        'properties': ['name', 'fact', 'group_id'],
                        'state': 'ONLINE',
                    },
                ],
                None,
                None,
            )

    class _Client:
        driver = _Driver()

    class _GraphitiService:
        config = SimpleNamespace(database=SimpleNamespace(provider='neo4j'))

        async def get_client(self):
            return _Client()

    monkeypatch.setattr(server, 'graphiti_service', _GraphitiService(), raising=False)

    response = await server.get_status()

    assert response['status'] == 'ok'
    assert response['details']['index_check']['status'] == 'drift'
    assert any(
        item['name'] == 'node_name_and_summary'
        for item in response['details']['index_check']['stale']
    )
