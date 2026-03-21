import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

graphiti_module = importlib.import_module('graphiti_core.graphiti')
Graphiti = graphiti_module.Graphiti
GraphProvider = importlib.import_module('graphiti_core.driver.driver').GraphProvider
EntityNode = importlib.import_module('graphiti_core.nodes').EntityNode
EpisodeType = importlib.import_module('graphiti_core.nodes').EpisodeType
NodeNotFoundError = importlib.import_module('graphiti_core.errors').NodeNotFoundError


class _Driver:
    provider = GraphProvider.NEO4J

    def __init__(self):
        self._database = 'neo4j'

    def clone(self, database):
        self._database = database
        return self


class _Span:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def add_attributes(self, attrs):
        return None

    def set_status(self, status, message):
        return None

    def record_exception(self, exc):
        return None


class _Tracer:
    def start_span(self, name):
        return _Span()


@pytest.mark.asyncio
async def test_add_episode_persists_searchable_nodes_before_attribute_extraction(monkeypatch):
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = _Driver()
    graphiti.llm_client = SimpleNamespace()
    graphiti.embedder = SimpleNamespace(create_batch=AsyncMock(return_value=[[0.1, 0.2, 0.3]]))
    graphiti.cross_encoder = SimpleNamespace()
    graphiti.clients = SimpleNamespace(
        driver=graphiti.driver,
        llm_client=graphiti.llm_client,
        embedder=graphiti.embedder,
        cross_encoder=graphiti.cross_encoder,
    )
    graphiti.tracer = _Tracer()
    graphiti.max_coroutines = None
    graphiti.store_raw_episode_content = True
    resolved_node = EntityNode(
        uuid='node-1',
        name='Codex Smoke Tester',
        labels=['Entity'],
        group_id='work-item-1',
        summary='',
    )
    persisted_calls = AsyncMock()

    monkeypatch.setattr(graphiti, 'retrieve_episodes', AsyncMock(return_value=[]))
    monkeypatch.setattr(graphiti_module, 'extract_nodes', AsyncMock(return_value=[resolved_node]))
    monkeypatch.setattr(
        graphiti_module,
        'resolve_extracted_nodes',
        AsyncMock(return_value=([resolved_node], {resolved_node.uuid: resolved_node.uuid}, [])),
    )
    monkeypatch.setattr(
        graphiti,
        '_extract_and_resolve_edges',
        AsyncMock(return_value=([], [], [])),
    )

    async def _fake_extract_attributes(*args, **kwargs):
        assert persisted_calls.await_count == 1
        assert resolved_node.name_embedding == [0.1, 0.2, 0.3]
        return [resolved_node.model_copy(update={'summary': 'hydrated'})]

    async def _fake_process_episode_data(episode, nodes, entity_edges, now, group_id, *args):
        return [], episode

    monkeypatch.setattr(graphiti_module, 'extract_attributes_from_nodes', _fake_extract_attributes)
    monkeypatch.setattr(graphiti, '_process_episode_data', _fake_process_episode_data)
    monkeypatch.setattr(graphiti_module, 'add_nodes_and_edges_bulk', persisted_calls)

    await graphiti.add_episode(
        name='Episode',
        episode_body='Codex Smoke Tester works at Graphiti Validation Lab.',
        source_description='test',
        reference_time=datetime.now(timezone.utc),
        source=EpisodeType.text,
        group_id='work-item-1',
    )

    early_persist_args = persisted_calls.await_args_list[0].args
    assert early_persist_args[1] == []
    assert early_persist_args[2] == []
    assert [node.uuid for node in early_persist_args[3]] == ['node-1']
    assert early_persist_args[4] == []


@pytest.mark.asyncio
async def test_add_episode_creates_new_episode_when_uuid_not_found(monkeypatch):
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = _Driver()
    graphiti.llm_client = SimpleNamespace()
    graphiti.embedder = SimpleNamespace(create_batch=AsyncMock(return_value=[]))
    graphiti.cross_encoder = SimpleNamespace()
    graphiti.clients = SimpleNamespace(
        driver=graphiti.driver,
        llm_client=graphiti.llm_client,
        embedder=graphiti.embedder,
        cross_encoder=graphiti.cross_encoder,
    )
    graphiti.tracer = _Tracer()
    graphiti.max_coroutines = None
    graphiti.store_raw_episode_content = True

    monkeypatch.setattr(graphiti, 'retrieve_episodes', AsyncMock(return_value=[]))
    monkeypatch.setattr(graphiti_module, 'extract_nodes', AsyncMock(return_value=[]))
    monkeypatch.setattr(
        graphiti_module,
        'resolve_extracted_nodes',
        AsyncMock(return_value=([], {}, [])),
    )
    monkeypatch.setattr(
        graphiti,
        '_extract_and_resolve_edges',
        AsyncMock(return_value=([], [], [])),
    )
    monkeypatch.setattr(
        graphiti_module, 'extract_attributes_from_nodes', AsyncMock(return_value=[])
    )
    monkeypatch.setattr(graphiti_module, 'add_nodes_and_edges_bulk', AsyncMock())

    captured = {}

    async def _fake_process_episode_data(episode, nodes, entity_edges, now, group_id, *args):
        captured['episode_uuid'] = episode.uuid
        return [], episode

    monkeypatch.setattr(graphiti, '_process_episode_data', _fake_process_episode_data)
    monkeypatch.setattr(
        graphiti_module.EpisodicNode,
        'get_by_uuid',
        AsyncMock(side_effect=NodeNotFoundError('episode-uuid-1')),
    )

    await graphiti.add_episode(
        name='Episode',
        episode_body='content',
        source_description='test',
        reference_time=datetime.now(timezone.utc),
        source=EpisodeType.text,
        group_id='work-item-1',
        uuid='episode-uuid-1',
    )

    assert captured['episode_uuid'] == 'episode-uuid-1'
