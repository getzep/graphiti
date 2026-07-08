import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_config import EdgeReranker, NodeReranker, SearchResults

import graphiti_mcp_server as srv
from config.schema import GraphitiConfig


class _FakeClient:
    def __init__(self):
        self.calls = []

    async def search_(self, *, query, config, group_ids, center_node_uuid, search_filter):
        self.calls.append(
            {'config': config, 'search_filter': search_filter, 'group_ids': group_ids}
        )
        edge = EntityEdge(
            uuid='e1',
            name='USES',
            fact='A uses B',
            group_id='g',
            source_node_uuid='n1',
            target_node_uuid='n2',
            created_at=datetime.now(timezone.utc),
        )
        node = EntityNode(
            uuid='n1',
            name='FanWeb',
            group_id='g',
            labels=['Entity'],
            created_at=datetime.now(timezone.utc),
            summary='s',
        )
        return SearchResults(
            edges=[edge],
            nodes=[node],
            edge_reranker_scores=[0.87],
            node_reranker_scores=[0.73],
        )


class _FakeService:
    def __init__(self, client):
        self._client = client

    async def get_client(self):
        return self._client


def _install_fake(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(srv, 'graphiti_service', _FakeService(client))
    # The module-level ``config`` global is only assigned inside initialize_server();
    # the tool functions read config.search / config.graphiti, so provide a default.
    monkeypatch.setattr(srv, 'config', GraphitiConfig(), raising=False)
    return client


def test_facts_use_mmr_and_surface_score(monkeypatch):
    client = _install_fake(monkeypatch)
    resp = asyncio.run(srv.search_memory_facts(query='q', group_ids='g'))
    assert resp['facts'][0]['score'] == 0.87
    call = client.calls[0]
    assert call['config'].edge_config.reranker == EdgeReranker.mmr
    # liveness filter applied by default
    assert call['search_filter'].expired_at is not None


def test_facts_include_invalidated_opt_in(monkeypatch):
    client = _install_fake(monkeypatch)
    asyncio.run(srv.search_memory_facts(query='q', group_ids='g', include_invalidated=True))
    call = client.calls[0]
    assert call['search_filter'].expired_at is None


def test_facts_default_limit_is_six(monkeypatch):
    client = _install_fake(monkeypatch)
    asyncio.run(srv.search_memory_facts(query='q', group_ids='g'))
    assert client.calls[0]['config'].limit == 6


def test_nodes_use_mmr_and_surface_score(monkeypatch):
    client = _install_fake(monkeypatch)
    resp = asyncio.run(srv.search_nodes(query='q', group_ids='g'))
    assert resp['nodes'][0]['score'] == 0.73
    assert client.calls[0]['config'].node_config.reranker == NodeReranker.mmr


def test_nodes_default_limit_is_six(monkeypatch):
    client = _install_fake(monkeypatch)
    asyncio.run(srv.search_nodes(query='q', group_ids='g'))
    assert client.calls[0]['config'].limit == 6
