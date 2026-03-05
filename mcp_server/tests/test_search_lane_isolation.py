from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import graphiti_mcp_server as srv
from config.schema import GraphitiConfig


def _base_config() -> GraphitiConfig:
    config = GraphitiConfig()
    config.database.provider = 'neo4j'
    config.graphiti.group_id = 's1_sessions_main'
    return config


@pytest.mark.asyncio
async def test_non_om_lanes_keep_existing_graphiti_search(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(srv, '_SEARCH_RATE_LIMIT_ENABLED', False)
    monkeypatch.setattr(srv, 'config', _base_config())

    graphiti_node = SimpleNamespace(
        uuid='entity-1',
        name='Neo4j',
        labels=['Entity'],
        created_at=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
        summary='Neo4j is configured for this environment.',
        group_id='s1_sessions_main',
        attributes={'source': 'graphiti'},
    )

    fake_client = SimpleNamespace(
        search_=AsyncMock(return_value=SimpleNamespace(nodes=[graphiti_node], edges=[])),
        driver=SimpleNamespace(execute_query=AsyncMock()),
    )
    fake_service = SimpleNamespace(
        config=SimpleNamespace(database=SimpleNamespace(provider='neo4j')),
        get_client_for_group=AsyncMock(return_value=fake_client),
        get_client=AsyncMock(return_value=fake_client),
    )
    monkeypatch.setattr(srv, 'graphiti_service', fake_service)

    response = await srv.search_nodes(
        query='Neo4j',
        group_ids=['s1_sessions_main'],
        max_nodes=5,
    )

    assert response['message'] == 'Nodes retrieved successfully'
    assert response['nodes'][0]['uuid'] == 'entity-1'
    fake_client.search_.assert_awaited_once()
    fake_client.driver.execute_query.assert_not_called()


@pytest.mark.asyncio
async def test_search_nodes_om_only_errors_fail_closed_without_graphiti_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(srv, '_SEARCH_RATE_LIMIT_ENABLED', False)
    monkeypatch.setattr(srv, 'config', _base_config())

    fake_client = SimpleNamespace(
        driver=SimpleNamespace(execute_query=AsyncMock(side_effect=RuntimeError('neo4j unavailable'))),
        search_=AsyncMock(),
    )
    fake_service = SimpleNamespace(
        config=SimpleNamespace(database=SimpleNamespace(provider='neo4j')),
        get_client=AsyncMock(return_value=fake_client),
        get_client_for_group=AsyncMock(return_value=fake_client),
    )
    monkeypatch.setattr(srv, 'graphiti_service', fake_service)

    response = await srv.search_nodes(
        query='latency',
        group_ids=['s1_observational_memory'],
        max_nodes=5,
    )

    assert response == {'message': 'No relevant nodes found', 'nodes': []}
    fake_client.search_.assert_not_called()
    fake_service.get_client_for_group.assert_not_called()


@pytest.mark.asyncio
async def test_search_memory_facts_om_only_errors_fail_closed_without_graphiti_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(srv, '_SEARCH_RATE_LIMIT_ENABLED', False)
    monkeypatch.setattr(srv, 'config', _base_config())

    fake_client = SimpleNamespace(
        driver=SimpleNamespace(execute_query=AsyncMock(side_effect=RuntimeError('neo4j unavailable'))),
        search_=AsyncMock(),
    )
    fake_service = SimpleNamespace(
        config=SimpleNamespace(database=SimpleNamespace(provider='neo4j')),
        get_client=AsyncMock(return_value=fake_client),
        get_client_for_group=AsyncMock(return_value=fake_client),
    )
    monkeypatch.setattr(srv, 'graphiti_service', fake_service)

    response = await srv.search_memory_facts(
        query='latency',
        group_ids=['s1_observational_memory'],
        max_facts=5,
    )

    assert response == {'message': 'No relevant facts found', 'facts': []}
    fake_client.search_.assert_not_called()
    fake_service.get_client_for_group.assert_not_called()


@pytest.mark.asyncio
async def test_search_nodes_mixed_lane_merges_om_and_graphiti_results(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(srv, '_SEARCH_RATE_LIMIT_ENABLED', False)
    monkeypatch.setattr(srv, 'config', _base_config())

    graphiti_node = SimpleNamespace(
        uuid='entity-1',
        name='Neo4j',
        labels=['Entity'],
        created_at=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
        summary='Neo4j is configured for this environment.',
        group_id='s1_sessions_main',
        attributes={'source': 'graphiti'},
    )

    om_node_rows = [
        {
            'uuid': 'om-node-1',
            'content': 'Latency pattern from observational memory.',
            'created_at': '2026-03-01T12:05:00Z',
            'group_id': 's1_observational_memory',
            'status': 'open',
            'semantic_domain': 'sessions_main',
            'urgency_score': 3,
            'lexical_score': 2,
        }
    ]

    fake_client = SimpleNamespace(
        driver=SimpleNamespace(execute_query=AsyncMock(return_value=(om_node_rows, None, None))),
        search_=AsyncMock(return_value=SimpleNamespace(nodes=[graphiti_node], edges=[])),
    )
    fake_service = SimpleNamespace(
        config=SimpleNamespace(database=SimpleNamespace(provider='neo4j')),
        get_client=AsyncMock(return_value=fake_client),
        get_client_for_group=AsyncMock(return_value=fake_client),
    )
    monkeypatch.setattr(srv, 'graphiti_service', fake_service)

    response = await srv.search_nodes(
        query='latency',
        group_ids=['s1_observational_memory', 's1_sessions_main'],
        max_nodes=5,
    )

    assert response['message'] == 'Nodes retrieved successfully'
    assert response['nodes'][0]['uuid'] == 'entity-1'
    assert response['nodes'][1]['uuid'] == 'om-node-1'
    fake_client.search_.assert_awaited_once()
    fake_client.driver.execute_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_nodes_mixed_lane_om_errors_fall_back_to_graphiti(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(srv, '_SEARCH_RATE_LIMIT_ENABLED', False)
    monkeypatch.setattr(srv, 'config', _base_config())

    graphiti_node = SimpleNamespace(
        uuid='entity-1',
        name='Neo4j',
        labels=['Entity'],
        created_at=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
        summary='Neo4j is configured for this environment.',
        group_id='s1_sessions_main',
        attributes={'source': 'graphiti'},
    )

    fake_client = SimpleNamespace(
        driver=SimpleNamespace(execute_query=AsyncMock(side_effect=RuntimeError('neo4j unavailable'))),
        search_=AsyncMock(return_value=SimpleNamespace(nodes=[graphiti_node], edges=[])),
    )
    fake_service = SimpleNamespace(
        config=SimpleNamespace(database=SimpleNamespace(provider='neo4j')),
        get_client=AsyncMock(return_value=fake_client),
        get_client_for_group=AsyncMock(return_value=fake_client),
    )
    monkeypatch.setattr(srv, 'graphiti_service', fake_service)

    response = await srv.search_nodes(
        query='latency',
        group_ids=['s1_observational_memory', 's1_sessions_main'],
        max_nodes=5,
    )

    assert response['message'] == 'Nodes retrieved successfully'
    assert response['nodes'][0]['uuid'] == 'entity-1'
    fake_client.search_.assert_awaited_once()
    fake_service.get_client_for_group.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_memory_facts_mixed_lane_om_errors_fall_back_to_graphiti(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(srv, '_SEARCH_RATE_LIMIT_ENABLED', False)
    monkeypatch.setattr(srv, 'config', _base_config())

    class _FakeEdge:
        def model_dump(self, *, mode: str = 'json', exclude: set[str] | None = None):
            return {
                'uuid': 'edge-1',
                'name': 'RELATES_TO',
                'fact': 'RELATES_TO: source -> target',
                'group_id': 's1_sessions_main',
                'source_node_uuid': 'node-source',
                'target_node_uuid': 'node-target',
                'created_at': '2026-03-01T16:00:00Z',
                'valid_at': None,
                'invalid_at': None,
                'expired_at': None,
                'episodes': [],
                'attributes': {'source': 'graphiti'},
            }

    fake_client = SimpleNamespace(
        driver=SimpleNamespace(execute_query=AsyncMock(side_effect=RuntimeError('neo4j unavailable'))),
        search_=AsyncMock(return_value=SimpleNamespace(nodes=[], edges=[_FakeEdge()])),
    )
    fake_service = SimpleNamespace(
        config=SimpleNamespace(database=SimpleNamespace(provider='neo4j')),
        get_client=AsyncMock(return_value=fake_client),
        get_client_for_group=AsyncMock(return_value=fake_client),
    )
    monkeypatch.setattr(srv, 'graphiti_service', fake_service)

    response = await srv.search_memory_facts(
        query='latency',
        group_ids=['s1_observational_memory', 's1_sessions_main'],
        max_facts=5,
    )

    assert response['message'] == 'Facts retrieved successfully'
    assert response['facts'][0]['uuid'] == 'edge-1'
    fake_client.search_.assert_awaited_once()
    fake_service.get_client_for_group.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_memory_facts_mixed_lane_merges_om_and_graphiti_results(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(srv, '_SEARCH_RATE_LIMIT_ENABLED', False)
    monkeypatch.setattr(srv, 'config', _base_config())

    om_fact_rows = [
        {
            'uuid': 'om-rel-1',
            'relation_type': 'RESOLVES',
            'source_node_id': 'om-node-1',
            'target_node_id': 'om-node-2',
            'created_at': '2026-03-01T13:00:00Z',
            'group_id': 's1_observational_memory',
            'source_content': 'Investigate latency spike.',
            'target_content': 'Latency incident closed.',
            'lexical_score': 2,
        }
    ]

    class _FakeEdge:
        def model_dump(self, *, mode: str = 'json', exclude: set[str] | None = None):
            return {
                'uuid': 'edge-1',
                'name': 'RELATES_TO',
                'fact': 'RELATES_TO: source -> target',
                'group_id': 's1_sessions_main',
                'source_node_uuid': 'node-source',
                'target_node_uuid': 'node-target',
                'created_at': '2026-03-01T16:00:00Z',
                'valid_at': None,
                'invalid_at': None,
                'expired_at': None,
                'episodes': [],
                'attributes': {'source': 'graphiti'},
            }

    fake_client = SimpleNamespace(
        driver=SimpleNamespace(execute_query=AsyncMock(return_value=(om_fact_rows, None, None))),
        search_=AsyncMock(return_value=SimpleNamespace(nodes=[], edges=[_FakeEdge()])),
    )
    fake_service = SimpleNamespace(
        config=SimpleNamespace(database=SimpleNamespace(provider='neo4j')),
        get_client=AsyncMock(return_value=fake_client),
        get_client_for_group=AsyncMock(return_value=fake_client),
    )
    monkeypatch.setattr(srv, 'graphiti_service', fake_service)

    response = await srv.search_memory_facts(
        query='latency incident',
        group_ids=['s1_observational_memory', 's1_sessions_main'],
        max_facts=5,
    )

    assert response['message'] == 'Facts retrieved successfully'
    assert response['facts'][0]['uuid'] == 'edge-1'
    assert response['facts'][1]['uuid'] == 'om-rel-1'
    fake_client.search_.assert_awaited_once()
    fake_client.driver.execute_query.assert_awaited_once()
