from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import graphiti_mcp_server as srv
from config.schema import GraphitiConfig


def _base_config(provider: str = 'neo4j') -> GraphitiConfig:
    config = GraphitiConfig()
    config.database.provider = provider
    config.graphiti.group_id = 's1_sessions_main'
    return config


@pytest.mark.parametrize(
    'value',
    [
        'nan',
        'NaN',
        'inf',
        '+inf',
        '-inf',
        'Infinity',
        '-Infinity',
    ],
)
def test_env_float_rejects_non_finite_values_returns_default(monkeypatch, value: str):
    """Ensure fusion float config parsing falls back on non-finite env values."""

    monkeypatch.setenv('SEARCH_FUSION_OVERLAP_BOOST', value)

    assert (
        srv._env_float(
            'SEARCH_FUSION_OVERLAP_BOOST',
            1.75,
            min_value=0.0,
            max_value=10.0,
        )
        == 1.75
    )


@pytest.mark.asyncio
async def test_search_nodes_returns_om_primitive_results(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(srv, '_SEARCH_RATE_LIMIT_ENABLED', False)
    monkeypatch.setattr(srv, 'config', _base_config())

    om_node_rows = [
        {
            'uuid': 'om-node-1',
            'content': 'Neo4j heap cap should stay below 70 percent.',
            'created_at': '2026-03-01T12:00:00Z',
            'group_id': 's1_observational_memory',
            'status': 'open',
            'semantic_domain': 'sessions_main',
            'urgency_score': 4,
            'lexical_score': 3,
        }
    ]

    fake_client = SimpleNamespace(
        driver=SimpleNamespace(execute_query=AsyncMock(return_value=(om_node_rows, None, None))),
        search_=AsyncMock(),
    )
    fake_service = SimpleNamespace(
        config=SimpleNamespace(database=SimpleNamespace(provider='neo4j')),
        get_client=AsyncMock(return_value=fake_client),
        get_client_for_group=AsyncMock(return_value=fake_client),
    )

    monkeypatch.setattr(srv, 'graphiti_service', fake_service)

    response = await srv.search_nodes(
        query='neo4j heap cap',
        group_ids=['s1_observational_memory'],
        max_nodes=5,
    )

    assert response['message'] == 'Nodes retrieved successfully'
    assert len(response['nodes']) == 1
    assert response['nodes'][0]['uuid'] == 'om-node-1'
    assert response['nodes'][0]['labels'] == ['OMNode']
    assert response['nodes'][0]['group_id'] == 's1_observational_memory'
    assert response['nodes'][0]['attributes']['source'] == 'om_primitive'
    fake_client.search_.assert_not_called()
    fake_client.driver.execute_query.assert_awaited()


@pytest.mark.asyncio
async def test_search_memory_facts_returns_om_relation_results(monkeypatch: pytest.MonkeyPatch):
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

    fake_client = SimpleNamespace(
        driver=SimpleNamespace(execute_query=AsyncMock(return_value=(om_fact_rows, None, None))),
        search_=AsyncMock(),
    )
    fake_service = SimpleNamespace(
        config=SimpleNamespace(database=SimpleNamespace(provider='neo4j')),
        get_client=AsyncMock(return_value=fake_client),
        get_client_for_group=AsyncMock(return_value=fake_client),
    )

    monkeypatch.setattr(srv, 'graphiti_service', fake_service)

    response = await srv.search_memory_facts(
        query='latency incident',
        group_ids=['s1_observational_memory'],
        max_facts=5,
    )

    assert response['message'] == 'Facts retrieved successfully'
    assert len(response['facts']) == 1
    assert response['facts'][0]['uuid'] == 'om-rel-1'
    assert response['facts'][0]['name'] == 'RESOLVES'
    assert response['facts'][0]['group_id'] == 's1_observational_memory'
    assert response['facts'][0]['attributes']['source'] == 'om_primitive'
    fake_client.search_.assert_not_called()
    fake_client.driver.execute_query.assert_awaited()


@pytest.mark.asyncio
async def test_search_nodes_all_lanes_scope_merges_om_and_graphiti_results(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(srv, '_SEARCH_RATE_LIMIT_ENABLED', False)
    monkeypatch.setattr(srv, 'config', _base_config())

    graphiti_node = SimpleNamespace(
        uuid='entity-1',
        name='Latency Incident',
        labels=['Entity'],
        created_at=datetime(2026, 3, 1, 14, 0, tzinfo=timezone.utc),
        summary='Graphiti search result from Neo4j-backed corpus.',
        group_id='s1_sessions_main',
        attributes={'source': 'graphiti'},
    )

    om_node_rows = [
        {
            'uuid': 'om-node-2',
            'content': 'All-lanes scope should still include OM retrieval.',
            'created_at': '2026-03-01T14:00:00Z',
            'group_id': 's1_observational_memory',
            'status': 'open',
            'semantic_domain': 'sessions_main',
            'urgency_score': 3,
            'lexical_score': 1,
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
        query='all lanes om',
        group_ids=[],
        max_nodes=5,
    )

    assert response['message'] == 'Nodes retrieved successfully'
    assert response['nodes'][0]['uuid'] == 'entity-1'
    assert response['nodes'][1]['uuid'] == 'om-node-2'
    fake_client.search_.assert_awaited_once()
    fake_client.driver.execute_query.assert_awaited()


@pytest.mark.asyncio
async def test_search_nodes_all_lanes_with_om_error_falls_back_to_graphiti(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(srv, '_SEARCH_RATE_LIMIT_ENABLED', False)
    monkeypatch.setattr(srv, 'config', _base_config())

    graphiti_node = SimpleNamespace(
        uuid='entity-1',
        name='Latency Incident',
        labels=['Entity'],
        created_at=datetime(2026, 3, 1, 14, 30, tzinfo=timezone.utc),
        summary='Graphiti search result from Neo4j-backed corpus.',
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
        query='all lanes om',
        group_ids=[],
        max_nodes=5,
    )

    assert response['message'] == 'Nodes retrieved successfully'
    assert response['nodes'][0]['uuid'] == 'entity-1'
    fake_client.search_.assert_awaited_once()
    fake_client.driver.execute_query.assert_awaited()


@pytest.mark.asyncio
async def test_search_memory_facts_all_lanes_scope_merges_om_and_graphiti_results(
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
        group_ids=[],
        max_facts=5,
    )

    assert response['message'] == 'Facts retrieved successfully'
    assert response['facts'][0]['uuid'] == 'edge-1'
    assert response['facts'][1]['uuid'] == 'om-rel-1'
    fake_client.search_.assert_awaited_once()
    fake_client.driver.execute_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_memory_facts_all_lanes_with_om_error_falls_back_to_graphiti(
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
        query='latency incident',
        group_ids=[],
        max_facts=5,
    )

    assert response['message'] == 'Facts retrieved successfully'
    assert response['facts'][0]['uuid'] == 'edge-1'
    fake_client.search_.assert_awaited_once()
    fake_client.driver.execute_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_nodes_falkordb_all_lanes_uses_graphiti_path(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(srv, '_SEARCH_RATE_LIMIT_ENABLED', False)
    monkeypatch.setattr(srv, 'config', _base_config(provider='falkordb'))

    graphiti_node = SimpleNamespace(
        uuid='entity-1',
        name='Latency Incident',
        labels=['Entity'],
        created_at=datetime(2026, 3, 1, 15, 0, tzinfo=timezone.utc),
        summary='Graphiti search result from FalkorDB-backed corpus.',
        group_id='s1_sessions_main',
        attributes={'source': 'graphiti'},
    )

    fake_client = SimpleNamespace(
        search_=AsyncMock(return_value=SimpleNamespace(nodes=[graphiti_node], edges=[])),
        driver=SimpleNamespace(execute_query=AsyncMock()),
    )
    fake_service = SimpleNamespace(
        config=SimpleNamespace(database=SimpleNamespace(provider='falkordb')),
        get_client=AsyncMock(return_value=fake_client),
        get_client_for_group=AsyncMock(return_value=fake_client),
    )
    monkeypatch.setattr(srv, 'graphiti_service', fake_service)

    response = await srv.search_nodes(
        query='latency incident',
        group_ids=[],
        max_nodes=5,
    )

    assert response['message'] == 'Nodes retrieved successfully'
    assert response['nodes'][0]['uuid'] == 'entity-1'
    fake_client.search_.assert_awaited_once()
    fake_client.driver.execute_query.assert_not_called()
    fake_service.get_client.assert_not_called()


@pytest.mark.asyncio
async def test_search_memory_facts_falkordb_all_lanes_uses_graphiti_path(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(srv, '_SEARCH_RATE_LIMIT_ENABLED', False)
    monkeypatch.setattr(srv, 'config', _base_config(provider='falkordb'))

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
        search_=AsyncMock(return_value=SimpleNamespace(nodes=[], edges=[_FakeEdge()])),
        driver=SimpleNamespace(execute_query=AsyncMock()),
    )
    fake_service = SimpleNamespace(
        config=SimpleNamespace(database=SimpleNamespace(provider='falkordb')),
        get_client=AsyncMock(return_value=fake_client),
        get_client_for_group=AsyncMock(return_value=fake_client),
    )
    monkeypatch.setattr(srv, 'graphiti_service', fake_service)

    response = await srv.search_memory_facts(
        query='incident relationship',
        group_ids=[],
        max_facts=5,
    )

    assert response['message'] == 'Facts retrieved successfully'
    assert response['facts'][0]['uuid'] == 'edge-1'
    fake_client.search_.assert_awaited_once()
    fake_client.driver.execute_query.assert_not_called()
    fake_service.get_client.assert_not_called()


@pytest.mark.asyncio
async def test_search_nodes_om_adapter_honors_entity_types_filter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(srv, '_SEARCH_RATE_LIMIT_ENABLED', False)
    monkeypatch.setattr(srv, 'config', _base_config())

    fake_client = SimpleNamespace(
        driver=SimpleNamespace(execute_query=AsyncMock(return_value=([], None, None))),
        search_=AsyncMock(),
    )
    fake_service = SimpleNamespace(
        config=SimpleNamespace(database=SimpleNamespace(provider='neo4j')),
        get_client=AsyncMock(return_value=fake_client),
        get_client_for_group=AsyncMock(return_value=fake_client),
    )

    monkeypatch.setattr(srv, 'graphiti_service', fake_service)

    response = await srv.search_nodes(
        query='neo4j heap cap',
        group_ids=['s1_observational_memory'],
        entity_types=['Person'],
        max_nodes=5,
    )

    assert response == {'message': 'No relevant nodes found', 'nodes': []}
    fake_service.get_client.assert_not_called()
    fake_client.driver.execute_query.assert_not_called()
    fake_client.search_.assert_not_called()


@pytest.mark.asyncio
async def test_search_memory_facts_om_adapter_passes_center_node_uuid(
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

    fake_client = SimpleNamespace(
        driver=SimpleNamespace(execute_query=AsyncMock(return_value=(om_fact_rows, None, None))),
        search_=AsyncMock(),
    )
    fake_service = SimpleNamespace(
        config=SimpleNamespace(database=SimpleNamespace(provider='neo4j')),
        get_client=AsyncMock(return_value=fake_client),
        get_client_for_group=AsyncMock(return_value=fake_client),
    )
    monkeypatch.setattr(srv, 'graphiti_service', fake_service)

    response = await srv.search_memory_facts(
        query='latency incident',
        group_ids=['s1_observational_memory'],
        max_facts=5,
        center_node_uuid='om-node-1',
    )

    assert response['message'] == 'Facts retrieved successfully'
    assert response['facts'][0]['uuid'] == 'om-rel-1'
    assert fake_client.driver.execute_query.await_args.kwargs['center_node_uuid'] == 'om-node-1'


def test_fuse_node_like_results_is_rank_based_not_source_order():
    graphiti_results = [
        {
            'uuid': 'graphiti-1',
            'name': 'Highest graphiti rank',
            'summary': 'top graphiti result',
            'attributes': {'source': 'graphiti'},
        },
        {
            'uuid': 'graphiti-2',
            'name': 'Lower graphiti rank',
            'summary': 'second graphiti result',
            'attributes': {'source': 'graphiti'},
        },
    ]
    om_results = [
        {
            'uuid': 'om-1',
            'name': 'Top OM rank',
            'summary': 'om result should outrank graphiti rank #2 by RRF',
            'attributes': {'source': 'om_primitive'},
        }
    ]

    fused = srv._fuse_node_like_results(
        primary=graphiti_results,
        supplemental=om_results,
        max_items=2,
    )

    # Source-order merge/truncate would return graphiti-1, graphiti-2.
    # Rank fusion should keep graphiti-1 and include OM rank #1 next.
    assert [item['uuid'] for item in fused] == ['graphiti-1', 'om-1']


def test_fuse_node_like_results_can_promote_om_with_corroboration_boost():
    graphiti_results = [
        {
            'uuid': 'graphiti-1',
            'name': 'Unrelated graphiti lead',
            'summary': 'different concept with no corroboration',
            'attributes': {'source': 'graphiti'},
        },
        {
            'uuid': 'graphiti-2',
            'name': 'Cache saturation incident',
            'summary': 'latency spike from cache saturation in checkout path',
            'attributes': {'source': 'graphiti'},
        },
    ]
    om_results = [
        {
            'uuid': 'om-1',
            'name': 'Cache saturation incident',
            'summary': 'latency spike from cache saturation in checkout path',
            'attributes': {'source': 'om_primitive'},
        }
    ]

    fused = srv._fuse_node_like_results(
        primary=graphiti_results,
        supplemental=om_results,
        max_items=3,
    )

    # OM rank #1 receives corroboration boost from graphiti-2 and can outrank
    # the uncorroborated graphiti rank #1 item.
    assert fused[0]['uuid'] == 'om-1'
    assert any(item['uuid'].startswith('graphiti-') for item in fused)


def test_fuse_node_like_results_tie_breaks_deterministically_by_uuid(monkeypatch):
    monkeypatch.setattr(
        srv,
        '_SEARCH_FUSION_SOURCE_WEIGHTS',
        {'graphiti': 1.0, 'om_primitive': 1.0},
    )
    monkeypatch.setattr(srv, '_SEARCH_FUSION_REQUIRE_GRAPHITI_FLOOR', False)

    graphiti_results = [
        {
            'uuid': 'tie-z',
            'name': 'Graphiti tie z',
            'summary': 'shared tie candidate z',
            'attributes': {'source': 'graphiti'},
        },
        {
            'uuid': 'tie-a',
            'name': 'Graphiti tie a',
            'summary': 'shared tie candidate a',
            'attributes': {'source': 'graphiti'},
        },
    ]
    om_results = [
        {
            'uuid': 'tie-a',
            'name': 'OM tie a',
            'summary': 'shared tie candidate a',
            'attributes': {'source': 'om_primitive'},
        },
        {
            'uuid': 'tie-z',
            'name': 'OM tie z',
            'summary': 'shared tie candidate z',
            'attributes': {'source': 'om_primitive'},
        },
    ]

    fused = srv._fuse_node_like_results(
        primary=graphiti_results,
        supplemental=om_results,
        max_items=2,
    )

    assert [item['uuid'] for item in fused] == ['tie-a', 'tie-z']


def test_fuse_node_like_results_enforces_graphiti_floor_when_window_allows(monkeypatch):
    monkeypatch.setattr(
        srv,
        '_SEARCH_FUSION_SOURCE_WEIGHTS',
        {'graphiti': 0.10, 'om_primitive': 1.0},
    )
    monkeypatch.setattr(srv, '_SEARCH_FUSION_REQUIRE_GRAPHITI_FLOOR', True)

    graphiti_results = [
        {
            'uuid': 'graphiti-1',
            'name': 'Graphiti-only fallback result',
            'summary': 'must stay present in mixed/all-lane fusion window',
            'attributes': {'source': 'graphiti'},
        }
    ]
    om_results = [
        {
            'uuid': 'om-1',
            'name': 'OM leader',
            'summary': 'top om result',
            'attributes': {'source': 'om_primitive'},
        },
        {
            'uuid': 'om-2',
            'name': 'OM runner-up',
            'summary': 'second om result',
            'attributes': {'source': 'om_primitive'},
        },
    ]

    fused = srv._fuse_node_like_results(
        primary=graphiti_results,
        supplemental=om_results,
        max_items=2,
    )

    # Even when source weights would otherwise produce OM-only top-k, mixed
    # scope must retain at least one Graphiti result when window size > 1.
    assert any(item['uuid'] == 'graphiti-1' for item in fused)


@pytest.mark.asyncio
async def test_search_nodes_oversized_max_nodes_is_capped_and_adapter_receives_cap_in_om_only_scope(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(srv, '_SEARCH_RATE_LIMIT_ENABLED', False)
    monkeypatch.setattr(srv, '_MAX_NODES_CAP', 2)
    monkeypatch.setattr(srv, 'config', _base_config())

    search_service = SimpleNamespace(
        includes_observational_memory=lambda group_ids: True,
        search_observational_nodes=AsyncMock(
            return_value=[
                {
                    'uuid': f'om-node-{idx}',
                    'content': f'OM node {idx}',
                    'created_at': '2026-03-01T12:00:00Z',
                    'group_id': 's1_observational_memory',
                    'status': 'open',
                    'semantic_domain': 'sessions_main',
                    'urgency_score': idx,
                    'lexical_score': idx,
                }
                for idx in range(4)
            ]
        ),
    )
    fake_client = SimpleNamespace(
        driver=SimpleNamespace(execute_query=AsyncMock()),
        search_=AsyncMock(),
    )
    fake_service = SimpleNamespace(
        config=SimpleNamespace(database=SimpleNamespace(provider='neo4j')),
        get_client=AsyncMock(return_value=fake_client),
        get_client_for_group=AsyncMock(return_value=fake_client),
    )

    monkeypatch.setattr(srv, 'search_service', search_service)
    monkeypatch.setattr(srv, 'graphiti_service', fake_service)

    response = await srv.search_nodes(
        query='latency incident',
        group_ids=['s1_observational_memory'],
        max_nodes=10,
    )

    assert response['message'] == 'Nodes retrieved successfully'
    assert len(response['nodes']) == 2
    search_service.search_observational_nodes.assert_awaited_once()
    assert (
        search_service.search_observational_nodes.await_args.kwargs['max_nodes']
        == 2
    )
    fake_service.get_client_for_group.assert_not_called()
    fake_service.get_client.assert_not_called()


@pytest.mark.asyncio
async def test_search_memory_facts_oversized_max_facts_is_capped_and_adapter_receives_cap_in_om_only_scope(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(srv, '_SEARCH_RATE_LIMIT_ENABLED', False)
    monkeypatch.setattr(srv, '_MAX_FACTS_CAP', 2)
    monkeypatch.setattr(srv, 'config', _base_config())

    search_service = SimpleNamespace(
        includes_observational_memory=lambda group_ids: True,
        search_observational_facts=AsyncMock(
            return_value=[
                {
                    'uuid': f'om-fact-{idx}',
                    'relation_type': 'RESOLVES',
                    'source_node_id': 'om-node-1',
                    'target_node_id': 'om-node-2',
                    'created_at': '2026-03-01T13:00:00Z',
                    'group_id': 's1_observational_memory',
                    'source_content': f'source-{idx}',
                    'target_content': f'target-{idx}',
                    'lexical_score': idx,
                }
                for idx in range(4)
            ]
        ),
    )
    fake_client = SimpleNamespace(
        driver=SimpleNamespace(execute_query=AsyncMock()),
        search_=AsyncMock(),
    )
    fake_service = SimpleNamespace(
        config=SimpleNamespace(database=SimpleNamespace(provider='neo4j')),
        get_client=AsyncMock(return_value=fake_client),
        get_client_for_group=AsyncMock(return_value=fake_client),
    )

    monkeypatch.setattr(srv, 'search_service', search_service)
    monkeypatch.setattr(srv, 'graphiti_service', fake_service)

    response = await srv.search_memory_facts(
        query='latency incident',
        group_ids=['s1_observational_memory'],
        max_facts=10,
    )

    assert response['message'] == 'Facts retrieved successfully'
    assert len(response['facts']) == 2
    search_service.search_observational_facts.assert_awaited_once()
    assert (
        search_service.search_observational_facts.await_args.kwargs['max_facts']
        == 2
    )
    fake_service.get_client_for_group.assert_not_called()
    fake_service.get_client.assert_not_called()


@pytest.mark.asyncio
async def test_search_nodes_mixed_lane_caps_output_and_adapter_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(srv, '_SEARCH_RATE_LIMIT_ENABLED', False)
    monkeypatch.setattr(srv, '_MAX_NODES_CAP', 2)
    monkeypatch.setattr(srv, 'config', _base_config())

    graphiti_nodes = [
        SimpleNamespace(
            uuid=f'entity-{idx}',
            name=f'Entity {idx}',
            labels=['Entity'],
            created_at=datetime(2026, 3, 1, 14, 0, tzinfo=timezone.utc),
            summary=f'Graphiti result {idx}',
            group_id='s1_sessions_main',
            attributes={'source': 'graphiti'},
        )
        for idx in range(1, 5)
    ]

    search_service = SimpleNamespace(
        includes_observational_memory=lambda group_ids: True,
        search_observational_nodes=AsyncMock(
            return_value=[
                {
                    'uuid': f'om-node-{idx}',
                    'content': f'OM node {idx}',
                    'created_at': '2026-03-01T12:00:00Z',
                    'group_id': 's1_observational_memory',
                    'status': 'open',
                    'semantic_domain': 'sessions_main',
                    'urgency_score': idx,
                    'lexical_score': idx,
                }
                for idx in range(4)
            ]
        ),
    )
    fake_client = SimpleNamespace(
        search_=AsyncMock(
            return_value=SimpleNamespace(nodes=graphiti_nodes, edges=[])
        ),
        driver=SimpleNamespace(execute_query=AsyncMock()),
    )
    fake_service = SimpleNamespace(
        config=SimpleNamespace(database=SimpleNamespace(provider='neo4j')),
        get_client=AsyncMock(return_value=fake_client),
        get_client_for_group=AsyncMock(return_value=fake_client),
    )

    monkeypatch.setattr(srv, 'search_service', search_service)
    monkeypatch.setattr(srv, 'graphiti_service', fake_service)

    response = await srv.search_nodes(
        query='all lanes om',
        group_ids=[],
        max_nodes=10,
    )

    assert response['message'] == 'Nodes retrieved successfully'
    assert len(response['nodes']) <= 2
    assert len(response['nodes']) == 2
    search_service.search_observational_nodes.assert_awaited_once()
    assert (
        search_service.search_observational_nodes.await_args.kwargs['max_nodes']
        == 2
    )
    fake_service.get_client_for_group.assert_awaited_once()
    fake_client.search_.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_memory_facts_mixed_lane_caps_output_and_adapter_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(srv, '_SEARCH_RATE_LIMIT_ENABLED', False)
    monkeypatch.setattr(srv, '_MAX_FACTS_CAP', 2)
    monkeypatch.setattr(srv, 'config', _base_config())

    class _FakeEdge:
        def __init__(self, idx: int):
            self._idx = idx

        def model_dump(self, *, mode: str = 'json', exclude: set[str] | None = None):
            return {
                'uuid': f'edge-{self._idx}',
                'name': 'RELATES_TO',
                'fact': f'RELATES_TO: source {self._idx} -> target {self._idx}',
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

    search_service = SimpleNamespace(
        includes_observational_memory=lambda group_ids: True,
        search_observational_facts=AsyncMock(
            return_value=[
                {
                    'uuid': f'om-fact-{idx}',
                    'relation_type': 'RESOLVES',
                    'source_node_id': 'om-node-1',
                    'target_node_id': 'om-node-2',
                    'created_at': '2026-03-01T13:00:00Z',
                    'group_id': 's1_observational_memory',
                    'source_content': f'source-{idx}',
                    'target_content': f'target-{idx}',
                    'lexical_score': idx,
                }
                for idx in range(4)
            ]
        ),
    )
    fake_client = SimpleNamespace(
        search_=AsyncMock(
            return_value=SimpleNamespace(
                nodes=[],
                edges=[_FakeEdge(1), _FakeEdge(2), _FakeEdge(3)],
            )
        ),
        driver=SimpleNamespace(execute_query=AsyncMock()),
    )
    fake_service = SimpleNamespace(
        config=SimpleNamespace(database=SimpleNamespace(provider='neo4j')),
        get_client=AsyncMock(return_value=fake_client),
        get_client_for_group=AsyncMock(return_value=fake_client),
    )

    monkeypatch.setattr(srv, 'search_service', search_service)
    monkeypatch.setattr(srv, 'graphiti_service', fake_service)

    response = await srv.search_memory_facts(
        query='latency incident',
        group_ids=[],
        max_facts=10,
    )

    assert response['message'] == 'Facts retrieved successfully'
    assert len(response['facts']) <= 2
    assert len(response['facts']) == 2
    search_service.search_observational_facts.assert_awaited_once()
    assert (
        search_service.search_observational_facts.await_args.kwargs['max_facts']
        == 2
    )
    fake_service.get_client_for_group.assert_awaited_once()
    fake_client.search_.assert_awaited_once()
