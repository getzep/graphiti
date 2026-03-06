import asyncio
from types import SimpleNamespace

from tests.helpers_mcp_import import load_graphiti_mcp_server

server = load_graphiti_mcp_server()

REQUIRED_FIELDS = {
    'source_lane',
    'source_node_id',
    'source_event_id',
    'source_group_id',
    'evidence_refs',
    'created_at',
}


def _run(coro):
    return asyncio.run(coro)


def _test_config(default_group_id: str):
    return SimpleNamespace(
        database=SimpleNamespace(provider='neo4j'),
        graphiti=SimpleNamespace(
            group_id=default_group_id,
            lane_aliases={
                'sessions_main': ['s1_sessions_main'],
                'observational_memory': ['s1_observational_memory'],
                'curated': ['s1_curated_refs'],
            },
        ),
    )


def _execute_search_with_om_facts(
    om_facts,
    *,
    default_group_id: str = 's1_observational_memory',
    group_ids: list[str] | None = None,
):
    original_graphiti_service = server.graphiti_service
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    if server.config is None:
        server.config = _test_config(default_group_id)
    original_provider = server.config.database.provider
    original_group_id = server.config.graphiti.group_id

    effective_group_ids = ['s1_observational_memory'] if group_ids is None else group_ids

    server._SEARCH_RATE_LIMIT_ENABLED = False
    server.config.database.provider = 'neo4j'
    server.config.graphiti.group_id = default_group_id
    server.graphiti_service = object()

    async def fake_search_observational_facts(
        *, graphiti_service, query, group_ids, max_facts, center_node_uuid
    ):
        return om_facts

    original_method = server.search_service.search_observational_facts
    server.search_service.search_observational_facts = fake_search_observational_facts

    try:
        return _run(
            server.search_memory_facts(
                query='What happened with om bridge?',
                group_ids=effective_group_ids,
                lane_alias=None,
                search_mode='hybrid',
                max_facts=10,
                center_node_uuid=None,
                ctx=None,
            )
        )
    finally:
        server.search_service.search_observational_facts = original_method
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.config.database.provider = original_provider
        server.config.graphiti.group_id = original_group_id
        server.graphiti_service = original_graphiti_service


def test_om_candidate_bridge_emits_provenance_complete_rows_in_runtime_path():
    response = _execute_search_with_om_facts(
        [
            {
                'uuid': 'om-fact-1',
                'source_node_uuid': 'om-node-1',
                'group_id': 's1_observational_memory',
                'created_at': '2026-03-05T00:00:00Z',
            }
        ]
    )

    assert isinstance(response, dict)
    assert response['message'] == 'Facts retrieved successfully'
    candidate_rows = response['candidate_rows']
    assert isinstance(candidate_rows, list) and len(candidate_rows) == 1

    row = candidate_rows[0]
    assert row.keys() >= REQUIRED_FIELDS
    assert row['source_group_id'] == 's1_observational_memory'
    assert row['source_node_id'] == 'om-node-1'
    assert row['source_event_id'] == 'om-fact-1'
    assert row['source_lane'] == 's1_observational_memory'
    assert row['created_at'] == '2026-03-05T00:00:00Z'
    assert isinstance(row['evidence_refs'], list) and row['evidence_refs']


def test_om_candidate_bridge_source_lane_tracks_source_group_when_default_group_is_not_om():
    response = _execute_search_with_om_facts(
        [
            {
                'uuid': 'om-fact-1',
                'source_node_uuid': 'om-node-1',
                'group_id': 's1_observational_memory',
                'created_at': '2026-03-05T00:00:00Z',
            }
        ],
        default_group_id='s1_sessions_main',
    )

    assert isinstance(response, dict)
    assert response['message'] == 'Facts retrieved successfully'
    candidate_rows = response['candidate_rows']
    assert isinstance(candidate_rows, list) and len(candidate_rows) == 1

    row = candidate_rows[0]
    assert row['source_group_id'] == 's1_observational_memory'
    assert row['source_lane'] == row['source_group_id']
    assert row['source_lane'] != 's1_sessions_main'


def test_om_candidate_bridge_fail_closed_when_created_at_missing():
    response = _execute_search_with_om_facts(
        [
            {
                'uuid': 'om-fact-1',
                'source_node_uuid': 'om-node-1',
                'group_id': 's1_observational_memory',
            }
        ]
    )

    assert isinstance(response, dict)
    assert response['message'] == 'Facts retrieved successfully'
    assert 'candidate_rows' not in response
