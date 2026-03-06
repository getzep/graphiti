import asyncio
from types import SimpleNamespace

from tests.helpers_mcp_import import load_graphiti_mcp_server

server = load_graphiti_mcp_server()


def _run(coro):
    return asyncio.run(coro)


def _test_config():
    return SimpleNamespace(
        database=SimpleNamespace(provider='neo4j'),
        graphiti=SimpleNamespace(
            group_id='s1_sessions_main',
            lane_aliases={
                'sessions_main': ['s1_sessions_main'],
                'observational_memory': ['s1_observational_memory'],
                'curated': ['s1_curated_refs'],
            },
        ),
    )


def _setup_falkordb_runtime(isolate_calls):
    original_graphiti_service = server.graphiti_service
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    if server.config is None:
        server.config = _test_config()
    original_provider = server.config.database.provider
    original_group_id = server.config.graphiti.group_id
    original_methods = (
        server.search_service.search_observational_nodes,
        server.search_service.search_observational_facts,
    )
    original_search_builders = (
        server._build_node_search_config,
        server._build_edge_search_config,
    )

    async def fake_search_observational_nodes(
        *, graphiti_service, query, group_ids, max_nodes, entity_types
    ):
        isolate_calls['nodes'] = True
        return [{'uuid': 'should-not-be-called'}]

    async def fake_search_observational_facts(
        *, graphiti_service, query, group_ids, max_facts, center_node_uuid
    ):
        isolate_calls['facts'] = True
        return [{'uuid': 'should-not-be-called'}]

    class _FakeClient:
        async def search_(self, *args, **kwargs):
            return SimpleNamespace(nodes=[], edges=[])

    class _FakeService:
        async def get_client_for_group(self, *_args, **_kwargs):
            return _FakeClient()

        async def get_client(self, *_args, **_kwargs):
            return _FakeClient()

    server._SEARCH_RATE_LIMIT_ENABLED = False
    server.config.database.provider = 'falkordb'
    server.config.graphiti.group_id = 's1_sessions_main'
    server.graphiti_service = _FakeService()
    server.search_service.search_observational_nodes = fake_search_observational_nodes
    server.search_service.search_observational_facts = fake_search_observational_facts
    server._build_node_search_config = lambda *_args, **_kwargs: object()
    server._build_edge_search_config = lambda *_args, **_kwargs: object()

    return (
        isolate_calls,
        original_graphiti_service,
        original_rate_limit,
        original_provider,
        original_group_id,
        original_methods,
        original_search_builders,
    )


def _teardown_falkordb_runtime(state):
    (
        _isolate_calls,
        original_graphiti_service,
        original_rate_limit,
        original_provider,
        original_group_id,
        original_methods,
        original_search_builders,
    ) = state
    server.graphiti_service = original_graphiti_service
    server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
    server.config.database.provider = original_provider
    server.config.graphiti.group_id = original_group_id
    server.search_service.search_observational_nodes = original_methods[0]
    server.search_service.search_observational_facts = original_methods[1]
    server._build_node_search_config = original_search_builders[0]
    server._build_edge_search_config = original_search_builders[1]


def test_falkordb_uses_graphiti_search_when_om_observational_scope_for_nodes():
    isolate_calls = {'nodes': False, 'facts': False}
    state = _setup_falkordb_runtime(isolate_calls)
    try:
        response = _run(
            server.search_nodes(
                query='test query',
                group_ids=['s1_observational_memory'],
                lane_alias=None,
                search_mode='hybrid',
                max_nodes=5,
                entity_types=None,
                ctx=None,
            )
        )
    finally:
        _teardown_falkordb_runtime(state)

    assert isinstance(response, dict)
    assert response['message'] in {'No relevant nodes found', 'Nodes retrieved successfully'}
    assert isolate_calls['nodes'] is False


def test_falkordb_uses_graphiti_search_when_om_observational_scope_for_facts():
    isolate_calls = {'nodes': False, 'facts': False}
    state = _setup_falkordb_runtime(isolate_calls)
    try:
        response = _run(
            server.search_memory_facts(
                query='test query',
                group_ids=['s1_observational_memory'],
                lane_alias=None,
                search_mode='hybrid',
                max_facts=5,
                center_node_uuid=None,
                ctx=None,
            )
        )
    finally:
        _teardown_falkordb_runtime(state)

    assert isinstance(response, dict)
    assert response['message'] in {'No relevant facts found', 'Facts retrieved successfully'}
    assert isolate_calls['facts'] is False
