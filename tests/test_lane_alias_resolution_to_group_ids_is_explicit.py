from types import SimpleNamespace

from tests.helpers_mcp_import import load_graphiti_mcp_server

server = load_graphiti_mcp_server()
_resolve_effective_group_ids = server._resolve_effective_group_ids


def _ensure_alias_config() -> None:
    server.config = SimpleNamespace(
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


def test_explicit_group_ids_block_implicit_aliases():
    _ensure_alias_config()
    effective_group_ids, invalid_aliases = _resolve_effective_group_ids(
        group_ids=['s1_curated_refs'],
        lane_alias=['sessions_main'],
    )

    assert effective_group_ids == ['s1_curated_refs']
    assert invalid_aliases == []


def test_empty_lane_aliases_are_respected_without_alias_validation_error():
    _ensure_alias_config()
    effective_group_ids, invalid_aliases = _resolve_effective_group_ids(
        group_ids=None,
        lane_alias=[],
    )

    assert isinstance(effective_group_ids, list)
    assert invalid_aliases == []
