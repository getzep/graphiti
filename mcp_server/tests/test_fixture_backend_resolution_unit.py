import pytest
import test_fixtures


def test_resolve_test_database_fails_fast_without_nas_or_local_falkordb(monkeypatch):
    monkeypatch.setattr(test_fixtures, '_local_falkordb_reachable', lambda uri: False)

    with pytest.raises(RuntimeError, match=r'mcp_server/\.env\.nas'):
        test_fixtures.resolve_test_database('falkordb', {}, {})


def test_resolve_test_database_prefers_nas_neo4j_when_available(monkeypatch):
    monkeypatch.setattr(test_fixtures, '_local_falkordb_reachable', lambda uri: False)

    resolved = test_fixtures.resolve_test_database(
        'falkordb',
        {},
        {'NEO4J_URI': 'bolt://192.168.1.10:7687'},
    )

    assert resolved == 'neo4j'
