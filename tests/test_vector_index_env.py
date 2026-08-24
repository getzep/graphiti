"""Env-var opt-in for the Neo4j vector index path.

Deployments that construct Graphiti indirectly (e.g. the MCP server, which calls
Graphiti(uri=..., user=..., password=...) and never touches Neo4jDriver directly)
cannot pass use_vector_index=True. Support GRAPHITI_NEO4J_USE_VECTOR_INDEX so the
feature is reachable from configuration without a code change, while the explicit
constructor argument still wins when supplied.
"""

from unittest.mock import patch

from graphiti_core.driver.neo4j_driver import Neo4jDriver


def _make_driver(**kwargs):
    with patch('graphiti_core.driver.neo4j_driver.AsyncGraphDatabase.driver'):
        return Neo4jDriver(uri='bolt://localhost:7687', user='neo4j', password='x', **kwargs)


class TestUseVectorIndexEnvVar:
    def test_defaults_to_false_without_env(self, monkeypatch):
        monkeypatch.delenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', raising=False)
        assert _make_driver().use_vector_index is False

    def test_env_true_enables(self, monkeypatch):
        monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', 'true')
        assert _make_driver().use_vector_index is True

    def test_env_accepts_common_truthy_spellings(self, monkeypatch):
        for raw in ('1', 'TRUE', 'True', 'yes', 'on'):
            monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', raw)
            assert _make_driver().use_vector_index is True, raw

    def test_env_accepts_common_falsy_spellings(self, monkeypatch):
        for raw in ('0', 'false', 'False', 'no', 'off', ''):
            monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', raw)
            assert _make_driver().use_vector_index is False, raw

    def test_explicit_argument_overrides_env(self, monkeypatch):
        monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', 'false')
        assert _make_driver(use_vector_index=True).use_vector_index is True

        monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', 'true')
        assert _make_driver(use_vector_index=False).use_vector_index is False

    def test_embedding_dim_env_override(self, monkeypatch):
        monkeypatch.setenv('GRAPHITI_NEO4J_EMBEDDING_DIM', '2560')
        assert _make_driver().embedding_dim == 2560

    def test_embedding_dim_explicit_argument_wins(self, monkeypatch):
        monkeypatch.setenv('GRAPHITI_NEO4J_EMBEDDING_DIM', '2560')
        assert _make_driver(embedding_dim=1024).embedding_dim == 1024

    def test_embedding_dim_ignores_malformed_env(self, monkeypatch):
        from graphiti_core.driver.neo4j_driver import EMBEDDING_DIM

        monkeypatch.setenv('GRAPHITI_NEO4J_EMBEDDING_DIM', 'not-a-number')
        assert _make_driver().embedding_dim == EMBEDDING_DIM
