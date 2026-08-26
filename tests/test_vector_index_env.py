"""Env-var opt-in for the Neo4j vector index path.

Deployments that construct Graphiti indirectly (e.g. the MCP server, which calls
Graphiti(uri=..., user=..., password=...) and never touches Neo4jDriver directly)
can use GRAPHITI_NEO4J_USE_VECTOR_INDEX. Explicit constructor configuration still
wins, while index dimensions come from the configured embedder rather than a second
environment setting that can drift from it.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.graphiti import Graphiti


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
        for raw in ('0', 'false', 'False', 'no', 'off', '', 'malformed'):
            monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', raw)
            assert _make_driver().use_vector_index is False, raw

    def test_explicit_argument_overrides_env(self, monkeypatch):
        monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', 'false')
        assert _make_driver(use_vector_index=True).use_vector_index is True

        monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', 'true')
        assert _make_driver(use_vector_index=False).use_vector_index is False

    def test_embedding_dim_env_does_not_override_embedder_coupled_default(self, monkeypatch):
        from graphiti_core.driver.neo4j_driver import EMBEDDING_DIM

        monkeypatch.setenv('GRAPHITI_NEO4J_EMBEDDING_DIM', '2560')
        assert _make_driver().embedding_dim == EMBEDDING_DIM

    def test_embedding_dim_explicit_argument_wins(self, monkeypatch):
        monkeypatch.setenv('GRAPHITI_NEO4J_EMBEDDING_DIM', '2560')
        assert _make_driver(embedding_dim=1024).embedding_dim == 1024

    @pytest.mark.parametrize('embedding_dim', [1.5, True, 0, -1])
    def test_embedding_dim_rejects_invalid_values_without_vector_index(self, embedding_dim):
        with pytest.raises(ValueError, match='embedding_dim must be a positive integer'):
            _make_driver(embedding_dim=embedding_dim, use_vector_index=False)

    @pytest.mark.parametrize('embedding_dim', [4097, 8192])
    @pytest.mark.parametrize('use_vector_index', [False, None])
    def test_large_embedding_dim_is_accepted_when_vector_index_is_disabled(
        self, monkeypatch, embedding_dim, use_vector_index
    ):
        monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', 'false')

        driver = _make_driver(
            embedding_dim=embedding_dim,
            use_vector_index=use_vector_index,
        )

        assert driver.embedding_dim == embedding_dim
        assert driver.use_vector_index is False

    def test_large_embedding_dim_is_accepted_with_default_opt_out(self, monkeypatch):
        monkeypatch.delenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', raising=False)

        driver = _make_driver(embedding_dim=8192)

        assert driver.embedding_dim == 8192
        assert driver.use_vector_index is False

    @pytest.mark.parametrize('embedding_dim', [4097, 8192])
    def test_large_embedding_dim_is_rejected_with_explicit_vector_index(self, embedding_dim):
        with pytest.raises(ValueError, match='embedding_dim must not exceed 4096'):
            _make_driver(embedding_dim=embedding_dim, use_vector_index=True)

    @pytest.mark.parametrize('embedding_dim', [4097, 8192])
    def test_large_embedding_dim_is_rejected_with_environment_vector_index(
        self, monkeypatch, embedding_dim
    ):
        monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', 'true')

        with pytest.raises(ValueError, match='embedding_dim must not exceed 4096'):
            _make_driver(embedding_dim=embedding_dim)

    @pytest.mark.parametrize('use_vector_index', [None, False])
    def test_graphiti_accepts_large_configured_dimension_without_vector_index(
        self, monkeypatch, use_vector_index
    ):
        monkeypatch.delenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', raising=False)
        monkeypatch.setattr('graphiti_core.driver.neo4j_driver.AsyncGraphDatabase.driver', Mock())
        monkeypatch.setattr(
            'graphiti_core.graphiti.GraphitiClients', Mock(return_value=SimpleNamespace())
        )
        monkeypatch.setattr('graphiti_core.graphiti.NodeNamespace', Mock())
        monkeypatch.setattr('graphiti_core.graphiti.EdgeNamespace', Mock())
        embedder = OpenAIEmbedder(
            config=OpenAIEmbedderConfig(api_key='test-key', embedding_dim=8192)
        )

        graphiti = Graphiti(
            uri='bolt://unused',
            user='neo4j',
            password='unused',
            embedder=embedder,
            llm_client=SimpleNamespace(set_tracer=Mock()),
            cross_encoder=SimpleNamespace(),
            use_vector_index=use_vector_index,
        )

        assert graphiti.driver.embedding_dim == 8192
        assert graphiti.driver.use_vector_index is False

    def test_embedding_dim_ignores_disconnected_malformed_env(self, monkeypatch):
        from graphiti_core.driver.neo4j_driver import EMBEDDING_DIM

        monkeypatch.setenv('GRAPHITI_NEO4J_EMBEDDING_DIM', 'not-a-number')
        assert _make_driver().embedding_dim == EMBEDDING_DIM

    def test_embedding_dim_ignores_disconnected_non_positive_env(self, monkeypatch):
        from graphiti_core.driver.neo4j_driver import EMBEDDING_DIM

        for raw in ('0', '-1'):
            monkeypatch.setenv('GRAPHITI_NEO4J_EMBEDDING_DIM', raw)
            assert _make_driver().embedding_dim == EMBEDDING_DIM
