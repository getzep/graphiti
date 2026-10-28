"""Owned-driver opt-out must not impose a configuration contract on embedders."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from graphiti_core.embedder.client import EMBEDDING_DIM, EmbedderClient
from graphiti_core.graphiti import Graphiti


class BareEmbedder(EmbedderClient):
    async def create(self, input_data):
        return [0.1] * 768


class AutoDimensionEmbedder(BareEmbedder):
    config_reads = 0

    @property
    def config(self):
        self.config_reads += 1
        return SimpleNamespace(embedding_dim='auto')


class OpaqueConfigEmbedder(BareEmbedder):
    config_reads = 0

    @property
    def config(self):
        self.config_reads += 1
        raise RuntimeError('config is private to the custom embedder')


@pytest.fixture(autouse=True)
def offline_dependencies(monkeypatch):
    monkeypatch.delenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', raising=False)
    factory = Mock()
    monkeypatch.setattr('graphiti_core.driver.neo4j_driver.AsyncGraphDatabase.driver', factory)
    monkeypatch.setattr(
        'graphiti_core.graphiti.GraphitiClients',
        Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    monkeypatch.setattr('graphiti_core.graphiti.NodeNamespace', Mock())
    monkeypatch.setattr('graphiti_core.graphiti.EdgeNamespace', Mock())
    return factory


def make_graphiti(embedder, **options):
    return Graphiti(
        uri='bolt://unused',
        embedder=embedder,
        llm_client=SimpleNamespace(set_tracer=Mock()),
        cross_encoder=SimpleNamespace(),
        **options,
    )


@pytest.mark.parametrize(
    'embedder_type', [AutoDimensionEmbedder, OpaqueConfigEmbedder, BareEmbedder]
)
@pytest.mark.parametrize('opt_out', ['default', 'explicit', 'env-false', 'env-override'])
def test_owned_opt_out_never_reads_embedder_config(monkeypatch, embedder_type, opt_out):
    options = {'use_vector_index': False} if opt_out in ('explicit', 'env-override') else {}
    if opt_out == 'env-false':
        monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', 'false')
    elif opt_out == 'env-override':
        monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', 'true')
    embedder = embedder_type()

    graphiti = make_graphiti(embedder, **options)

    assert graphiti.embedder is embedder
    assert graphiti.driver.use_vector_index is False
    assert graphiti.driver.embedding_dim == EMBEDDING_DIM
    assert getattr(embedder, 'config_reads', 0) == 0


@pytest.mark.parametrize(
    'embedder_type', [AutoDimensionEmbedder, OpaqueConfigEmbedder, BareEmbedder]
)
@pytest.mark.parametrize('opt_in', ['explicit', 'env'])
def test_owned_opt_in_rejects_unusable_config_before_driver_creation(
    monkeypatch, offline_dependencies, embedder_type, opt_in
):
    options = {'use_vector_index': True} if opt_in == 'explicit' else {}
    if opt_in == 'env':
        monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', 'true')
    embedder = embedder_type()

    with pytest.raises(
        ValueError, match=f'{embedder_type.__name__}.*config.embedding_dim'
    ) as error:
        make_graphiti(embedder, **options)

    assert 'positive integer' in str(error.value)
    assert 'disable' in str(error.value)
    offline_dependencies.assert_not_called()
