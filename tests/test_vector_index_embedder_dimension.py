"""Vector-index opt-in requires a declared, compatible embedder dimension."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient
from graphiti_core.embedder.client import EMBEDDING_DIM, EmbedderClient
from graphiti_core.graphiti import Graphiti
from graphiti_core.search.search_filters import SearchFilters


class BareEmbedder(EmbedderClient):
    async def create(self, input_data):
        return [0.1] * 1536


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


def make_embedder(kind):
    if kind == 'azure':
        sdk = SimpleNamespace(
            embeddings=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * 1536)])
                )
            )
        )
        return AzureOpenAIEmbedderClient(sdk)
    return BareEmbedder()


def make_graphiti(embedder, **kwargs):
    return Graphiti(
        uri='bolt://unused',
        embedder=embedder,
        llm_client=SimpleNamespace(set_tracer=Mock()),
        cross_encoder=SimpleNamespace(),
        **kwargs,
    )


@pytest.mark.parametrize('kind', ['azure', 'custom'])
@pytest.mark.parametrize('opt_in', ['explicit', '1', 'TRUE', 'yes', 'on'])
def test_owned_driver_rejects_undeclared_dimension_before_driver_creation(
    monkeypatch, offline_dependencies, kind, opt_in
):
    embedder = make_embedder(kind)
    options = {'use_vector_index': True} if opt_in == 'explicit' else {}
    if opt_in != 'explicit':
        monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', opt_in)
    with pytest.raises(ValueError) as error:
        make_graphiti(embedder, **options)
    message = str(error.value)
    assert type(embedder).__name__ in message
    assert 'embedding_dim' in message
    assert 'disable' in message.lower()
    assert 'embedder' in message.lower()
    offline_dependencies.assert_not_called()


@pytest.mark.parametrize('kind', ['azure', 'custom'])
def test_supplied_driver_rejects_undeclared_dimension(kind):
    embedder = make_embedder(kind)
    driver = Neo4jDriver('bolt://unused', None, None, use_vector_index=True)
    # The supplied driver's flag is authoritative, not Graphiti's owned-driver option.
    with pytest.raises(ValueError, match=type(embedder).__name__):
        make_graphiti(embedder, graph_driver=driver, use_vector_index=False)


@pytest.mark.parametrize('dimension', [None, 0, -1, True, 1.5, '768'])
@pytest.mark.parametrize('supplied', [False, True])
def test_opt_in_rejects_non_positive_integer_declarations(dimension, supplied):
    embedder = SimpleNamespace(config=SimpleNamespace(embedding_dim=dimension))
    options = (
        {'graph_driver': Neo4jDriver('bolt://unused', None, None, use_vector_index=True)}
        if supplied
        else {'use_vector_index': True}
    )
    with pytest.raises(ValueError, match='SimpleNamespace.*embedding_dim'):
        make_graphiti(embedder, **options)


@pytest.mark.parametrize('dimension', [768, 1536])
def test_supplied_driver_rejects_dimension_mismatch(dimension):
    embedder = SimpleNamespace(config=SimpleNamespace(embedding_dim=dimension))
    driver = Neo4jDriver('bolt://unused', None, None, use_vector_index=True)
    with pytest.raises(ValueError) as error:
        make_graphiti(embedder, graph_driver=driver)
    message = str(error.value)
    assert 'SimpleNamespace' in message
    assert str(dimension) in message
    assert str(driver.embedding_dim) in message
    assert 'embedding_dim' in message


@pytest.mark.parametrize('supplied', [False, True])
@pytest.mark.parametrize('dimension', [768, 1536])
def test_opt_in_accepts_matching_declared_dimension(supplied, dimension):
    embedder = SimpleNamespace(config=SimpleNamespace(embedding_dim=dimension))
    options = {'use_vector_index': True}
    if supplied:
        options['graph_driver'] = Neo4jDriver(
            'bolt://unused', None, None, embedding_dim=dimension, use_vector_index=True
        )
    graphiti = make_graphiti(embedder, **options)
    assert graphiti.driver.embedding_dim == dimension
    assert graphiti.driver.use_vector_index is True


@pytest.mark.parametrize('kind', ['azure', 'custom'])
@pytest.mark.parametrize('supplied', [False, True])
@pytest.mark.parametrize('opt_out', ['default', 'explicit', 'env'])
def test_opt_out_preserves_undeclared_embedder_output_and_exact_search(
    monkeypatch, kind, supplied, opt_out
):
    embedder = make_embedder(kind)
    options = {}
    if opt_out == 'explicit':
        monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', 'true')
        options['use_vector_index'] = False
    elif opt_out == 'env':
        monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', 'false')
    if supplied:
        options['graph_driver'] = Neo4jDriver(
            'bolt://unused', None, None, embedding_dim=768, use_vector_index=False
        )
    graphiti = make_graphiti(embedder, **options)
    assert graphiti.embedder is embedder
    assert graphiti.driver.embedding_dim == (768 if supplied else EMBEDDING_DIM)
    assert graphiti.driver.use_vector_index is False
    query = AsyncMock(return_value=([], None, None))
    monkeypatch.setattr(graphiti.driver, 'execute_query', query)

    async def search():
        vector = await embedder.create('query')
        assert len(vector) == 1536
        assert (
            await graphiti.driver.search_ops.node_similarity_search(
                graphiti.driver, vector, SearchFilters(), limit=1
            )
            == []
        )
        assert (
            await graphiti.driver.search_ops.edge_similarity_search(
                graphiti.driver, vector, None, None, SearchFilters(), limit=1
            )
            == []
        )

    asyncio.run(search())
    assert query.await_count == 2
    for call in query.await_args_list:
        assert 'vector.similarity.cosine' in call.args[0]
        assert 'db.index.vector.query' not in call.args[0]


def test_supplied_opt_out_does_not_inspect_embedder_config():
    class OpaqueEmbedder(BareEmbedder):
        @property
        def config(self):
            raise AssertionError('opt-out must not inspect the embedder config')

    driver = Neo4jDriver('bolt://unused', None, None, use_vector_index=False)
    assert make_graphiti(OpaqueEmbedder(), graph_driver=driver).driver is driver
