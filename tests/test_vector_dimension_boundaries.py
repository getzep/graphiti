"""Dimension contracts at the driver, Graphiti and actual MCP factory/service boundaries."""

import __future__

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest

import graphiti_core.driver.neo4j_driver as dm
from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient
from graphiti_core.embedder.client import EMBEDDING_DIM, EmbedderClient
from graphiti_core.embedder.gemini import GeminiEmbedder
from graphiti_core.graphiti import Graphiti

ROOT = Path(__file__).parents[1]


class BareEmbedder(EmbedderClient):
    async def create(self, input_data):
        return [0.1] * 768


class OpaqueEmbedder(BareEmbedder):
    @property
    def config(self):
        raise RuntimeError('private config')


class OpaqueDimensionEmbedder(BareEmbedder):
    class Config:
        @property
        def embedding_dim(self):
            raise RuntimeError('private dimension')

    config = Config()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', raising=False)
    monkeypatch.setattr(dm.AsyncGraphDatabase, 'driver', Mock())
    monkeypatch.setattr(dm.Neo4jDriver, 'build_indices_and_constraints', AsyncMock())
    monkeypatch.setattr('graphiti_core.graphiti.GraphitiClients', lambda **kw: NS(**kw))
    monkeypatch.setattr('graphiti_core.graphiti.NodeNamespace', Mock())
    monkeypatch.setattr('graphiti_core.graphiti.EdgeNamespace', Mock())


@pytest.mark.parametrize('dimension', [0, -1, False, 'auto', [], 8192])
@pytest.mark.parametrize('route', ['default', 'explicit', 'env-false', 'override'])
def test_direct_optout_accepts_dimension_without_validation(monkeypatch, dimension, route):
    options = {'use_vector_index': False} if route in ('explicit', 'override') else {}
    if route in ('env-false', 'override'):
        monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', str(route == 'override'))
    driver = dm.Neo4jDriver('bolt://unused', None, None, embedding_dim=dimension, **options)
    assert driver.use_vector_index is False
    # Opt-out retains the input verbatim; None alone retains the existing default.
    assert driver.embedding_dim is dimension


@pytest.mark.parametrize('dimension', [0, -1, False, 'auto', [], 8192])
@pytest.mark.parametrize('route', ['explicit', 'env'])
def test_direct_optin_still_rejects_invalid_dimension(monkeypatch, dimension, route):
    options = {'use_vector_index': True} if route == 'explicit' else {}
    if route == 'env':
        monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', 'true')
    with pytest.raises(ValueError, match='embedding_dim'):
        dm.Neo4jDriver('bolt://unused', None, None, embedding_dim=dimension, **options)


@pytest.mark.parametrize('kind', [OpaqueEmbedder, OpaqueDimensionEmbedder])
@pytest.mark.parametrize(
    'route', ['owned-explicit', 'owned-env', 'supplied-explicit', 'supplied-env']
)
def test_unreadable_optin_declaration_has_one_chained_value_error(monkeypatch, kind, route):
    if route.endswith('env'):
        monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', 'true')
    options = {'use_vector_index': True} if route.endswith('explicit') else {}
    if route.startswith('supplied'):
        options = {
            'graph_driver': dm.Neo4jDriver(
                'bolt://unused', None, None, embedding_dim=768, **options
            ),
            'use_vector_index': False,  # The supplied driver's flag is authoritative.
        }
    with pytest.raises(ValueError, match=kind.__name__ + '.*config.embedding_dim') as error:
        Graphiti(
            uri='bolt://unused',
            embedder=kind(),
            llm_client=NS(set_tracer=Mock()),
            cross_encoder=NS(),
            **options,
        )
    assert isinstance(error.value.__cause__, RuntimeError)
    assert 'disable the vector-index opt-in' in str(error.value)


def extract_class(path, name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(
        compile(
            ast.Module(body=[cls], type_ignores=[]),
            str(path),
            'exec',
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[name]


@pytest.mark.asyncio
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize(
    'kind', ['gemini', 'azure', 'bare', 'opaque', 'opaque-dimension', 'invalid', 'bool']
)
async def test_mcp_uses_effective_declaration_or_driver_default(monkeypatch, enabled, kind):
    # Real factory branches and embedder classes; only SDK construction is doubled.
    schema_ns = {'__name__': 'dimension_boundary_schema'}
    schema_path = ROOT / 'mcp_server/src/config/schema.py'
    exec(compile(schema_path.read_text(), str(schema_path), 'exec'), schema_ns)
    config = schema_ns['EmbedderConfig'](
        provider='azure_openai' if kind == 'azure' else 'gemini',
        model='models/text-embedding-004',
        dimensions=0,
        providers={
            'gemini': {'api_key': 'offline-placeholder'},
            'azure_openai': {'api_key': 'offline-placeholder', 'api_url': 'https://unused.invalid'},
        },
    )
    monkeypatch.setattr('openai.AsyncOpenAI', Mock())
    factory_ns = {
        '__name__': 'dimension_boundary_factory',
        '_validate_api_key': Mock(),
        'HAS_GEMINI_EMBEDDER': True,
        'HAS_AZURE_EMBEDDER': True,
        'GeminiEmbedder': lambda config: GeminiEmbedder(config=config, client=NS()),
        'AzureOpenAIEmbedderClient': AzureOpenAIEmbedderClient,
    }
    factory = extract_class(
        ROOT / 'mcp_server/src/services/factories.py', 'EmbedderFactory', factory_ns
    )
    embedder = factory.create(config)
    if kind == 'gemini':
        assert embedder.config.embedding_dim == 768
    elif kind not in ('azure', 'gemini'):
        embedder = {
            'bare': BareEmbedder,
            'opaque': OpaqueEmbedder,
            'opaque-dimension': OpaqueDimensionEmbedder,
            'invalid': lambda: NS(config=NS(embedding_dim='auto')),
            'bool': lambda: NS(config=NS(embedding_dim=True)),
        }[kind]()
    monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', str(enabled))
    driver_factory = Mock(side_effect=dm.Neo4jDriver)
    monkeypatch.setattr(dm, 'Neo4jDriver', driver_factory)
    namespace = {
        'asyncio': asyncio,
        'Graphiti': Graphiti,
        'logger': Mock(),
        'LLMClientFactory': NS(create=Mock(return_value=NS(set_tracer=Mock()))),
        'EmbedderFactory': NS(create=Mock(return_value=embedder)),
        'CrossEncoderFactory': NS(create=Mock(return_value=NS())),
        'DatabaseDriverFactory': NS(
            create_config=Mock(
                return_value={
                    'uri': 'bolt://unused',
                    'user': 'unused',
                    'password': 'unused',
                    'database': 'test',
                }
            )
        ),
        'build_entity_types': Mock(return_value={}),
        'build_edge_types': Mock(return_value={}),
        'build_edge_type_map': Mock(return_value={}),
    }
    service_cls = extract_class(
        ROOT / 'mcp_server/src/graphiti_mcp_server.py', 'GraphitiService', namespace
    )
    service = service_cls(
        NS(
            llm=NS(provider='mock', model='mock'),
            embedder=config,
            database=NS(provider='neo4j'),
            graphiti=NS(entity_types={}, edge_types={}, edge_type_map={}, group_id='test'),
        )
    )
    if enabled and kind != 'gemini':
        with pytest.raises(ValueError, match=type(embedder).__name__ + '.*config.embedding_dim'):
            await service.initialize()
        assert service.client is None
    else:
        await service.initialize()
        assert service.client.embedder is embedder
        assert service.client.driver.use_vector_index is enabled
        assert service.client.driver.embedding_dim == (768 if kind == 'gemini' else EMBEDDING_DIM)
        await service.client.driver._init_task
    if kind == 'gemini':
        assert driver_factory.call_args.kwargs['embedding_dim'] == 768
    else:
        assert 'embedding_dim' not in driver_factory.call_args.kwargs
    # Drain any caller-owned initialization task after rejected Graphiti construction.
    await asyncio.sleep(0)
