import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

from graphiti_core.graphiti import Graphiti


def test_mcp_factory_dimension_reaches_default_driver_without_vector_index(monkeypatch):
    schema = ModuleType('config.schema')
    schema.DatabaseConfig = SimpleNamespace
    schema.EmbedderConfig = SimpleNamespace
    schema.LLMConfig = SimpleNamespace
    config_package = ModuleType('config')
    config_package.schema = schema
    monkeypatch.setitem(sys.modules, 'config', config_package)
    monkeypatch.setitem(sys.modules, 'config.schema', schema)

    factory_path = Path(__file__).parents[1] / 'mcp_server/src/services/factories.py'
    spec = importlib.util.spec_from_file_location('mcp_embedding_factories', factory_path)
    assert spec is not None and spec.loader is not None
    factories = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factories)

    embedder = factories.EmbedderFactory.create(
        SimpleNamespace(
            provider='openai',
            model='text-embedding-3-small',
            dimensions=8192,
            providers=SimpleNamespace(
                openai=SimpleNamespace(api_key='test-key', api_url='https://api.openai.com/v1')
            ),
        )
    )
    monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', 'false')
    monkeypatch.setattr('graphiti_core.driver.neo4j_driver.AsyncGraphDatabase.driver', Mock())
    monkeypatch.setattr(
        'graphiti_core.graphiti.GraphitiClients', Mock(return_value=SimpleNamespace())
    )
    monkeypatch.setattr('graphiti_core.graphiti.NodeNamespace', Mock())
    monkeypatch.setattr('graphiti_core.graphiti.EdgeNamespace', Mock())

    graphiti = Graphiti(
        uri='bolt://unused',
        user='neo4j',
        password='unused',
        embedder=embedder,
        llm_client=SimpleNamespace(set_tracer=Mock()),
        cross_encoder=SimpleNamespace(),
    )

    assert graphiti.driver.embedding_dim == 8192
    assert graphiti.driver.use_vector_index is False
