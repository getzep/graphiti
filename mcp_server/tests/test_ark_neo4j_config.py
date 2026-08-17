import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import graphiti_mcp_server
from config.schema import (
    DatabaseConfig,
    DatabaseProvidersConfig,
    GraphitiConfig,
    Neo4jProviderConfig,
)
from services.factories import DatabaseDriverFactory

ARK_CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'config-ark-neo4j.yaml'


def _set_ark_config_environment(monkeypatch, *, database='neo4j'):
    monkeypatch.setenv('CONFIG_PATH', str(ARK_CONFIG_PATH))
    monkeypatch.setenv('ARK_API_KEY', 'ark-placeholder')
    monkeypatch.setenv('ARK_CHAT_MODEL', 'ep-placeholder')
    monkeypatch.setenv('NEO4J_PASSWORD', 'password-placeholder')
    monkeypatch.setenv('NEO4J_DATABASE', database)
    monkeypatch.delenv('LLM__MODEL', raising=False)
    monkeypatch.delenv('LLM__STRUCTURED_OUTPUT_MODE', raising=False)
    monkeypatch.delenv('EMBEDDER__DIMENSIONS', raising=False)
    monkeypatch.delenv('GRAPHITI__GROUP_ID', raising=False)
    monkeypatch.delenv('DATABASE__PROVIDERS__NEO4J__DATABASE', raising=False)


def test_ark_yaml_keeps_default_group_aligned_with_neo4j_database(monkeypatch):
    _set_ark_config_environment(monkeypatch, database='knowledge')
    monkeypatch.setenv('EMBEDDING_DIM', '256')

    config = GraphitiConfig()

    assert config.llm.model == 'ep-placeholder'
    assert config.llm.structured_output_mode == 'prompt_only'
    assert config.embedder.provider == 'local_hash'
    assert config.embedder.dimensions == 256
    assert config.database.providers.neo4j is not None
    assert config.database.providers.neo4j.database == 'knowledge'
    assert config.graphiti.group_id == 'knowledge'


def test_ark_yaml_fails_fast_when_chat_model_is_missing(monkeypatch):
    _set_ark_config_environment(monkeypatch)
    monkeypatch.delenv('ARK_CHAT_MODEL')

    with pytest.raises(ValidationError, match='llm.model'):
        GraphitiConfig()


def test_neo4j_factory_requires_password(monkeypatch):
    monkeypatch.delenv('NEO4J_URI', raising=False)
    monkeypatch.delenv('NEO4J_USER', raising=False)
    monkeypatch.delenv('NEO4J_PASSWORD', raising=False)
    monkeypatch.delenv('NEO4J_DATABASE', raising=False)
    config = DatabaseConfig(
        provider='neo4j',
        providers=DatabaseProvidersConfig(
            neo4j=Neo4jProviderConfig(password=None),
        ),
    )

    with pytest.raises(ValueError, match='NEO4J_PASSWORD'):
        DatabaseDriverFactory.create_config(config)


@pytest.mark.asyncio
async def test_graphiti_service_passes_database_to_neo4j_driver(monkeypatch):
    fake_driver = object()
    fake_client = Mock()
    fake_client.build_indices_and_constraints = AsyncMock()
    driver_factory = Mock(return_value=fake_driver)
    graphiti_factory = Mock(return_value=fake_client)

    monkeypatch.setattr(graphiti_mcp_server.LLMClientFactory, 'create', Mock())
    monkeypatch.setattr(graphiti_mcp_server.EmbedderFactory, 'create', Mock())
    monkeypatch.setattr(graphiti_mcp_server.CrossEncoderFactory, 'create', Mock())
    monkeypatch.setattr(
        graphiti_mcp_server.DatabaseDriverFactory,
        'create_config',
        Mock(
            return_value={
                'uri': 'bolt://neo4j.example.test:7687',
                'user': 'neo4j',
                'password': 'password-placeholder',
                'database': 'knowledge',
            }
        ),
    )
    monkeypatch.setattr(
        'graphiti_core.driver.neo4j_driver.Neo4jDriver',
        driver_factory,
    )
    monkeypatch.setattr(graphiti_mcp_server, 'Graphiti', graphiti_factory)

    service = graphiti_mcp_server.GraphitiService(
        GraphitiConfig(database=DatabaseConfig(provider='neo4j'))
    )
    await service.initialize()

    driver_factory.assert_called_once_with(
        'bolt://neo4j.example.test:7687',
        'neo4j',
        'password-placeholder',
        database='knowledge',
    )
    assert graphiti_factory.call_args.kwargs['graph_driver'] is fake_driver
    fake_client.build_indices_and_constraints.assert_awaited_once()
