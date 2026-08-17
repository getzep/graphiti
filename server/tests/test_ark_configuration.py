from unittest.mock import AsyncMock, Mock

import pytest
from graphiti_core.embedder import LocalHashEmbedder, OpenAIEmbedder
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from graph_service import zep_graphiti
from graph_service.config import Settings
from graph_service.zep_graphiti import _create_embedder, _create_llm_client, is_llm_configured

ENVIRONMENT_NAMES = (
    'ARK_API_KEY',
    'OPENAI_API_KEY',
    'ARK_BASE_URL',
    'OPENAI_BASE_URL',
    'ARK_CHAT_MODEL',
    'OPENAI_MODEL',
    'OPENAI_MODEL_NAME',
    'MODEL_NAME',
    'ARK_TEMPERATURE',
    'OPENAI_TEMPERATURE',
    'LLM_TEMPERATURE',
    'ARK_MAX_TOKENS',
    'OPENAI_MAX_TOKENS',
    'LLM_MAX_TOKENS',
    'ARK_STRUCTURED_OUTPUT_MODE',
    'OPENAI_STRUCTURED_OUTPUT_MODE',
    'STRUCTURED_OUTPUT_MODE',
    'ARK_EMBEDDING_API_KEY',
    'OPENAI_EMBEDDING_API_KEY',
    'EMBEDDING_API_KEY',
    'ARK_EMBEDDING_BASE_URL',
    'OPENAI_EMBEDDING_BASE_URL',
    'EMBEDDING_BASE_URL',
    'ARK_EMBEDDING_MODEL',
    'OPENAI_EMBEDDING_MODEL',
    'OPENAI_EMBEDDING_MODEL_NAME',
    'EMBEDDING_MODEL_NAME',
    'ARK_EMBEDDING_DIM',
    'OPENAI_EMBEDDING_DIM',
    'EMBEDDING_DIM',
)


def _clear_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_settings_accept_ark_aliases_with_precedence(monkeypatch):
    _clear_provider_environment(monkeypatch)
    monkeypatch.setenv('ARK_API_KEY', 'ark-placeholder')
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-placeholder')
    monkeypatch.setenv('ARK_BASE_URL', 'https://ark.example.test/api/v3')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    monkeypatch.setenv('ARK_CHAT_MODEL', 'ep-placeholder')
    monkeypatch.setenv('OPENAI_MODEL', 'gpt-placeholder')
    monkeypatch.setenv('ARK_TEMPERATURE', '0.25')
    monkeypatch.setenv('ARK_MAX_TOKENS', '4096')
    monkeypatch.setenv('ARK_STRUCTURED_OUTPUT_MODE', 'prompt_only')
    monkeypatch.setenv('ARK_EMBEDDING_API_KEY', 'embedding-placeholder')
    monkeypatch.setenv('ARK_EMBEDDING_BASE_URL', 'https://embedding.example.test/api/v3')
    monkeypatch.setenv('ARK_EMBEDDING_MODEL', 'embedding-endpoint-placeholder')
    monkeypatch.setenv('EMBEDDING_DIM', '256')

    settings = Settings(_env_file=None)

    assert settings.openai_api_key == 'ark-placeholder'
    assert settings.openai_base_url == 'https://ark.example.test/api/v3'
    assert settings.model_name == 'ep-placeholder'
    assert settings.llm_temperature == 0.25
    assert settings.llm_max_tokens == 4096
    assert settings.structured_output_mode == 'prompt_only'
    assert settings.embedding_api_key == 'embedding-placeholder'
    assert settings.embedding_base_url == 'https://embedding.example.test/api/v3'
    assert settings.embedding_model_name == 'embedding-endpoint-placeholder'
    assert settings.embedding_dim == 256


def test_settings_accept_openai_and_legacy_model_aliases(monkeypatch):
    _clear_provider_environment(monkeypatch)
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-placeholder')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    monkeypatch.setenv('MODEL_NAME', 'gpt-placeholder')
    monkeypatch.setenv('OPENAI_STRUCTURED_OUTPUT_MODE', 'json_schema')

    settings = Settings(_env_file=None)

    assert settings.openai_api_key == 'openai-placeholder'
    assert settings.openai_base_url == 'https://api.openai.com/v1'
    assert settings.model_name == 'gpt-placeholder'
    assert settings.structured_output_mode == 'json_schema'


def test_embedding_dimension_only_uses_core_shared_environment_name(monkeypatch):
    _clear_provider_environment(monkeypatch)
    monkeypatch.setenv('ARK_EMBEDDING_DIM', '96')
    monkeypatch.setenv('OPENAI_EMBEDDING_DIM', '128')
    monkeypatch.setenv('EMBEDDING_DIM', '256')

    settings = Settings(_env_file=None)

    assert settings.embedding_dim == 256


def test_ark_client_uses_generic_chat_completions_and_prompt_only_mode():
    settings = Settings(
        _env_file=None,
        openai_api_key='ark-placeholder',
        openai_base_url='https://ark.example.test/api/v3',
        model_name='ep-placeholder',
        structured_output_mode='prompt_only',
    )

    client = _create_llm_client(settings)

    assert isinstance(client, OpenAIGenericClient)
    assert client.model == 'ep-placeholder'
    assert client.config.base_url == 'https://ark.example.test/api/v3'
    assert client.structured_output_mode == 'prompt_only'


def test_official_openai_endpoint_uses_openai_client():
    settings = Settings(
        _env_file=None,
        openai_api_key='openai-placeholder',
        openai_base_url='https://api.openai.com/v1',
        model_name='gpt-placeholder',
    )

    client = _create_llm_client(settings)

    assert isinstance(client, OpenAIClient)
    assert not isinstance(client, OpenAIGenericClient)


@pytest.mark.parametrize(
    ('settings', 'expected'),
    [
        (Settings(_env_file=None, openai_api_key='', model_name='ep-placeholder'), False),
        (Settings(_env_file=None, openai_api_key='placeholder', model_name=''), False),
        (
            Settings(
                _env_file=None,
                openai_api_key='placeholder',
                model_name='ep-placeholder',
                openai_base_url='https://api.openai.com/v1',
            ),
            False,
        ),
        (
            Settings(
                _env_file=None,
                openai_api_key='placeholder',
                model_name='ep-placeholder',
                openai_base_url='not-an-absolute-url',
            ),
            False,
        ),
        (
            Settings(
                _env_file=None,
                openai_api_key='placeholder',
                model_name='ep-placeholder',
                openai_base_url='https://ark.example.test/api/v3',
            ),
            True,
        ),
        (
            Settings(
                _env_file=None,
                openai_api_key='placeholder',
                model_name='gpt-placeholder',
            ),
            True,
        ),
    ],
)
def test_llm_readiness_requires_complete_provider_configuration(settings, expected):
    assert is_llm_configured(settings) is expected


@pytest.mark.parametrize(
    ('settings', 'message'),
    [
        (
            Settings(
                _env_file=None,
                openai_api_key='',
                model_name='ep-placeholder',
            ),
            'ARK_API_KEY',
        ),
        (
            Settings(
                _env_file=None,
                openai_api_key='ark-placeholder',
                openai_base_url='https://ark.example.test/api/v3',
                model_name='',
            ),
            'ARK_CHAT_MODEL',
        ),
        (
            Settings(
                _env_file=None,
                openai_api_key='ark-placeholder',
                openai_base_url='https://api.openai.com/v1',
                model_name='ep-placeholder',
            ),
            'ARK_BASE_URL',
        ),
    ],
)
def test_llm_configuration_fails_fast(settings, message):
    with pytest.raises(ValueError, match=message):
        _create_llm_client(settings)


def test_local_hash_embedder_needs_no_embedding_api_key():
    settings = Settings(_env_file=None, embedding_provider='local_hash')

    embedder = _create_embedder(settings)

    assert isinstance(embedder, LocalHashEmbedder)
    assert embedder.config.embedding_dim == settings.embedding_dim


def test_openai_compatible_embedder_uses_separate_configuration():
    settings = Settings(
        _env_file=None,
        embedding_provider='openai',
        embedding_api_key='embedding-placeholder',
        embedding_base_url='https://embedding.example.test/api/v3',
        embedding_model_name='embedding-endpoint-placeholder',
    )

    embedder = _create_embedder(settings)

    assert isinstance(embedder, OpenAIEmbedder)
    assert embedder.config.api_key == 'embedding-placeholder'
    assert embedder.config.base_url == 'https://embedding.example.test/api/v3'
    assert embedder.config.embedding_model == 'embedding-endpoint-placeholder'


def test_openai_compatible_embedder_requires_a_model():
    settings = Settings(
        _env_file=None,
        embedding_provider='openai',
        embedding_api_key='embedding-placeholder',
    )

    with pytest.raises(ValueError, match='ARK_EMBEDDING_MODEL'):
        _create_embedder(settings)


def test_neo4j_driver_receives_explicit_database(monkeypatch):
    fake_llm = object()
    fake_embedder = object()
    fake_driver = object()
    fake_graphiti = object()
    driver_factory = Mock(return_value=fake_driver)
    graphiti_factory = Mock(return_value=fake_graphiti)

    monkeypatch.setattr(zep_graphiti, '_create_llm_client', Mock(return_value=fake_llm))
    monkeypatch.setattr(zep_graphiti, '_create_embedder', Mock(return_value=fake_embedder))
    monkeypatch.setattr(
        'graphiti_core.driver.neo4j_driver.Neo4jDriver',
        driver_factory,
    )
    monkeypatch.setattr(zep_graphiti, 'ZepGraphiti', graphiti_factory)

    settings = Settings(
        _env_file=None,
        db_backend='neo4j',
        neo4j_uri='bolt://neo4j.example.test:7687',
        neo4j_user='neo4j',
        neo4j_password='password-placeholder',
        neo4j_database='knowledge',
    )

    result = zep_graphiti.create_graphiti_client(settings)

    assert result is fake_graphiti
    driver_factory.assert_called_once_with(
        'bolt://neo4j.example.test:7687',
        'neo4j',
        'password-placeholder',
        database='knowledge',
    )
    assert graphiti_factory.call_args.kwargs['graph_driver'] is fake_driver
    assert graphiti_factory.call_args.kwargs['llm_client'] is fake_llm
    assert graphiti_factory.call_args.kwargs['embedder'] is fake_embedder


@pytest.mark.asyncio
async def test_database_initialization_does_not_require_llm_configuration(monkeypatch):
    fake_driver = Mock()
    fake_driver.build_indices_and_constraints = AsyncMock()
    fake_driver.close = AsyncMock()
    monkeypatch.setattr(
        zep_graphiti,
        '_create_graph_driver',
        Mock(return_value=fake_driver),
    )

    await zep_graphiti.initialize_graphiti(Settings(_env_file=None))

    fake_driver.build_indices_and_constraints.assert_awaited_once()
    fake_driver.close.assert_awaited_once()
