#!/usr/bin/env python3
"""Unit tests for CrossEncoderFactory reranker selection."""

import builtins
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

# Add the src directory to the path (mirrors the other factory tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

import graphiti_mcp_server
from config.schema import (
    AnthropicProviderConfig,
    AzureOpenAIProviderConfig,
    DatabaseConfig,
    EmbedderConfig,
    EmbedderProvidersConfig,
    GeminiProviderConfig,
    GraphitiConfig,
    LLMConfig,
    LLMProvidersConfig,
    OpenAIProviderConfig,
    RerankerConfig,
    VoyageProviderConfig,
)
from services.factories import CrossEncoderFactory


class TestCrossEncoderFactory:
    """The reranker is inferred from the providers, so a non-OpenAI setup does not need OPENAI_API_KEY."""

    def test_openai_llm_uses_openai_reranker(self):
        llm = LLMConfig(
            provider='openai',
            providers=LLMProvidersConfig(openai=OpenAIProviderConfig(api_key='test-key')),
        )
        embedder = EmbedderConfig(
            provider='openai',
            providers=EmbedderProvidersConfig(openai=OpenAIProviderConfig(api_key='test-key')),
        )
        assert isinstance(CrossEncoderFactory.create(llm, embedder), OpenAIRerankerClient)

    def test_anthropic_llm_falls_back_to_gemini_embedder(self):
        # Anthropic has no native reranker, so the factory should pick up the Gemini embedder's
        # key instead of defaulting to OpenAIRerankerClient (which would need OPENAI_API_KEY).
        llm = LLMConfig(
            provider='anthropic',
            providers=LLMProvidersConfig(anthropic=AnthropicProviderConfig(api_key='test-key')),
        )
        embedder = EmbedderConfig(
            provider='gemini',
            providers=EmbedderProvidersConfig(gemini=GeminiProviderConfig(api_key='test-key')),
        )
        assert isinstance(CrossEncoderFactory.create(llm, embedder), GeminiRerankerClient)

    def test_graphiti_config_accepts_explicit_reranker_provider(self):
        config = GraphitiConfig(reranker={'provider': 'gemini', 'model': 'gemini-2.5-flash'})

        assert config.model_dump().get('reranker') == {
            'provider': 'gemini',
            'model': 'gemini-2.5-flash',
        }

    def test_explicit_gemini_reranker_overrides_provider_inference(self):
        llm = LLMConfig(
            provider='openai',
            providers=LLMProvidersConfig(
                openai=OpenAIProviderConfig(api_key='openai-key'),
                gemini=GeminiProviderConfig(api_key='gemini-key'),
            ),
        )
        embedder = EmbedderConfig(
            provider='openai',
            providers=EmbedderProvidersConfig(openai=OpenAIProviderConfig(api_key='openai-key')),
        )

        reranker = CrossEncoderFactory.create(llm, embedder, RerankerConfig(provider='gemini'))

        assert isinstance(reranker, GeminiRerankerClient)

    def test_explicit_reranker_model_is_passed_to_provider_client(self):
        llm = LLMConfig(
            provider='openai',
            providers=LLMProvidersConfig(
                openai=OpenAIProviderConfig(api_key='openai-key'),
                gemini=GeminiProviderConfig(api_key='gemini-key'),
            ),
        )
        embedder = EmbedderConfig(
            provider='openai',
            providers=EmbedderProvidersConfig(openai=OpenAIProviderConfig(api_key='openai-key')),
        )

        reranker = CrossEncoderFactory.create(
            llm,
            embedder,
            RerankerConfig(provider='gemini', model='gemini-2.5-flash'),
        )

        assert isinstance(reranker, GeminiRerankerClient)
        assert reranker.config.model == 'gemini-2.5-flash'

    def test_reranker_model_applies_when_provider_is_inferred(self):
        llm = LLMConfig(
            provider='openai',
            providers=LLMProvidersConfig(openai=OpenAIProviderConfig(api_key='openai-key')),
        )
        embedder = EmbedderConfig(
            provider='openai',
            providers=EmbedderProvidersConfig(openai=OpenAIProviderConfig(api_key='openai-key')),
        )

        reranker = CrossEncoderFactory.create(
            llm,
            embedder,
            RerankerConfig(model='gpt-4.1-mini'),
        )

        assert isinstance(reranker, OpenAIRerankerClient)
        assert reranker.config.model == 'gpt-4.1-mini'

    @pytest.mark.parametrize(
        ('provider', 'provider_config', 'client_type'),
        [
            ('openai', OpenAIProviderConfig, OpenAIRerankerClient),
            ('gemini', GeminiProviderConfig, GeminiRerankerClient),
        ],
    )
    def test_explicit_reranker_skips_keyless_llm_entry(
        self, monkeypatch, provider, provider_config, client_type
    ):
        # An unset ${VAR} leaves a provider entry with api_key=None. That entry must not hide a
        # configured embedder entry for the same provider.
        for name in ('OPENAI_API_KEY', 'GOOGLE_API_KEY', 'GEMINI_API_KEY'):
            monkeypatch.delenv(name, raising=False)
        llm = LLMConfig(
            provider='anthropic',
            providers=LLMProvidersConfig(**{provider: provider_config(api_key=None)}),
        )
        embedder = EmbedderConfig(
            provider=provider,
            providers=EmbedderProvidersConfig(
                **{provider: provider_config(api_key='embedder-key')}
            ),
        )

        reranker = CrossEncoderFactory.create(llm, embedder, RerankerConfig(provider=provider))

        assert isinstance(reranker, client_type)
        assert reranker.config.api_key == 'embedder-key'

    def test_explicit_reranker_prefers_llm_entry_when_both_have_keys(self):
        llm = LLMConfig(
            provider='anthropic',
            providers=LLMProvidersConfig(gemini=GeminiProviderConfig(api_key='llm-key')),
        )
        embedder = EmbedderConfig(
            provider='gemini',
            providers=EmbedderProvidersConfig(gemini=GeminiProviderConfig(api_key='embedder-key')),
        )

        reranker = CrossEncoderFactory.create(llm, embedder, RerankerConfig(provider='gemini'))

        assert reranker.config.api_key == 'llm-key'

    def test_explicit_reranker_keyless_entry_can_use_ambient_key(self, monkeypatch):
        monkeypatch.setenv('OPENAI_API_KEY', 'ambient-key')
        llm = LLMConfig(
            provider='anthropic',
            providers=LLMProvidersConfig(openai=OpenAIProviderConfig(api_key=None)),
        )
        embedder = EmbedderConfig(
            provider='voyage',
            providers=EmbedderProvidersConfig(voyage=VoyageProviderConfig(api_key='test-key')),
        )

        reranker = CrossEncoderFactory.create(llm, embedder, RerankerConfig(provider='openai'))

        assert isinstance(reranker, OpenAIRerankerClient)
        assert reranker.client.api_key == 'ambient-key'

    def test_explicit_reranker_without_matching_entry_fails_startup(self):
        llm = LLMConfig(
            provider='openai',
            providers=LLMProvidersConfig(openai=OpenAIProviderConfig(api_key='openai-key')),
        )
        embedder = EmbedderConfig(
            provider='openai',
            providers=EmbedderProvidersConfig(openai=OpenAIProviderConfig(api_key='openai-key')),
        )

        with pytest.raises(ValueError, match="Reranker provider 'gemini' is not configured"):
            CrossEncoderFactory.create(llm, embedder, RerankerConfig(provider='gemini'))

    def test_explicit_bge_reranker_skips_api_providers(self, monkeypatch):
        local_reranker = Mock()
        monkeypatch.setattr(
            CrossEncoderFactory, '_local_reranker', staticmethod(lambda _logger: local_reranker)
        )
        llm = LLMConfig(
            provider='openai',
            providers=LLMProvidersConfig(openai=OpenAIProviderConfig(api_key='openai-key')),
        )
        embedder = EmbedderConfig(
            provider='openai',
            providers=EmbedderProvidersConfig(openai=OpenAIProviderConfig(api_key='openai-key')),
        )

        reranker = CrossEncoderFactory.create(llm, embedder, RerankerConfig(provider='bge'))

        assert reranker is local_reranker

    def test_explicit_azure_reranker_uses_v1_endpoint_and_model(self):
        llm = LLMConfig(
            provider='openai',
            providers=LLMProvidersConfig(
                openai=OpenAIProviderConfig(api_key='openai-key'),
                azure_openai=AzureOpenAIProviderConfig(
                    api_key='azure-key', api_url='https://example.openai.azure.com'
                ),
            ),
        )
        embedder = EmbedderConfig(
            provider='openai',
            providers=EmbedderProvidersConfig(openai=OpenAIProviderConfig(api_key='openai-key')),
        )

        reranker = CrossEncoderFactory.create(
            llm,
            embedder,
            RerankerConfig(provider='azure_openai', model='reranker-deployment'),
        )

        assert isinstance(reranker, OpenAIRerankerClient)
        assert str(reranker.client.base_url) == 'https://example.openai.azure.com/openai/v1/'
        assert reranker.config.model == 'reranker-deployment'

    def test_missing_local_reranker_dependency_is_actionable(self, monkeypatch, caplog):
        llm = LLMConfig(
            provider='anthropic',
            providers=LLMProvidersConfig(anthropic=AnthropicProviderConfig(api_key='test-key')),
        )
        embedder = EmbedderConfig(
            provider='voyage',
            providers=EmbedderProvidersConfig(voyage=VoyageProviderConfig(api_key='test-key')),
        )
        real_import = builtins.__import__

        def import_without_bge(name, *args, **kwargs):
            if name == 'graphiti_core.cross_encoder.bge_reranker_client':
                raise ImportError('sentence-transformers is not installed')
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', import_without_bge)
        caplog.set_level(logging.INFO)

        with pytest.raises(ValueError, match="MCP server's 'providers' extra"):
            CrossEncoderFactory.create(llm, embedder)

        assert '~2.3 GB' in caplog.text


@pytest.mark.parametrize(
    'config_file',
    sorted((Path(__file__).parent.parent / 'config').glob('config*.yaml')),
    ids=lambda path: path.name,
)
def test_shipped_configs_read_reranker_settings_from_env(monkeypatch, config_file):
    # The Docker images load the config-docker-*.yaml files, not config.yaml, so each shipped
    # config must expose the documented RERANKER_* and *_RERANKER variables.
    for name in (
        'RERANKER__PROVIDER',
        'RERANKER__MODEL',
        'GRAPHITI__FACT_RERANKER',
        'GRAPHITI__NODE_RERANKER',
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('CONFIG_PATH', str(config_file))
    monkeypatch.setenv('RERANKER_PROVIDER', 'gemini')
    monkeypatch.setenv('RERANKER_MODEL', 'gemini-2.5-flash')
    monkeypatch.setenv('FACT_RERANKER', 'cross_encoder')
    monkeypatch.setenv('NODE_RERANKER', 'cross_encoder')

    config = GraphitiConfig()

    assert config.reranker.provider == 'gemini'
    assert config.reranker.model == 'gemini-2.5-flash'
    assert config.graphiti.fact_reranker == 'cross_encoder'
    assert config.graphiti.node_reranker == 'cross_encoder'


@pytest.mark.asyncio
async def test_graphiti_service_does_not_swallow_reranker_configuration_error(monkeypatch):
    error = ValueError('reranker setup failed')

    def fail_reranker_setup(*_args):
        raise error

    fake_client = Mock()
    fake_client.build_indices_and_constraints = AsyncMock()
    monkeypatch.setattr(CrossEncoderFactory, 'create', fail_reranker_setup)
    monkeypatch.setattr(graphiti_mcp_server, 'Graphiti', Mock(return_value=fake_client))
    service = graphiti_mcp_server.GraphitiService(
        GraphitiConfig(database=DatabaseConfig(provider='neo4j'))
    )

    with pytest.raises(ValueError, match='reranker setup failed'):
        await service.initialize()


@pytest.mark.asyncio
async def test_graphiti_service_uses_explicit_reranker_config(monkeypatch):
    constructed = {}
    fake_client = Mock()
    fake_client.build_indices_and_constraints = AsyncMock()

    def capture_graphiti(**kwargs):
        constructed.update(kwargs)
        return fake_client

    monkeypatch.setattr(graphiti_mcp_server, 'Graphiti', capture_graphiti)
    config = GraphitiConfig(
        llm=LLMConfig(
            provider='openai',
            providers=LLMProvidersConfig(
                openai=OpenAIProviderConfig(api_key='openai-key'),
                gemini=GeminiProviderConfig(api_key='gemini-key'),
            ),
        ),
        embedder=EmbedderConfig(
            provider='openai',
            providers=EmbedderProvidersConfig(openai=OpenAIProviderConfig(api_key='openai-key')),
        ),
        reranker=RerankerConfig(provider='gemini'),
        database=DatabaseConfig(provider='neo4j'),
    )

    await graphiti_mcp_server.GraphitiService(config).initialize()

    assert isinstance(constructed['cross_encoder'], GeminiRerankerClient)
