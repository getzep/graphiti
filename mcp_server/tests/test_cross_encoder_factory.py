#!/usr/bin/env python3
"""Unit tests for CrossEncoderFactory reranker selection."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

# Add the src directory to the path (mirrors the other factory tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.cross_encoder.lexical_reranker_client import LexicalRerankerClient
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

import graphiti_mcp_server
from config.schema import (
    AnthropicProviderConfig,
    DatabaseConfig,
    EmbedderConfig,
    EmbedderProvidersConfig,
    GeminiProviderConfig,
    GraphitiConfig,
    LLMConfig,
    LLMProvidersConfig,
    OpenAIProviderConfig,
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
        reranker = CrossEncoderFactory.create(llm, embedder)

        assert isinstance(reranker, OpenAIRerankerClient)
        assert reranker.config.model == llm.model

    def test_openai_embedder_does_not_become_reranker_chat_model(self):
        llm = LLMConfig(
            provider='anthropic',
            providers=LLMProvidersConfig(anthropic=AnthropicProviderConfig(api_key='test-key')),
        )
        embedder = EmbedderConfig(
            provider='openai',
            model='text-embedding-placeholder',
            providers=EmbedderProvidersConfig(openai=OpenAIProviderConfig(api_key='test-key')),
        )

        reranker = CrossEncoderFactory.create(llm, embedder)

        assert isinstance(reranker, OpenAIRerankerClient)
        assert reranker.config.model is None

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

    def test_non_native_providers_use_dependency_free_lexical_reranker(self):
        llm = LLMConfig(
            provider='anthropic',
            providers=LLMProvidersConfig(anthropic=AnthropicProviderConfig(api_key='test-key')),
        )
        embedder = EmbedderConfig(
            provider='voyage',
            providers=EmbedderProvidersConfig(voyage=VoyageProviderConfig(api_key='test-key')),
        )
        assert isinstance(CrossEncoderFactory.create(llm, embedder), LexicalRerankerClient)

    def test_openai_compatible_provider_uses_lexical_reranker(self):
        llm = LLMConfig(
            provider='openai',
            model='ep-test',
            providers=LLMProvidersConfig(
                openai=OpenAIProviderConfig(
                    api_key='test-key', api_url='https://ark.example.com/api/v3'
                )
            ),
        )
        embedder = EmbedderConfig(provider='local_hash', dimensions=64)

        assert isinstance(CrossEncoderFactory.create(llm, embedder), LexicalRerankerClient)


@pytest.mark.asyncio
async def test_graphiti_service_does_not_swallow_reranker_configuration_error(monkeypatch):
    error = ValueError('reranker setup failed')

    def fail_reranker_setup(*_args):
        raise error

    fake_client = Mock()
    fake_client.build_indices_and_constraints = AsyncMock()
    monkeypatch.setattr(graphiti_mcp_server.LLMClientFactory, 'create', Mock())
    monkeypatch.setattr(graphiti_mcp_server.EmbedderFactory, 'create', Mock())
    monkeypatch.setattr(CrossEncoderFactory, 'create', fail_reranker_setup)
    monkeypatch.setattr(graphiti_mcp_server, 'Graphiti', Mock(return_value=fake_client))
    service = graphiti_mcp_server.GraphitiService(
        GraphitiConfig(database=DatabaseConfig(provider='neo4j'))
    )

    with pytest.raises(ValueError, match='reranker setup failed'):
        await service.initialize()


@pytest.mark.asyncio
async def test_graphiti_service_does_not_swallow_llm_configuration_error(monkeypatch):
    error = ValueError('llm setup failed')

    def fail_llm_setup(*_args):
        raise error

    monkeypatch.setattr(graphiti_mcp_server.LLMClientFactory, 'create', fail_llm_setup)
    service = graphiti_mcp_server.GraphitiService(GraphitiConfig())

    with pytest.raises(ValueError, match='llm setup failed'):
        await service.initialize()


@pytest.mark.asyncio
async def test_graphiti_service_does_not_swallow_embedder_configuration_error(monkeypatch):
    error = ValueError('embedder setup failed')

    def fail_embedder_setup(*_args):
        raise error

    monkeypatch.setattr(graphiti_mcp_server.LLMClientFactory, 'create', Mock())
    monkeypatch.setattr(graphiti_mcp_server.EmbedderFactory, 'create', fail_embedder_setup)
    service = graphiti_mcp_server.GraphitiService(GraphitiConfig())

    with pytest.raises(ValueError, match='embedder setup failed'):
        await service.initialize()
