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
        config = GraphitiConfig(reranker={'provider': 'gemini'})

        assert config.model_dump().get('reranker') == {'provider': 'gemini'}

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
