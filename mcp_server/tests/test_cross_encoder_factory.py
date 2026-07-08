#!/usr/bin/env python3
"""Unit tests for CrossEncoderFactory reranker selection."""

import sys
from pathlib import Path

# Add the src directory to the path (mirrors the other factory tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

from config.schema import (
    AnthropicProviderConfig,
    EmbedderConfig,
    EmbedderProvidersConfig,
    GeminiProviderConfig,
    LLMConfig,
    LLMProvidersConfig,
    OpenAIProviderConfig,
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
