#!/usr/bin/env python3
"""Unit tests for service factory provider detection and client routing."""

import sys
from pathlib import Path

import pytest

# Add the src directory to the path (mirrors the other factory tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from config.schema import LLMConfig, LLMProvidersConfig, OpenAIProviderConfig
from services.factories import LLMClientFactory, is_non_openai_provider


class TestIsNonOpenAIProvider:
    """Tests for the base_url-based provider detection."""

    @pytest.mark.parametrize(
        'base_url',
        [
            None,
            '',
            'https://api.openai.com/v1',
            'https://api.openai.com',
            'https://my-resource.openai.azure.com',
        ],
    )
    def test_official_or_unset_is_openai(self, base_url):
        """Unset, empty, or official OpenAI/Azure endpoints are treated as OpenAI."""
        assert is_non_openai_provider(base_url) is False

    @pytest.mark.parametrize(
        'base_url',
        [
            'http://localhost:11434/v1',  # Ollama
            'http://localhost:1234/v1',  # LM Studio
            'http://localhost:8000/v1',  # vLLM
            'https://my-proxy.internal/v1',
        ],
    )
    def test_compatible_providers_are_non_openai(self, base_url):
        """OpenAI-compatible third-party endpoints are detected as non-OpenAI."""
        assert is_non_openai_provider(base_url) is True


class TestLLMClientFactoryRouting:
    """Tests that the factory selects the right client based on base_url."""

    @staticmethod
    def _config(api_url: str) -> LLMConfig:
        return LLMConfig(
            provider='openai',
            model='gpt-4o-mini',
            providers=LLMProvidersConfig(
                openai=OpenAIProviderConfig(api_key='test-key', api_url=api_url)
            ),
        )

    def test_official_openai_uses_openai_client(self):
        client = LLMClientFactory.create(self._config('https://api.openai.com/v1'))
        assert isinstance(client, OpenAIClient)
        assert not isinstance(client, OpenAIGenericClient)

    def test_ollama_uses_generic_client(self):
        client = LLMClientFactory.create(self._config('http://localhost:11434/v1'))
        assert isinstance(client, OpenAIGenericClient)
