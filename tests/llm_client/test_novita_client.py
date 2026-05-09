"""
Tests for NovitaClient.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.novita_client import DEFAULT_BASE_URL, DEFAULT_MODEL, NovitaClient


class DummyChatCompletions:
    def __init__(self):
        self.create_calls: list[dict] = []

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        message = SimpleNamespace(content='{"result": "success"}')
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class DummyChat:
    def __init__(self):
        self.completions = DummyChatCompletions()


class DummyOpenAIClient:
    def __init__(self):
        self.chat = DummyChat()


@pytest.mark.asyncio
async def test_novita_client_uses_default_config():
    """Test that NovitaClient uses default base URL and model when not specified."""
    with patch.dict(os.environ, {'NOVITA_API_KEY': 'test-key'}):
        client = NovitaClient()

        assert client.config.base_url == DEFAULT_BASE_URL
        assert client.config.model == DEFAULT_MODEL
        assert client.config.api_key == 'test-key'


@pytest.mark.asyncio
async def test_novita_client_uses_custom_config():
    """Test that NovitaClient respects custom configuration."""
    custom_config = LLMConfig(
        api_key='custom-key',
        model='zai-org/glm-5',
        base_url='https://custom.novita.api/openai',
    )
    client = NovitaClient(config=custom_config)

    assert client.config.base_url == 'https://custom.novita.api/openai'
    assert client.config.model == 'zai-org/glm-5'
    assert client.config.api_key == 'custom-key'


@pytest.mark.asyncio
async def test_novita_client_generate_response():
    """Test that NovitaClient can generate responses via OpenAI-compatible API."""
    dummy_client = DummyOpenAIClient()

    with patch.dict(os.environ, {'NOVITA_API_KEY': 'test-key'}):
        client = NovitaClient()
        client.client = dummy_client

        from graphiti_core.prompts.models import Message

        messages = [
            Message(role='system', content='You are a helpful assistant.'),
            Message(role='user', content='Hello!'),
        ]

        result = await client._generate_response(messages=messages)

        assert result == {'result': 'success'}
        assert len(dummy_client.chat.completions.create_calls) == 1

        call_args = dummy_client.chat.completions.create_calls[0]
        assert call_args['model'] == DEFAULT_MODEL
        assert 'response_format' in call_args


@pytest.mark.asyncio
async def test_novita_client_provider_type():
    """Test that _get_provider_type returns 'novita'."""
    with patch.dict(os.environ, {'NOVITA_API_KEY': 'test-key'}):
        client = NovitaClient()
        assert client._get_provider_type() == 'novita'
