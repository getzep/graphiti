"""
Tests for OpenAI LLM client.
"""

from unittest.mock import patch

from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient


class TestOpenAIClientInitialization:
    """Tests for OpenAIClient initialization and configuration."""

    def test_init_with_config_custom_max_tokens(self):
        """Test that OpenAIClient respects custom max_tokens from LLMConfig."""
        custom_max_tokens = 16000
        config = LLMConfig(
            api_key='test_api_key', model='gpt-4', temperature=0.5, max_tokens=custom_max_tokens
        )

        # Mock the AsyncOpenAI client to avoid actual API calls
        with patch('graphiti_core.llm_client.openai_client.AsyncOpenAI'):
            client = OpenAIClient(config=config, cache=False)

        assert client.config == config
        assert client.model == 'gpt-4'
        assert client.temperature == 0.5
        assert client.max_tokens == custom_max_tokens, (
            f'Expected max_tokens to be {custom_max_tokens} from config, '
            f'but got {client.max_tokens}'
        )

    def test_init_with_default_max_tokens_uses_constructor_param(self):
        """Test that constructor max_tokens parameter is used when config has default value."""
        constructor_max_tokens = 4000
        config = LLMConfig(
            api_key='test_api_key',
            model='gpt-4',
            # max_tokens will be DEFAULT_MAX_TOKENS (8192)
        )

        with patch('graphiti_core.llm_client.openai_client.AsyncOpenAI'):
            client = OpenAIClient(config=config, cache=False, max_tokens=constructor_max_tokens)

        assert client.max_tokens == constructor_max_tokens, (
            f'Expected max_tokens to be {constructor_max_tokens} from constructor, '
            f'but got {client.max_tokens}'
        )

    def test_init_config_max_tokens_takes_precedence_over_constructor(self):
        """Test that config max_tokens takes precedence over constructor parameter."""
        config_max_tokens = 20000
        constructor_max_tokens = 4000

        config = LLMConfig(api_key='test_api_key', model='gpt-4', max_tokens=config_max_tokens)

        with patch('graphiti_core.llm_client.openai_client.AsyncOpenAI'):
            client = OpenAIClient(config=config, cache=False, max_tokens=constructor_max_tokens)

        assert client.max_tokens == config_max_tokens, (
            f'Expected config max_tokens ({config_max_tokens}) to take precedence over '
            f'constructor max_tokens ({constructor_max_tokens}), but got {client.max_tokens}'
        )

    def test_init_with_none_config_uses_constructor_max_tokens(self):
        """Test that constructor max_tokens is used when config is None."""
        constructor_max_tokens = 12000

        with patch('graphiti_core.llm_client.openai_client.AsyncOpenAI'):
            client = OpenAIClient(config=None, cache=False, max_tokens=constructor_max_tokens)

        assert client.max_tokens == constructor_max_tokens
        assert client.config.max_tokens == DEFAULT_MAX_TOKENS  # Config gets default
