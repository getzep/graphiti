"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Running tests: pytest -xvs tests/llm_client/test_anthropic_client.py

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.errors import RateLimitError, RefusalError
from graphiti_core.prompts.models import Message


# Rename class to avoid pytest collection as a test class
class ResponseModel(BaseModel):
    """Test model for response testing."""

    test_field: str
    optional_field: int = 0


def _make_usage(
    input_tokens=100,
    output_tokens=50,
    cache_creation_input_tokens=0,
    cache_read_input_tokens=0,
):
    """Create a mock usage object with the expected Anthropic response fields."""
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_creation_input_tokens = cache_creation_input_tokens
    usage.cache_read_input_tokens = cache_read_input_tokens
    return usage


def _make_response(content_items, usage=None):
    """Create a mock Anthropic API response."""
    resp = MagicMock()
    resp.content = content_items
    resp.usage = usage or _make_usage()
    return resp


@pytest.fixture
def mock_async_anthropic():
    """Fixture to mock the AsyncAnthropic client."""
    with patch('anthropic.AsyncAnthropic') as mock_client:
        # Setup mock instance and its create method
        mock_instance = mock_client.return_value
        mock_instance.messages.create = AsyncMock()
        yield mock_instance


@pytest.fixture
def anthropic_client(mock_async_anthropic):
    """Fixture to create an AnthropicClient with a mocked AsyncAnthropic."""
    # Use a context manager to patch the AsyncAnthropic constructor to avoid
    # the client actually trying to create a real connection
    with patch('anthropic.AsyncAnthropic', return_value=mock_async_anthropic):
        config = LLMConfig(
            api_key='test_api_key', model='test-model', temperature=0.5, max_tokens=1000
        )
        client = AnthropicClient(config=config, cache=False)
        # Replace the client's client with our mock to ensure we're using the mock
        client.client = mock_async_anthropic
        return client


class TestAnthropicClientInitialization:
    """Tests for AnthropicClient initialization."""

    def test_init_with_config(self):
        """Test initialization with a config object."""
        config = LLMConfig(
            api_key='test_api_key', model='test-model', temperature=0.5, max_tokens=1000
        )
        client = AnthropicClient(config=config, cache=False)

        assert client.config == config
        assert client.model == 'test-model'
        assert client.temperature == 0.5
        assert client.max_tokens == 1000

    def test_init_with_default_model(self):
        """Test initialization with default model when none is provided."""
        config = LLMConfig(api_key='test_api_key')
        client = AnthropicClient(config=config, cache=False)

        assert client.model == 'claude-sonnet-4-5-latest'

    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'env_api_key'})
    def test_init_without_config(self):
        """Test initialization without a config, using environment variable."""
        client = AnthropicClient(cache=False)

        assert client.config.api_key == 'env_api_key'
        assert client.model == 'claude-sonnet-4-5-latest'

    def test_init_with_custom_client(self):
        """Test initialization with a custom AsyncAnthropic client."""
        mock_client = MagicMock()
        client = AnthropicClient(client=mock_client)

        assert client.client == mock_client


class TestAnthropicClientModelSizeRouting:
    """Tests for model size routing."""

    def test_get_model_for_size_small(self):
        """Test that small size returns the small model."""
        config = LLMConfig(api_key='test_api_key')
        client = AnthropicClient(config=config, cache=False)

        result = client._get_model_for_size(ModelSize.small)
        assert result == 'claude-haiku-4-5-latest'

    def test_get_model_for_size_medium(self):
        """Test that medium size returns the default (larger) model."""
        config = LLMConfig(api_key='test_api_key')
        client = AnthropicClient(config=config, cache=False)

        result = client._get_model_for_size(ModelSize.medium)
        assert result == 'claude-sonnet-4-5-latest'

    def test_small_model_default_on_init(self):
        """Test that small_model is correctly defaulted during initialization."""
        config = LLMConfig(api_key='test_api_key')
        client = AnthropicClient(config=config, cache=False)

        assert client.small_model == 'claude-haiku-4-5-latest'

    def test_custom_small_model(self):
        """Test that a custom small_model is respected."""
        config = LLMConfig(api_key='test_api_key', small_model='claude-3-haiku-20240307')
        client = AnthropicClient(config=config, cache=False)

        result = client._get_model_for_size(ModelSize.small)
        assert result == 'claude-3-haiku-20240307'

    def test_custom_model_for_medium(self):
        """Test that a custom model is used for medium size."""
        config = LLMConfig(api_key='test_api_key', model='claude-3-5-sonnet-latest')
        client = AnthropicClient(config=config, cache=False)

        result = client._get_model_for_size(ModelSize.medium)
        assert result == 'claude-3-5-sonnet-latest'


class TestAnthropicClientGenerateResponse:
    """Tests for AnthropicClient generate_response method."""

    @pytest.mark.asyncio
    async def test_generate_response_with_tool_use(self, anthropic_client, mock_async_anthropic):
        """Test successful response generation with tool use."""
        content_item = MagicMock()
        content_item.type = 'tool_use'
        content_item.input = {'test_field': 'test_value'}

        mock_async_anthropic.messages.create.return_value = _make_response([content_item])

        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='User message'),
        ]
        result = await anthropic_client.generate_response(
            messages=messages, response_model=ResponseModel
        )

        assert isinstance(result, dict)
        assert result['test_field'] == 'test_value'
        mock_async_anthropic.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_with_text_response(
        self, anthropic_client, mock_async_anthropic
    ):
        """Test response generation when getting text response instead of tool use."""
        content_item = MagicMock()
        content_item.type = 'text'
        content_item.text = '{"test_field": "extracted_value"}'

        mock_async_anthropic.messages.create.return_value = _make_response([content_item])

        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='User message'),
        ]
        result = await anthropic_client.generate_response(
            messages=messages, response_model=ResponseModel
        )

        assert isinstance(result, dict)
        assert result['test_field'] == 'extracted_value'

    @pytest.mark.asyncio
    async def test_auto_caching_enabled(self, anthropic_client, mock_async_anthropic):
        """Test that top-level cache_control is passed for auto caching."""
        content_item = MagicMock()
        content_item.type = 'tool_use'
        content_item.input = {'test_field': 'value'}

        mock_async_anthropic.messages.create.return_value = _make_response([content_item])

        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='User message'),
        ]
        await anthropic_client.generate_response(messages=messages, response_model=ResponseModel)

        call_kwargs = mock_async_anthropic.messages.create.call_args
        # Top-level cache_control should be passed for auto caching
        cache_control_arg = call_kwargs.kwargs.get('cache_control')
        assert cache_control_arg == {'type': 'ephemeral'}

        # System message should be a plain string (not structured content blocks)
        system_arg = call_kwargs.kwargs.get('system')
        assert isinstance(system_arg, str)
        assert system_arg == 'System message'

        # Tools should NOT have block-level cache_control
        tools_arg = call_kwargs.kwargs.get('tools')
        assert 'cache_control' not in tools_arg[-1]

    @pytest.mark.asyncio
    async def test_cache_tokens_tracked(self, anthropic_client, mock_async_anthropic):
        """Test that cache creation and read tokens are tracked."""
        content_item = MagicMock()
        content_item.type = 'tool_use'
        content_item.input = {'test_field': 'value'}

        usage = _make_usage(
            input_tokens=50,
            output_tokens=30,
            cache_creation_input_tokens=500,
            cache_read_input_tokens=0,
        )
        mock_async_anthropic.messages.create.return_value = _make_response(
            [content_item], usage=usage
        )

        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='User message'),
        ]
        await anthropic_client.generate_response(
            messages=messages,
            response_model=ResponseModel,
            prompt_name='test_prompt',
        )

        tracker_usage = anthropic_client.token_tracker.get_usage()
        assert 'test_prompt' in tracker_usage
        assert tracker_usage['test_prompt'].total_cache_creation_tokens == 500
        assert tracker_usage['test_prompt'].total_cache_read_tokens == 0

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, anthropic_client, mock_async_anthropic):
        """Test handling of rate limit errors."""

        # Create a custom RateLimitError from Anthropic
        class MockRateLimitError(Exception):
            pass

        # Patch the Anthropic error with our mock to avoid constructor issues
        with patch('anthropic.RateLimitError', MockRateLimitError):
            # Setup mock to raise our mocked RateLimitError
            mock_async_anthropic.messages.create.side_effect = MockRateLimitError(
                'Rate limit exceeded'
            )

            # Call method and check exception
            messages = [Message(role='user', content='Test message')]
            with pytest.raises(RateLimitError):
                await anthropic_client.generate_response(messages)

    @pytest.mark.asyncio
    async def test_refusal_error(self, anthropic_client, mock_async_anthropic):
        """Test handling of content policy violations (refusal errors)."""

        # Create a custom APIError that matches what we need
        class MockAPIError(Exception):
            def __init__(self, message):
                self.message = message
                super().__init__(message)

        # Patch the Anthropic error with our mock
        with patch('anthropic.APIError', MockAPIError):
            # Setup mock to raise APIError with refusal message
            mock_async_anthropic.messages.create.side_effect = MockAPIError('refused to respond')

            # Call method and check exception
            messages = [Message(role='user', content='Test message')]
            with pytest.raises(RefusalError):
                await anthropic_client.generate_response(messages)

    @pytest.mark.asyncio
    async def test_extract_json_from_text(self, anthropic_client):
        """Test the _extract_json_from_text method."""
        # Valid JSON embedded in text
        text = 'Some text before {"test_field": "value"} and after'
        result = anthropic_client._extract_json_from_text(text)
        assert result == {'test_field': 'value'}

        # Invalid JSON
        with pytest.raises(ValueError):
            anthropic_client._extract_json_from_text('Not JSON at all')

    @pytest.mark.asyncio
    async def test_create_tool(self, anthropic_client):
        """Test the _create_tool method with and without response model."""
        # With response model
        tools, tool_choice = anthropic_client._create_tool(ResponseModel)
        assert len(tools) == 1
        assert tools[0]['name'] == 'ResponseModel'
        assert tool_choice['name'] == 'ResponseModel'

        # Without response model (generic JSON)
        tools, tool_choice = anthropic_client._create_tool()
        assert len(tools) == 1
        assert tools[0]['name'] == 'generic_json_output'

    @pytest.mark.asyncio
    async def test_validation_error_retry(self, anthropic_client, mock_async_anthropic):
        """Test retry behavior on validation error."""
        content_item1 = MagicMock()
        content_item1.type = 'tool_use'
        content_item1.input = {'wrong_field': 'wrong_value'}

        content_item2 = MagicMock()
        content_item2.type = 'tool_use'
        content_item2.input = {'test_field': 'correct_value'}

        mock_async_anthropic.messages.create.side_effect = [
            _make_response([content_item1]),
            _make_response([content_item2]),
        ]

        messages = [Message(role='user', content='Test message')]
        result = await anthropic_client.generate_response(messages, response_model=ResponseModel)

        # Should have called create twice due to retry
        assert mock_async_anthropic.messages.create.call_count == 2
        assert result['test_field'] == 'correct_value'


if __name__ == '__main__':
    pytest.main(['-v', 'test_anthropic_client.py'])
