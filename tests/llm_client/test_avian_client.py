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

# Running tests: pytest -xvs tests/llm_client/test_avian_client.py

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.avian_client import AVIAN_BASE_URL, AvianClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.prompts.models import Message


# Rename class to avoid pytest collection as a test class
class ResponseModel(BaseModel):
    """Test model for response testing."""

    test_field: str
    optional_field: int = 0


@pytest.fixture
def mock_openai_client():
    """Fixture to mock the AsyncOpenAI client."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    return mock_client


@pytest.fixture
def avian_client(mock_openai_client):
    """Fixture to create an AvianClient with a mocked AsyncOpenAI client."""
    config = LLMConfig(
        api_key='test_api_key',
        model='deepseek/deepseek-v3.2',
        base_url=AVIAN_BASE_URL,
        temperature=0.5,
        max_tokens=1000,
    )
    client = AvianClient(config=config, cache=False, client=mock_openai_client, max_tokens=1000)
    return client


class TestAvianClientInitialization:
    """Tests for AvianClient initialization."""

    def test_init_with_config(self):
        """Test initialization with a config object."""
        config = LLMConfig(
            api_key='test_api_key',
            model='deepseek/deepseek-v3.2',
            base_url=AVIAN_BASE_URL,
            temperature=0.5,
            max_tokens=1000,
        )
        client = AvianClient(config=config, cache=False, max_tokens=1000)

        assert client.config == config
        assert client.model == 'deepseek/deepseek-v3.2'
        assert client.temperature == 0.5
        assert client.max_tokens == 1000

    def test_init_sets_avian_base_url(self):
        """Test that base URL defaults to Avian API when not provided."""
        config = LLMConfig(api_key='test_api_key')
        AvianClient(config=config, cache=False)

        assert config.base_url == AVIAN_BASE_URL

    @patch.dict(os.environ, {'AVIAN_API_KEY': 'env_api_key'})
    def test_init_without_config(self):
        """Test initialization without a config, using environment variable."""
        client = AvianClient(cache=False)

        assert client.config.api_key == 'env_api_key'
        assert client.config.base_url == AVIAN_BASE_URL
        assert client.model == 'deepseek/deepseek-v3.2'

    def test_init_with_custom_client(self):
        """Test initialization with a custom AsyncOpenAI client."""
        mock_client = MagicMock()
        client = AvianClient(client=mock_client)

        assert client.client == mock_client

    def test_init_with_custom_max_tokens(self):
        """Test initialization with custom max_tokens."""
        config = LLMConfig(api_key='test_api_key')
        client = AvianClient(config=config, max_tokens=32768)

        assert client.max_tokens == 32768

    def test_init_preserves_explicit_base_url(self):
        """Test that an explicitly set base_url is not overridden."""
        custom_url = 'https://custom.api.example.com/v1'
        config = LLMConfig(api_key='test_api_key', base_url=custom_url)
        AvianClient(config=config)

        assert config.base_url == custom_url


class TestAvianClientGenerateResponse:
    """Tests for AvianClient generate_response method."""

    @pytest.mark.asyncio
    async def test_generate_response_success(self, avian_client, mock_openai_client):
        """Test successful response generation."""
        # Setup mock response
        mock_message = MagicMock()
        mock_message.content = json.dumps({'test_field': 'test_value', 'optional_field': 42})

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_openai_client.chat.completions.create.return_value = mock_response

        # Call method
        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='User message'),
        ]
        result = await avian_client.generate_response(messages=messages)

        # Assertions
        assert isinstance(result, dict)
        assert result['test_field'] == 'test_value'
        assert result['optional_field'] == 42
        mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_with_response_model(self, avian_client, mock_openai_client):
        """Test response generation with a response model for structured output."""
        # Setup mock response
        mock_message = MagicMock()
        mock_message.content = json.dumps({'test_field': 'structured_value', 'optional_field': 0})

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_openai_client.chat.completions.create.return_value = mock_response

        # Call method
        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='User message'),
        ]
        result = await avian_client.generate_response(
            messages=messages, response_model=ResponseModel
        )

        # Assertions
        assert isinstance(result, dict)
        assert result['test_field'] == 'structured_value'

        # Verify json_schema response format was used
        call_kwargs = mock_openai_client.chat.completions.create.call_args
        response_format = call_kwargs.kwargs.get('response_format')
        assert response_format['type'] == 'json_schema'

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, avian_client, mock_openai_client):
        """Test handling of rate limit errors."""
        # Create a mock response for the RateLimitError
        mock_http_response = MagicMock()
        mock_http_response.status_code = 429
        mock_http_response.headers = {}

        mock_openai_client.chat.completions.create.side_effect = openai.RateLimitError(
            message='Rate limit exceeded',
            response=mock_http_response,
            body=None,
        )

        # Call method and check exception
        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='Test message'),
        ]
        with pytest.raises(RateLimitError):
            await avian_client.generate_response(messages)

    @pytest.mark.asyncio
    async def test_retry_on_invalid_json(self, avian_client, mock_openai_client):
        """Test retry behavior on invalid JSON response."""
        # First call returns invalid JSON, second returns valid JSON
        mock_message_invalid = MagicMock()
        mock_message_invalid.content = 'not valid json'

        mock_message_valid = MagicMock()
        mock_message_valid.content = json.dumps({'test_field': 'retry_value'})

        mock_choice_invalid = MagicMock()
        mock_choice_invalid.message = mock_message_invalid

        mock_choice_valid = MagicMock()
        mock_choice_valid.message = mock_message_valid

        mock_response_invalid = MagicMock()
        mock_response_invalid.choices = [mock_choice_invalid]

        mock_response_valid = MagicMock()
        mock_response_valid.choices = [mock_choice_valid]

        mock_openai_client.chat.completions.create.side_effect = [
            mock_response_invalid,
            mock_response_valid,
        ]

        # Call method
        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='Test message'),
        ]
        result = await avian_client.generate_response(messages=messages)

        # Should have called create twice due to retry
        assert mock_openai_client.chat.completions.create.call_count == 2
        assert result['test_field'] == 'retry_value'

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, avian_client, mock_openai_client):
        """Test that error is raised when max retries are exceeded."""
        # All calls return invalid JSON
        mock_message = MagicMock()
        mock_message.content = 'invalid json'

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_openai_client.chat.completions.create.return_value = mock_response

        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='Test message'),
        ]
        with pytest.raises(json.JSONDecodeError):
            await avian_client.generate_response(messages=messages)

        # Should have been called MAX_RETRIES + 1 times
        assert mock_openai_client.chat.completions.create.call_count == AvianClient.MAX_RETRIES + 1

    @pytest.mark.asyncio
    async def test_model_passed_correctly(self, avian_client, mock_openai_client):
        """Test that the configured model is passed to the API."""
        mock_message = MagicMock()
        mock_message.content = json.dumps({'result': 'ok'})

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_openai_client.chat.completions.create.return_value = mock_response

        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='Test message'),
        ]
        await avian_client.generate_response(messages=messages)

        call_kwargs = mock_openai_client.chat.completions.create.call_args
        assert call_kwargs.kwargs['model'] == 'deepseek/deepseek-v3.2'

    @pytest.mark.asyncio
    async def test_provider_type(self, avian_client):
        """Test that provider type is correctly identified."""
        assert avian_client._get_provider_type() == 'avian'


if __name__ == '__main__':
    pytest.main(['-v', 'test_avian_client.py'])
