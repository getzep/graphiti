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

# Running tests: pytest -xvs tests/llm_client/test_gemini_client.py

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.llm_client.gemini_client import DEFAULT_MODEL, DEFAULT_SMALL_MODEL, GeminiClient
from graphiti_core.prompts.models import Message


# Test model for response testing
class ResponseModel(BaseModel):
    """Test model for response testing."""

    test_field: str
    optional_field: int = 0


@pytest.fixture
def mock_gemini_client():
    """Fixture to mock the Google Gemini client."""
    with patch('google.genai.Client') as mock_client:
        # Setup mock instance and its methods
        mock_instance = mock_client.return_value
        mock_instance.aio = MagicMock()
        mock_instance.aio.models = MagicMock()
        mock_instance.aio.models.generate_content = AsyncMock()
        yield mock_instance


@pytest.fixture
def gemini_client(mock_gemini_client):
    """Fixture to create a GeminiClient with a mocked client."""
    config = LLMConfig(api_key='test_api_key', model='test-model', temperature=0.5, max_tokens=1000)
    client = GeminiClient(config=config, cache=False)
    # Replace the client's client with our mock to ensure we're using the mock
    client.client = mock_gemini_client
    return client


class TestGeminiClientInitialization:
    """Tests for GeminiClient initialization."""

    @patch('google.genai.Client')
    def test_init_with_config(self, mock_client):
        """Test initialization with a config object."""
        config = LLMConfig(
            api_key='test_api_key', model='test-model', temperature=0.5, max_tokens=1000
        )
        client = GeminiClient(config=config, cache=False, max_tokens=1000)

        assert client.config == config
        assert client.model == 'test-model'
        assert client.temperature == 0.5
        assert client.max_tokens == 1000

    @patch('google.genai.Client')
    def test_init_with_default_model(self, mock_client):
        """Test initialization with default model when none is provided."""
        config = LLMConfig(api_key='test_api_key', model=DEFAULT_MODEL)
        client = GeminiClient(config=config, cache=False)

        assert client.model == DEFAULT_MODEL

    @patch('google.genai.Client')
    def test_init_without_config(self, mock_client):
        """Test initialization without a config uses defaults."""
        client = GeminiClient(cache=False)

        assert client.config is not None
        # When no config.model is set, it will be None, not DEFAULT_MODEL
        assert client.model is None

    @patch('google.genai.Client')
    def test_init_with_thinking_config(self, mock_client):
        """Test initialization with thinking config."""
        with patch('google.genai.types.ThinkingConfig') as mock_thinking_config:
            thinking_config = mock_thinking_config.return_value
            client = GeminiClient(thinking_config=thinking_config)
            assert client.thinking_config == thinking_config


class TestGeminiClientGenerateResponse:
    """Tests for GeminiClient generate_response method."""

    @pytest.mark.asyncio
    async def test_generate_response_simple_text(self, gemini_client, mock_gemini_client):
        """Test successful response generation with simple text."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = 'Test response text'
        mock_response.candidates = []
        mock_response.prompt_feedback = None
        mock_gemini_client.aio.models.generate_content.return_value = mock_response

        # Call method
        messages = [Message(role='user', content='Test message')]
        result = await gemini_client.generate_response(messages)

        # Assertions
        assert isinstance(result, dict)
        assert result['content'] == 'Test response text'
        mock_gemini_client.aio.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_with_structured_output(
        self, gemini_client, mock_gemini_client
    ):
        """Test response generation with structured output."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = '{"test_field": "test_value", "optional_field": 42}'
        mock_response.candidates = []
        mock_response.prompt_feedback = None
        mock_gemini_client.aio.models.generate_content.return_value = mock_response

        # Call method
        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='User message'),
        ]
        result = await gemini_client.generate_response(
            messages=messages, response_model=ResponseModel
        )

        # Assertions
        assert isinstance(result, dict)
        assert result['test_field'] == 'test_value'
        assert result['optional_field'] == 42
        mock_gemini_client.aio.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_with_system_message(self, gemini_client, mock_gemini_client):
        """Test response generation with system message handling."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = 'Response with system context'
        mock_response.candidates = []
        mock_response.prompt_feedback = None
        mock_gemini_client.aio.models.generate_content.return_value = mock_response

        # Call method
        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='User message'),
        ]
        await gemini_client.generate_response(messages)

        # Verify system message is processed correctly
        call_args = mock_gemini_client.aio.models.generate_content.call_args
        config = call_args[1]['config']
        assert 'System message' in config.system_instruction

    @pytest.mark.asyncio
    async def test_get_model_for_size(self, gemini_client):
        """Test model selection based on size."""
        # Test small model
        small_model = gemini_client._get_model_for_size(ModelSize.small)
        assert small_model == DEFAULT_SMALL_MODEL

        # Test medium/large model
        medium_model = gemini_client._get_model_for_size(ModelSize.medium)
        assert medium_model == gemini_client.model

    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self, gemini_client, mock_gemini_client):
        """Test handling of rate limit errors."""
        # Setup mock to raise rate limit error
        mock_gemini_client.aio.models.generate_content.side_effect = Exception(
            'Rate limit exceeded'
        )

        # Call method and check exception
        messages = [Message(role='user', content='Test message')]
        with pytest.raises(RateLimitError):
            await gemini_client.generate_response(messages)

    @pytest.mark.asyncio
    async def test_quota_error_handling(self, gemini_client, mock_gemini_client):
        """Test handling of quota errors."""
        # Setup mock to raise quota error
        mock_gemini_client.aio.models.generate_content.side_effect = Exception(
            'Quota exceeded for requests'
        )

        # Call method and check exception
        messages = [Message(role='user', content='Test message')]
        with pytest.raises(RateLimitError):
            await gemini_client.generate_response(messages)

    @pytest.mark.asyncio
    async def test_resource_exhausted_error_handling(self, gemini_client, mock_gemini_client):
        """Test handling of resource exhausted errors."""
        # Setup mock to raise resource exhausted error
        mock_gemini_client.aio.models.generate_content.side_effect = Exception(
            'resource_exhausted: Request limit exceeded'
        )

        # Call method and check exception
        messages = [Message(role='user', content='Test message')]
        with pytest.raises(RateLimitError):
            await gemini_client.generate_response(messages)

    @pytest.mark.asyncio
    async def test_safety_block_handling(self, gemini_client, mock_gemini_client):
        """Test handling of safety blocks."""
        # Setup mock response with safety block
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = 'SAFETY'
        mock_candidate.safety_ratings = [
            MagicMock(blocked=True, category='HARM_CATEGORY_HARASSMENT', probability='HIGH')
        ]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.prompt_feedback = None
        mock_response.text = ''
        mock_gemini_client.aio.models.generate_content.return_value = mock_response

        # Call method and check exception
        messages = [Message(role='user', content='Test message')]
        with pytest.raises(Exception, match='Content blocked by safety filters'):
            await gemini_client.generate_response(messages)

    @pytest.mark.asyncio
    async def test_prompt_block_handling(self, gemini_client, mock_gemini_client):
        """Test handling of prompt blocks."""
        # Setup mock response with prompt block
        mock_prompt_feedback = MagicMock()
        mock_prompt_feedback.block_reason = 'BLOCKED_REASON_OTHER'

        mock_response = MagicMock()
        mock_response.candidates = []
        mock_response.prompt_feedback = mock_prompt_feedback
        mock_response.text = ''
        mock_gemini_client.aio.models.generate_content.return_value = mock_response

        # Call method and check exception
        messages = [Message(role='user', content='Test message')]
        with pytest.raises(Exception, match='Content blocked by safety filters'):
            await gemini_client.generate_response(messages)

    @pytest.mark.asyncio
    async def test_structured_output_parsing_error(self, gemini_client, mock_gemini_client):
        """Test handling of structured output parsing errors."""
        # Setup mock response with invalid JSON that will exhaust retries
        mock_response = MagicMock()
        mock_response.text = 'Invalid JSON that cannot be parsed'
        mock_response.candidates = []
        mock_response.prompt_feedback = None
        mock_gemini_client.aio.models.generate_content.return_value = mock_response

        # Call method and check exception - should exhaust retries
        messages = [Message(role='user', content='Test message')]
        with pytest.raises(Exception):  # noqa: B017
            await gemini_client.generate_response(messages, response_model=ResponseModel)

        # Should have called generate_content MAX_RETRIES times (2 attempts total)
        assert mock_gemini_client.aio.models.generate_content.call_count == GeminiClient.MAX_RETRIES

    @pytest.mark.asyncio
    async def test_retry_logic_with_safety_block(self, gemini_client, mock_gemini_client):
        """Test that safety blocks are not retried."""
        # Setup mock response with safety block
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = 'SAFETY'
        mock_candidate.safety_ratings = [
            MagicMock(blocked=True, category='HARM_CATEGORY_HARASSMENT', probability='HIGH')
        ]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.prompt_feedback = None
        mock_response.text = ''
        mock_gemini_client.aio.models.generate_content.return_value = mock_response

        # Call method and check that it doesn't retry
        messages = [Message(role='user', content='Test message')]
        with pytest.raises(Exception, match='Content blocked by safety filters'):
            await gemini_client.generate_response(messages)

        # Should only be called once (no retries for safety blocks)
        assert mock_gemini_client.aio.models.generate_content.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_logic_with_validation_error(self, gemini_client, mock_gemini_client):
        """Test retry behavior on validation error."""
        # First call returns invalid JSON, second call returns valid data
        mock_response1 = MagicMock()
        mock_response1.text = 'Invalid JSON that cannot be parsed'
        mock_response1.candidates = []
        mock_response1.prompt_feedback = None

        mock_response2 = MagicMock()
        mock_response2.text = '{"test_field": "correct_value"}'
        mock_response2.candidates = []
        mock_response2.prompt_feedback = None

        mock_gemini_client.aio.models.generate_content.side_effect = [
            mock_response1,
            mock_response2,
        ]

        # Call method
        messages = [Message(role='user', content='Test message')]
        result = await gemini_client.generate_response(messages, response_model=ResponseModel)

        # Should have called generate_content twice due to retry
        assert mock_gemini_client.aio.models.generate_content.call_count == 2
        assert result['test_field'] == 'correct_value'

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, gemini_client, mock_gemini_client):
        """Test behavior when max retries are exceeded."""
        # Setup mock to always return invalid JSON
        mock_response = MagicMock()
        mock_response.text = 'Invalid JSON that cannot be parsed'
        mock_response.candidates = []
        mock_response.prompt_feedback = None
        mock_gemini_client.aio.models.generate_content.return_value = mock_response

        # Call method and check exception
        messages = [Message(role='user', content='Test message')]
        with pytest.raises(Exception):  # noqa: B017
            await gemini_client.generate_response(messages, response_model=ResponseModel)

        # Should have called generate_content MAX_RETRIES times (2 attempts total)
        assert mock_gemini_client.aio.models.generate_content.call_count == GeminiClient.MAX_RETRIES

    @pytest.mark.asyncio
    async def test_empty_response_handling(self, gemini_client, mock_gemini_client):
        """Test handling of empty responses."""
        # Setup mock response with no text
        mock_response = MagicMock()
        mock_response.text = ''
        mock_response.candidates = []
        mock_response.prompt_feedback = None
        mock_gemini_client.aio.models.generate_content.return_value = mock_response

        # Call method with structured output and check exception
        messages = [Message(role='user', content='Test message')]
        with pytest.raises(Exception):  # noqa: B017
            await gemini_client.generate_response(messages, response_model=ResponseModel)

        # Should have exhausted retries due to empty response (2 attempts total)
        assert mock_gemini_client.aio.models.generate_content.call_count == GeminiClient.MAX_RETRIES

    @pytest.mark.asyncio
    async def test_custom_max_tokens(self, gemini_client, mock_gemini_client):
        """Test that explicit max_tokens parameter takes precedence over all other values."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = 'Test response'
        mock_response.candidates = []
        mock_response.prompt_feedback = None
        mock_gemini_client.aio.models.generate_content.return_value = mock_response

        # Call method with custom max tokens (should take precedence)
        messages = [Message(role='user', content='Test message')]
        await gemini_client.generate_response(messages, max_tokens=500)

        # Verify explicit max_tokens parameter takes precedence
        call_args = mock_gemini_client.aio.models.generate_content.call_args
        config = call_args[1]['config']
        # Explicit parameter should override everything else
        assert config.max_output_tokens == 500

    @pytest.mark.asyncio
    async def test_max_tokens_precedence_fallback(self, mock_gemini_client):
        """Test max_tokens precedence when no explicit parameter is provided."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = 'Test response'
        mock_response.candidates = []
        mock_response.prompt_feedback = None
        mock_gemini_client.aio.models.generate_content.return_value = mock_response

        # Test case 1: No explicit max_tokens, has instance max_tokens
        config = LLMConfig(
            api_key='test_api_key', model='test-model', temperature=0.5, max_tokens=1000
        )
        client = GeminiClient(
            config=config, cache=False, max_tokens=2000, client=mock_gemini_client
        )

        messages = [Message(role='user', content='Test message')]
        await client.generate_response(messages)

        call_args = mock_gemini_client.aio.models.generate_content.call_args
        config = call_args[1]['config']
        # Instance max_tokens should be used
        assert config.max_output_tokens == 2000

        # Test case 2: No explicit max_tokens, no instance max_tokens, uses model mapping
        config = LLMConfig(api_key='test_api_key', model='gemini-2.5-flash', temperature=0.5)
        client = GeminiClient(config=config, cache=False, client=mock_gemini_client)

        messages = [Message(role='user', content='Test message')]
        await client.generate_response(messages)

        call_args = mock_gemini_client.aio.models.generate_content.call_args
        config = call_args[1]['config']
        # Model mapping should be used
        assert config.max_output_tokens == 65536

    @pytest.mark.asyncio
    async def test_model_size_selection(self, gemini_client, mock_gemini_client):
        """Test that the correct model is selected based on model size."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = 'Test response'
        mock_response.candidates = []
        mock_response.prompt_feedback = None
        mock_gemini_client.aio.models.generate_content.return_value = mock_response

        # Call method with small model size
        messages = [Message(role='user', content='Test message')]
        await gemini_client.generate_response(messages, model_size=ModelSize.small)

        # Verify correct model is used
        call_args = mock_gemini_client.aio.models.generate_content.call_args
        assert call_args[1]['model'] == DEFAULT_SMALL_MODEL

    @pytest.mark.asyncio
    async def test_gemini_model_max_tokens_mapping(self, mock_gemini_client):
        """Test that different Gemini models use their correct max tokens."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = 'Test response'
        mock_response.candidates = []
        mock_response.prompt_feedback = None
        mock_gemini_client.aio.models.generate_content.return_value = mock_response

        # Test data: (model_name, expected_max_tokens)
        test_cases = [
            ('gemini-2.5-flash', 65536),
            ('gemini-2.5-pro', 65536),
            ('gemini-2.5-flash-lite', 64000),
            ('models/gemini-2.5-flash-lite-preview-06-17', 64000),
            ('gemini-2.0-flash', 8192),
            ('gemini-1.5-pro', 8192),
            ('gemini-1.5-flash', 8192),
            ('unknown-model', 8192),  # Fallback case
        ]

        for model_name, expected_max_tokens in test_cases:
            # Create client with specific model, no explicit max_tokens to test mapping
            config = LLMConfig(api_key='test_api_key', model=model_name, temperature=0.5)
            client = GeminiClient(config=config, cache=False, client=mock_gemini_client)

            # Call method without explicit max_tokens to test model mapping fallback
            messages = [Message(role='user', content='Test message')]
            await client.generate_response(messages)

            # Verify correct max tokens is used from model mapping
            call_args = mock_gemini_client.aio.models.generate_content.call_args
            config = call_args[1]['config']
            assert config.max_output_tokens == expected_max_tokens, (
                f'Model {model_name} should use {expected_max_tokens} tokens'
            )


if __name__ == '__main__':
    pytest.main(['-v', 'test_gemini_client.py'])
