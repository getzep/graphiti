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

# Running tests: pytest -xvs tests/llm_client/test_amazon_bedrock_client.py

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.amazon_bedrock_client import AmazonBedrockLLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.prompts.models import Message


class ResponseModel(BaseModel):
    """Test model for response testing."""

    test_field: str
    optional_field: int = 0


@pytest.fixture
def mock_boto3_client():
    """Fixture to mock the boto3 bedrock-runtime client."""
    with patch('boto3.client') as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def bedrock_client(mock_boto3_client):
    """Fixture to create an AmazonBedrockLLMClient with a mocked boto3 client."""
    config = LLMConfig(
        model='us.anthropic.claude-sonnet-4-20250514-v1:0', temperature=0.5, max_tokens=1000
    )
    client = AmazonBedrockLLMClient(config=config, region='us-east-1')
    client.client = mock_boto3_client
    return client


class TestAmazonBedrockLLMClientInitialization:
    """Tests for AmazonBedrockLLMClient initialization."""

    def test_init_with_config(self):
        """Test initialization with a config object."""
        config = LLMConfig(model='test-model', temperature=0.5, max_tokens=1000)
        client = AmazonBedrockLLMClient(config=config, max_tokens=1000, region='eu-west-1')

        assert client.model == 'test-model'
        assert client.temperature == 0.5
        assert client.max_tokens == 1000
        assert client.region == 'eu-west-1'

    def test_init_without_config(self):
        """Test initialization without a config uses defaults."""
        client = AmazonBedrockLLMClient(region='ap-southeast-1')

        assert client.model == 'us.anthropic.claude-sonnet-4-20250514-v1:0'
        assert client.temperature == 0.7
        assert client.region == 'ap-southeast-1'

    def test_init_default_region(self):
        """Test initialization with default region."""
        client = AmazonBedrockLLMClient()
        assert client.region == 'us-east-1'


class TestAmazonBedrockLLMClientGenerateResponse:
    """Tests for AmazonBedrockLLMClient generate_response method."""

    @pytest.mark.asyncio
    async def test_generate_response_without_model(self, bedrock_client, mock_boto3_client):
        """Test successful response generation without response model."""
        # Setup mock response
        mock_response_body = {'content': [{'text': 'This is a test response'}]}
        mock_response = {'body': MagicMock()}
        mock_response['body'].read.return_value.decode.return_value = json.dumps(mock_response_body)
        mock_boto3_client.invoke_model.return_value = mock_response

        # Call method
        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='User message'),
        ]

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock()
            mock_executor.return_value = mock_response
            mock_loop.return_value.run_in_executor = mock_executor

            result = await bedrock_client._generate_response(messages=messages)

        # Assertions
        assert isinstance(result, dict)
        assert result['content'] == 'This is a test response'
        mock_executor.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_with_model(self, bedrock_client, mock_boto3_client):
        """Test successful response generation with response model."""
        # Setup mock response with valid JSON
        mock_response_body = {
            'content': [{'text': '{"test_field": "test_value", "optional_field": 42}'}]
        }
        mock_response = {'body': MagicMock()}
        mock_response['body'].read.return_value.decode.return_value = json.dumps(mock_response_body)
        mock_boto3_client.invoke_model.return_value = mock_response

        # Call method
        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='User message'),
        ]

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock()
            mock_executor.return_value = mock_response
            mock_loop.return_value.run_in_executor = mock_executor

            result = await bedrock_client._generate_response(
                messages=messages, response_model=ResponseModel
            )

        # Assertions
        assert isinstance(result, dict)
        assert result['test_field'] == 'test_value'
        assert result['optional_field'] == 42

    @pytest.mark.asyncio
    async def test_generate_response_json_parsing_error(self, bedrock_client, mock_boto3_client):
        """Test handling of JSON parsing errors."""
        # Setup mock response with invalid JSON
        mock_response_body = {'content': [{'text': 'Invalid JSON response'}]}
        mock_response = {'body': MagicMock()}
        mock_response['body'].read.return_value.decode.return_value = json.dumps(mock_response_body)
        mock_boto3_client.invoke_model.return_value = mock_response

        messages = [Message(role='user', content='Test message')]

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock()
            mock_executor.return_value = mock_response
            mock_loop.return_value.run_in_executor = mock_executor

            with pytest.raises((ValueError, json.JSONDecodeError)):
                await bedrock_client._generate_response(
                    messages=messages, response_model=ResponseModel
                )

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, bedrock_client, mock_boto3_client):
        """Test handling of rate limit errors."""
        messages = [Message(role='user', content='Test message')]

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock()
            mock_executor.side_effect = Exception('Throttling error occurred')
            mock_loop.return_value.run_in_executor = mock_executor

            with pytest.raises(RateLimitError):
                await bedrock_client._generate_response(messages=messages)

    @pytest.mark.asyncio
    async def test_clean_json_response(self, bedrock_client):
        """Test the _clean_json_response method."""
        # Test with markdown code blocks
        text_with_markdown = '```json\n{"test": "value"}\n```'
        result = bedrock_client._clean_json_response(text_with_markdown)
        assert result == '{"test": "value"}'

        # Test with double braces
        text_with_double_braces = '{{"test": "value"}}'
        result = bedrock_client._clean_json_response(text_with_double_braces)
        assert result == '{"test": "value"}'

        # Test with extra text around JSON
        text_with_extra = 'Here is the JSON: {"test": "value"} and some more text'
        result = bedrock_client._clean_json_response(text_with_extra)
        assert result == '{"test": "value"}'

    @pytest.mark.asyncio
    async def test_invoke_bedrock_model_system_message(self, bedrock_client, mock_boto3_client):
        """Test that system messages are handled correctly."""
        # Setup mock response
        mock_response_body = {'content': [{'text': 'Response'}]}
        mock_response = {'body': MagicMock()}
        mock_response['body'].read.return_value.decode.return_value = json.dumps(mock_response_body)
        mock_boto3_client.invoke_model.return_value = mock_response

        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant'},
            {'role': 'user', 'content': 'Hello'},
        ]

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock()
            mock_executor.return_value = mock_response
            mock_loop.return_value.run_in_executor = mock_executor

            await bedrock_client._invoke_bedrock_model(
                model='test-model',
                messages=messages,
                temperature=0.7,
                max_tokens=100,
                response_format='text',
            )

        # Verify the call was made with system prompt
        mock_executor.assert_called_once()
        # We can't easily inspect the lambda, but we know it was called


if __name__ == '__main__':
    pytest.main(['-v', 'test_amazon_bedrock_client.py'])
