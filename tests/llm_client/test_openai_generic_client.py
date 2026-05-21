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

# Running tests: pytest -xvs tests/llm_client/test_openai_generic_client.py

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message


class ExtractedEntity(BaseModel):
    name: str
    entity_type_id: int


class ExtractedEntities(BaseModel):
    extracted_entities: list[ExtractedEntity]


@pytest.fixture
def mock_openai_client():
    """Fixture to mock the AsyncOpenAI client."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    return mock_client


@pytest.fixture
def generic_client(mock_openai_client):
    """Fixture to create an OpenAIGenericClient with a mocked AsyncOpenAI."""
    config = LLMConfig(
        api_key='test_api_key',
        model='test-model',
        base_url='http://localhost:11434/v1',
        temperature=0.0,
    )
    client = OpenAIGenericClient(config=config, cache=False, client=mock_openai_client)
    return client


def _make_completion_response(content: str):
    """Helper to create a mock chat completion response."""
    mock_message = MagicMock()
    mock_message.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


class TestOpenAIGenericClientValidation:
    """Tests for Pydantic validation of LLM responses."""

    @pytest.mark.asyncio
    async def test_valid_response_passes_validation(self, generic_client, mock_openai_client):
        """Test that a valid response is validated and returned."""
        valid_response = json.dumps({
            'extracted_entities': [
                {'name': 'Alice', 'entity_type_id': 1},
                {'name': 'Bob', 'entity_type_id': 2},
            ]
        })
        mock_openai_client.chat.completions.create.return_value = _make_completion_response(
            valid_response
        )

        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='User message'),
        ]
        result = await generic_client.generate_response(
            messages=messages, response_model=ExtractedEntities
        )

        assert isinstance(result, dict)
        assert len(result['extracted_entities']) == 2
        assert result['extracted_entities'][0]['name'] == 'Alice'

    @pytest.mark.asyncio
    async def test_schema_echo_triggers_retry(self, generic_client, mock_openai_client):
        """Test that when the LLM returns the schema definition instead of data, it retries."""
        # First response: LLM returns the schema definition (the bug from issue #912)
        schema_echo = json.dumps({
            '$defs': {'ExtractedEntity': {'properties': {}}},
            'entity_type_id': 0,
        })
        # Second response: LLM returns valid data after retry
        valid_response = json.dumps({
            'extracted_entities': [{'name': 'Alice', 'entity_type_id': 1}]
        })

        mock_openai_client.chat.completions.create.side_effect = [
            _make_completion_response(schema_echo),
            _make_completion_response(valid_response),
        ]

        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='User message'),
        ]
        result = await generic_client.generate_response(
            messages=messages, response_model=ExtractedEntities
        )

        # Should have retried and succeeded
        assert mock_openai_client.chat.completions.create.call_count == 2
        assert result['extracted_entities'][0]['name'] == 'Alice'

    @pytest.mark.asyncio
    async def test_validation_error_after_max_retries(self, generic_client, mock_openai_client):
        """Test that persistent validation errors raise after max retries."""
        invalid_response = json.dumps({'wrong_field': 'wrong_value'})

        mock_openai_client.chat.completions.create.return_value = _make_completion_response(
            invalid_response
        )

        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='User message'),
        ]
        with pytest.raises(ValidationError):
            await generic_client.generate_response(
                messages=messages, response_model=ExtractedEntities
            )

        # Should have tried 1 initial + 2 retries = 3 calls
        assert mock_openai_client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_no_validation_without_response_model(self, generic_client, mock_openai_client):
        """Test that responses without response_model are returned as-is."""
        raw_response = json.dumps({'any_key': 'any_value'})
        mock_openai_client.chat.completions.create.return_value = _make_completion_response(
            raw_response
        )

        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='User message'),
        ]
        result = await generic_client.generate_response(messages=messages)

        assert result == {'any_key': 'any_value'}
        assert mock_openai_client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_uses_json_object_format(self, generic_client, mock_openai_client):
        """Test that json_object format is used instead of json_schema for compatibility."""
        valid_response = json.dumps({
            'extracted_entities': [{'name': 'Alice', 'entity_type_id': 1}]
        })
        mock_openai_client.chat.completions.create.return_value = _make_completion_response(
            valid_response
        )

        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='User message'),
        ]
        await generic_client.generate_response(
            messages=messages, response_model=ExtractedEntities
        )

        # Verify json_object format was used
        call_kwargs = mock_openai_client.chat.completions.create.call_args
        assert call_kwargs.kwargs['response_format'] == {'type': 'json_object'}


class TestOpenAIGenericClientInitialization:
    """Tests for OpenAIGenericClient initialization."""

    def test_init_with_config(self):
        """Test initialization with a config object."""
        config = LLMConfig(
            api_key='test_api_key',
            model='llama3',
            base_url='http://localhost:11434/v1',
        )
        client = OpenAIGenericClient(config=config)
        assert client.model == 'llama3'
        assert client.max_tokens == 16384

    def test_init_with_custom_max_tokens(self):
        """Test initialization with custom max_tokens."""
        config = LLMConfig(api_key='test_api_key')
        client = OpenAIGenericClient(config=config, max_tokens=8192)
        assert client.max_tokens == 8192

    def test_cache_not_implemented(self):
        """Test that caching raises NotImplementedError."""
        config = LLMConfig(api_key='test_api_key')
        with pytest.raises(NotImplementedError):
            OpenAIGenericClient(config=config, cache=True)


if __name__ == '__main__':
    pytest.main(['-v', 'test_openai_generic_client.py'])
