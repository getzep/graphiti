"""Tests for OpenAI Generic Client.

Tests the OpenAI-compatible client used for LiteLLM, Ollama, vLLM, etc.
"""

import json
from unittest.mock import patch

import pytest

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient


class TestIsSchemaReturnedAsData:
    """Tests for _is_schema_returned_as_data method."""

    @pytest.fixture
    def client(self):
        """Create a client instance for testing with mocked OpenAI client."""
        with patch('graphiti_core.llm_client.openai_generic_client.AsyncOpenAI'):
            config = LLMConfig(api_key='test-key')
            return OpenAIGenericClient(config=config)

    def test_detects_schema_with_properties(self, client):
        """Schema with 'properties' key should be detected."""
        response = {
            'type': 'object',
            'properties': {'name': {'type': 'string'}},
            'required': ['name'],
        }
        assert client._is_schema_returned_as_data(response) is True

    def test_detects_schema_with_defs(self, client):
        """Schema with '$defs' key should be detected."""
        response = {'$defs': {'Person': {'type': 'object'}}}
        assert client._is_schema_returned_as_data(response) is True

    def test_detects_schema_with_schema_key(self, client):
        """Schema with '$schema' key should be detected."""
        response = {'$schema': 'http://json-schema.org/draft-07/schema#'}
        assert client._is_schema_returned_as_data(response) is True

    def test_detects_schema_with_definitions(self, client):
        """Schema with 'definitions' key should be detected."""
        response = {'definitions': {'Item': {'type': 'string'}}}
        assert client._is_schema_returned_as_data(response) is True

    def test_detects_type_object_at_top_level(self, client):
        """Top-level 'type': 'object' indicates schema."""
        response = {'type': 'object'}
        assert client._is_schema_returned_as_data(response) is True

    def test_real_data_not_detected_as_schema(self, client):
        """Normal data responses should not be detected as schema."""
        response = {'name': 'John', 'age': 30, 'city': 'NYC'}
        assert client._is_schema_returned_as_data(response) is False

    def test_nested_type_object_not_detected(self, client):
        """Nested 'type' fields in data should not trigger detection."""
        response = {'entity': {'type': 'person', 'name': 'John'}}
        assert client._is_schema_returned_as_data(response) is False

    def test_empty_response_not_detected(self, client):
        """Empty response should not be detected as schema."""
        response = {}
        assert client._is_schema_returned_as_data(response) is False


class TestExtractJson:
    """Tests for _extract_json method."""

    @pytest.fixture
    def client(self):
        """Create a client instance for testing with mocked OpenAI client."""
        with patch('graphiti_core.llm_client.openai_generic_client.AsyncOpenAI'):
            config = LLMConfig(api_key='test-key')
            return OpenAIGenericClient(config=config)

    def test_extracts_clean_json(self, client):
        """Clean JSON without trailing text should parse normally."""
        text = '{"name": "John", "age": 30}'
        result = client._extract_json(text)
        assert result == {'name': 'John', 'age': 30}

    def test_extracts_json_with_trailing_text(self, client):
        """JSON followed by text should extract only the JSON."""
        text = '{"name": "John"}\n\nThis is some explanation text.'
        result = client._extract_json(text)
        assert result == {'name': 'John'}

    def test_extracts_json_with_trailing_newlines(self, client):
        """JSON with trailing whitespace should parse correctly."""
        text = '{"status": "ok"}\n\n\n'
        result = client._extract_json(text)
        assert result == {'status': 'ok'}

    def test_handles_nested_braces(self, client):
        """Nested JSON objects should be extracted correctly."""
        text = '{"outer": {"inner": {"deep": true}}}\nExtra text here'
        result = client._extract_json(text)
        assert result == {'outer': {'inner': {'deep': True}}}

    def test_handles_strings_with_braces(self, client):
        """Braces inside strings should not affect extraction."""
        text = '{"message": "Hello {world}"}\nTrailing text'
        result = client._extract_json(text)
        assert result == {'message': 'Hello {world}'}

    def test_handles_escaped_quotes(self, client):
        """Escaped quotes in strings should be handled correctly."""
        text = '{"quote": "He said \\"hello\\""}\nMore text'
        result = client._extract_json(text)
        assert result == {'quote': 'He said "hello"'}

    def test_raises_on_no_json(self, client):
        """Text without JSON should raise JSONDecodeError."""
        text = 'This is just plain text'
        with pytest.raises(json.JSONDecodeError):
            client._extract_json(text)

    def test_raises_on_incomplete_json(self, client):
        """Incomplete JSON should raise JSONDecodeError."""
        text = '{"name": "John"'
        with pytest.raises(json.JSONDecodeError):
            client._extract_json(text)

    def test_handles_whitespace_before_json(self, client):
        """Leading whitespace should be stripped."""
        text = '  \n  {"data": true}'
        result = client._extract_json(text)
        assert result == {'data': True}

    def test_real_world_gemini_response(self, client):
        """Test with real-world Gemini response pattern."""
        text = """{"entities": [{"name": "User", "type": "Person"}]}

I've extracted the entities from your text. The main entity identified is "User" which appears to be a Person type."""
        result = client._extract_json(text)
        assert result == {'entities': [{'name': 'User', 'type': 'Person'}]}
