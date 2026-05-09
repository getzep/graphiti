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

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message


def _make_client(content: str) -> OpenAIGenericClient:
    """Return an OpenAIGenericClient whose underlying API call returns *content*."""
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    mock_response = SimpleNamespace(choices=[choice])

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

    with patch('openai.AsyncOpenAI', return_value=mock_openai):
        client = OpenAIGenericClient(config=LLMConfig(api_key='test'))
        client.client = mock_openai
        return client


MESSAGES = [Message(role='user', content='hello')]


class TestCodeFenceStripping:
    """OpenAIGenericClient strips markdown code fences before JSON parsing."""

    @pytest.mark.asyncio
    async def test_plain_json_unchanged(self):
        client = _make_client('{"key": "value"}')
        result = await client._generate_response(MESSAGES)
        assert result == {'key': 'value'}

    @pytest.mark.asyncio
    async def test_json_code_fence(self):
        client = _make_client('```json\n{"key": "value"}\n```')
        result = await client._generate_response(MESSAGES)
        assert result == {'key': 'value'}

    @pytest.mark.asyncio
    async def test_plain_code_fence(self):
        client = _make_client('```\n{"key": "value"}\n```')
        result = await client._generate_response(MESSAGES)
        assert result == {'key': 'value'}

    @pytest.mark.asyncio
    async def test_code_fence_with_extra_whitespace(self):
        client = _make_client('```json\n  {"key": "value"}  \n```')
        result = await client._generate_response(MESSAGES)
        assert result == {'key': 'value'}
