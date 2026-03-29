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

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message


class DummyResponseModel(BaseModel):
    name: str


class MockLLMClient(LLMClient):
    """Concrete implementation of LLMClient for testing"""

    def __init__(self, config: LLMConfig | None = None):
        super().__init__(config or LLMConfig())
        self.last_messages = []
        self.last_response_mode = None

    async def _generate_response(
        self,
        messages,
        response_model=None,
        max_tokens=None,
        model_size=None,
        response_mode='structured_json',
    ):
        self.last_messages = [Message(role=m.role, content=m.content) for m in messages]
        self.last_response_mode = response_mode
        return {'content': 'test'}


def test_clean_input():
    client = MockLLMClient(LLMConfig())

    test_cases = [
        # Basic text should remain unchanged
        ('Hello World', 'Hello World'),
        # Control characters should be removed
        ('Hello\x00World', 'HelloWorld'),
        # Newlines, tabs, returns should be preserved
        ('Hello\nWorld\tTest\r', 'Hello\nWorld\tTest\r'),
        # Invalid Unicode should be removed
        ('Hello\udcdeWorld', 'HelloWorld'),
        # Zero-width characters should be removed
        ('Hello\u200bWorld', 'HelloWorld'),
        ('Test\ufeffWord', 'TestWord'),
        # Multiple issues combined
        ('Hello\x00\u200b\nWorld\udcde', 'Hello\nWorld'),
        # Empty string should remain empty
        ('', ''),
        # Form feed and other control characters from the error case
        ('{"edges":[{"relation_typ...\f\x04Hn\\?"}]}', '{"edges":[{"relation_typ...Hn\\?"}]}'),
        # More specific control character tests
        ('Hello\x0cWorld', 'HelloWorld'),  # form feed \f
        ('Hello\x04World', 'HelloWorld'),  # end of transmission
        # Combined JSON-like string with control characters
        ('{"test": "value\f\x00\x04"}', '{"test": "value"}'),
    ]

    for input_str, expected in test_cases:
        assert client._clean_input(input_str) == expected, f'Failed for input: {repr(input_str)}'


@pytest.mark.asyncio
async def test_generate_response_appends_schema_for_structured_json():
    client = MockLLMClient()
    messages = [
        Message(role='system', content='System'),
        Message(role='user', content='User prompt'),
    ]

    await client.generate_response(
        messages,
        response_model=DummyResponseModel,
        response_mode='structured_json',
    )

    assert client.last_response_mode == 'structured_json'
    assert 'Respond with a JSON object' in client.last_messages[-1].content


@pytest.mark.asyncio
async def test_generate_response_skips_schema_for_structured_text():
    client = MockLLMClient()
    messages = [
        Message(role='system', content='System'),
        Message(role='user', content='User prompt'),
    ]

    await client.generate_response(
        messages,
        response_model=DummyResponseModel,
        response_mode='structured_text',
    )

    assert client.last_response_mode == 'structured_text'
    assert 'Respond with a JSON object' not in client.last_messages[-1].content


@pytest.mark.asyncio
async def test_openai_generic_client_returns_plain_text_for_structured_text():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='BEGIN ITEMS\nEND ITEMS'))]
    )
    client_impl = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response)))
    )
    client = OpenAIGenericClient(config=LLMConfig(), client=client_impl)

    result = await client._generate_response(
        messages=[Message(role='user', content='Prompt')],
        response_model=DummyResponseModel,
        response_mode='structured_text',
    )

    assert result == {'content': 'BEGIN ITEMS\nEND ITEMS'}
    create_kwargs = client_impl.chat.completions.create.await_args.kwargs
    assert 'response_format' not in create_kwargs


@pytest.mark.asyncio
async def test_openai_generic_client_uses_json_schema_for_structured_json():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"name":"Alice"}'))]
    )
    client_impl = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response)))
    )
    client = OpenAIGenericClient(config=LLMConfig(), client=client_impl)

    result = await client._generate_response(
        messages=[Message(role='user', content='Prompt')],
        response_model=DummyResponseModel,
        response_mode='structured_json',
    )

    assert result == {'name': 'Alice'}
    create_kwargs = client_impl.chat.completions.create.await_args.kwargs
    assert create_kwargs['response_format']['type'] == 'json_schema'


@pytest.mark.asyncio
async def test_openai_client_supports_structured_text_mode():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='BEGIN ITEMS\nEND ITEMS'))],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
    )
    client_impl = SimpleNamespace(
        responses=SimpleNamespace(parse=AsyncMock()),
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response))),
    )
    client = OpenAIClient(config=LLMConfig(), client=client_impl)

    result = await client.generate_response(
        messages=[Message(role='user', content='Prompt')],
        response_model=DummyResponseModel,
        response_mode='structured_text',
    )

    assert result == {'content': 'BEGIN ITEMS\nEND ITEMS'}
    create_kwargs = client_impl.chat.completions.create.await_args.kwargs
    assert 'response_format' not in create_kwargs
    client_impl.responses.parse.assert_not_awaited()
