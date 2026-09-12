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

import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.prompts.models import Message


class MockLLMClient(LLMClient):
    """Concrete implementation of LLMClient for testing"""

    async def _generate_response(
        self, messages, response_model=None, max_tokens=None, model_size=None
    ):
        self.last_messages = messages
        return {'content': 'test'}


class ResponseModel(BaseModel):
    fact: str


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


def test_attribute_extraction_preamble_no_op_when_disabled():
    client = MockLLMClient(LLMConfig())
    messages = [Message(role='system', content='base'), Message(role='user', content='hi')]
    client._apply_attribute_extraction_preamble(messages, attribute_extraction=False)
    assert messages[0].content == 'base'
    assert messages[1].content == 'hi'


def test_attribute_extraction_preamble_appends_to_system():
    client = MockLLMClient(LLMConfig())
    messages = [
        Message(role='system', content='You are helpful.'),
        Message(role='user', content='hi'),
    ]
    client._apply_attribute_extraction_preamble(messages, attribute_extraction=True)
    assert messages[0].content.startswith('You are helpful.')
    assert 'ATTRIBUTE EXTRACTION:' in messages[0].content
    assert 'NEVER themselves valid values' in messages[0].content
    assert messages[1].content == 'hi'  # user message untouched


def test_attribute_extraction_preamble_is_idempotent():
    client = MockLLMClient(LLMConfig())
    messages = [
        Message(role='system', content='You are helpful.'),
        Message(role='user', content='hi'),
    ]
    client._apply_attribute_extraction_preamble(messages, attribute_extraction=True)
    once = messages[0].content
    client._apply_attribute_extraction_preamble(messages, attribute_extraction=True)
    assert messages[0].content == once, 'second call must not double-append'


def test_attribute_extraction_preamble_falls_back_to_first_message_if_no_system():
    client = MockLLMClient(LLMConfig())
    messages = [Message(role='user', content='hi')]
    client._apply_attribute_extraction_preamble(messages, attribute_extraction=True)
    assert 'ATTRIBUTE EXTRACTION:' in messages[0].content
    assert messages[0].content.endswith('hi')
    # Sentinel must be at the front so the idempotency check finds it.
    assert messages[0].content.startswith('<<graphiti.attr_extraction.preamble.v1>>')


def test_attribute_extraction_preamble_handles_empty_messages():
    client = MockLLMClient(LLMConfig())
    messages: list[Message] = []
    client._apply_attribute_extraction_preamble(messages, attribute_extraction=True)
    assert messages == []


@pytest.mark.asyncio
async def test_generate_response_does_not_mutate_caller_messages():
    client = MockLLMClient(LLMConfig())
    messages = [
        Message(role='system', content='System message'),
        Message(role='user', content='User message\x00'),
    ]
    original = [message.model_dump() for message in messages]

    await client.generate_response(
        messages,
        response_model=ResponseModel,
        group_id='test-group',
        attribute_extraction=True,
    )

    assert [message.model_dump() for message in messages] == original
    assert client.last_messages is not messages
    assert client.last_messages[0] is not messages[0]
    assert 'ATTRIBUTE EXTRACTION:' in client.last_messages[0].content
    assert 'same language' in client.last_messages[0].content
    assert 'Respond with a JSON object in the following format' in client.last_messages[-1].content
    assert '\x00' not in client.last_messages[-1].content


@pytest.mark.asyncio
async def test_generate_response_preparation_is_idempotent_for_reused_messages():
    client = MockLLMClient(LLMConfig())
    messages = [
        Message(role='system', content='System message'),
        Message(role='user', content='User message'),
    ]

    await client.generate_response(
        messages, response_model=ResponseModel, attribute_extraction=True
    )
    first_call = [message.model_dump() for message in client.last_messages]

    await client.generate_response(
        messages, response_model=ResponseModel, attribute_extraction=True
    )
    second_call = [message.model_dump() for message in client.last_messages]

    assert second_call == first_call
