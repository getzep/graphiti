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

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.prompts.models import Message


class MockLLMClient(LLMClient):
    """Concrete implementation of LLMClient for testing"""

    async def _generate_response(
        self,
        messages,
        response_model=None,
        max_tokens=None,
        model_size=None,
        *,
        model=None,
        small_model=None,
    ):
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


class RecordingLLMClient(LLMClient):
    """Captures the model overrides forwarded into ``_generate_response``."""

    def __init__(self) -> None:
        super().__init__(LLMConfig(model='default-model'), cache=False)
        self.received: dict | None = None

    async def _generate_response(
        self,
        messages,
        response_model=None,
        max_tokens=None,
        model_size=None,
        *,
        model=None,
        small_model=None,
    ):
        self.received = {'model': model, 'small_model': small_model}
        return {'content': 'test'}


class LegacySignatureClient(LLMClient):
    """Pre-override ``_generate_response`` signature; must still work when unused."""

    def __init__(self) -> None:
        super().__init__(LLMConfig(model='default-model'), cache=False)
        self.called = False

    async def _generate_response(
        self, messages, response_model=None, max_tokens=None, model_size=None
    ):
        self.called = True
        return {'content': 'ok'}


def _sys_user() -> list[Message]:
    return [Message(role='system', content='sys'), Message(role='user', content='hi')]


@pytest.mark.asyncio
async def test_generate_response_forwards_model_overrides():
    client = RecordingLLMClient()
    await client.generate_response(_sys_user(), model='x', small_model='y')
    assert client.received == {'model': 'x', 'small_model': 'y'}


@pytest.mark.asyncio
async def test_generate_response_omitted_overrides_do_not_break_legacy_signature():
    client = LegacySignatureClient()
    result = await client.generate_response(_sys_user())
    assert client.called is True
    assert result == {'content': 'ok'}


def test_cache_key_includes_model_override():
    client = MockLLMClient(LLMConfig(model='default-model'))
    messages = _sys_user()
    default_key = client._get_cache_key(messages)
    omitted_key = client._get_cache_key(messages, model=None)
    override_key = client._get_cache_key(messages, model='x')
    assert default_key == omitted_key
    assert override_key != default_key
