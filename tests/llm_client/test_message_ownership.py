"""
Copyright 2026, Zep Software, Inc.

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

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.prompts.models import Message


class Output(BaseModel):
    value: str


class StubClient(LLMClient):
    async def _generate_response(self, *args, **kwargs):
        return {'value': 'ok'}


def make_client(provider):
    config = LLMConfig(model='offline-test', api_key='test-key')
    transport = MagicMock()
    if provider == 'base':
        return StubClient(config), {'value': 'ok'}
    if provider == 'openai':
        from graphiti_core.llm_client.openai_client import OpenAIClient

        return OpenAIClient(config=config, client=transport), ({'value': 'ok'}, 1, 2)
    if provider == 'azure':
        from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient

        return AzureOpenAILLMClient(azure_client=transport, config=config), ({'value': 'ok'}, 1, 2)
    if provider == 'generic':
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        return OpenAIGenericClient(
            config=config, client=transport, structured_output_mode='json_object'
        ), {'value': 'ok'}
    if provider == 'anthropic':
        from graphiti_core.llm_client.anthropic_client import AnthropicClient

        return AnthropicClient(config=config, client=transport), ({'value': 'ok'}, 1, 2)
    from graphiti_core.llm_client.gemini_client import GeminiClient

    return GeminiClient(config=config, client=transport), ({'value': 'ok'}, 1, 2)


@pytest.mark.parametrize('provider', ['base', 'openai', 'azure', 'generic', 'anthropic', 'gemini'])
async def test_reusing_messages_keeps_request_specific_framing_local(provider):
    client, result = make_client(provider)
    requests = []

    async def generate(messages, *args, **kwargs):
        requests.append([message.model_dump() for message in messages])
        return result

    client._generate_response = AsyncMock(side_effect=generate)
    messages = [
        Message(role='system', content='Extract facts.'),
        Message(role='user', content='Alice lives in Paris.'),
    ]
    original_messages = list(messages)
    original_content = [message.model_dump() for message in messages]

    assert await client.generate_response(
        messages, response_model=Output, attribute_extraction=True
    ) == {'value': 'ok'}
    assert await client.generate_response(
        messages, response_model=Output, attribute_extraction=False
    ) == {'value': 'ok'}

    assert [message.model_dump() for message in messages] == original_content
    assert all(
        current is original for current, original in zip(messages, original_messages, strict=True)
    )
    assert '<<graphiti.attr_extraction.preamble.v1>>' in requests[0][0]['content']
    assert '<<graphiti.attr_extraction.preamble.v1>>' not in requests[1][0]['content']
    if provider in ('base', 'generic'):
        assert requests[1][-1]['content'].count('Respond with a JSON object') == 1


async def test_reusing_messages_hits_the_response_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        'graphiti_core.llm_client.client.DEFAULT_CACHE_DIR', str(tmp_path / 'cache')
    )
    client = StubClient(LLMConfig(model='offline-test'), cache=True)
    client._generate_response = AsyncMock(return_value={'value': 'ok'})
    messages = [
        Message(role='system', content='Extract facts.'),
        Message(role='user', content='Alice lives in Paris.'),
    ]
    try:
        for _ in range(2):
            assert await client.generate_response(messages, response_model=Output) == {
                'value': 'ok'
            }
        client._generate_response.assert_awaited_once()
    finally:
        client.cache_dir.close()


@pytest.mark.parametrize('provider', ['openai', 'azure', 'anthropic', 'gemini'])
@pytest.mark.parametrize('recovers', [True, False])
async def test_application_retry_feedback_does_not_escape_the_request(provider, recovers):
    client, result = make_client(provider)
    requests = []

    async def generate(messages, *args, **kwargs):
        requests.append([message.model_dump() for message in messages])
        if len(requests) == 1 or not recovers:
            raise ValueError('invalid structured output')
        return result

    client._generate_response = AsyncMock(side_effect=generate)
    messages = [
        Message(role='system', content='Extract facts.'),
        Message(role='user', content='Alice.'),
    ]
    original_content = [message.model_dump() for message in messages]

    if recovers:
        assert await client.generate_response(messages) == {'value': 'ok'}
        assert len(requests) == 2
    else:
        with pytest.raises(ValueError, match='invalid structured output'):
            await client.generate_response(messages)
        assert len(requests) == (2 if provider == 'gemini' else 3)

    assert len(requests[1]) == 3
    assert 'invalid structured output' in requests[1][-1]['content']
    assert [message.model_dump() for message in messages] == original_content
