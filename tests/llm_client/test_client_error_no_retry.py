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

# A request the provider rejects as malformed or unauthorized (400, 401, 403, 404, 422)
# is rejected the same way on every attempt. The application-level retry loops in the
# OpenAI, Anthropic and Gemini clients used to resend it, with an extra error message
# appended to the prompt each time. These tests pin one call per rejected request.

from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
import openai
import pytest
from google.genai import errors as genai_errors

from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.prompts.models import Message

STATUS_ERRORS = [
    (400, 'BadRequestError'),
    (401, 'AuthenticationError'),
    (403, 'PermissionDeniedError'),
    (404, 'NotFoundError'),
    (422, 'UnprocessableEntityError'),
]


def _response(status: int) -> httpx.Response:
    request = httpx.Request('POST', 'https://api.example.com/v1/messages')
    return httpx.Response(status, request=request)


def _messages() -> list[Message]:
    return [Message(role='user', content='Extract the entities.')]


@pytest.mark.asyncio
@pytest.mark.parametrize(('status', 'name'), STATUS_ERRORS)
async def test_openai_client_error_is_sent_once(status, name):
    error = getattr(openai, name)('rejected', response=_response(status), body=None)
    client = OpenAIClient(config=LLMConfig(api_key='test'), client=MagicMock())
    client._generate_response = AsyncMock(side_effect=error)

    with pytest.raises(getattr(openai, name)):
        await client.generate_response(_messages())

    assert client._generate_response.await_count == 1


@pytest.mark.asyncio
async def test_openai_application_error_is_still_retried():
    client = OpenAIClient(config=LLMConfig(api_key='test'), client=MagicMock())
    client._generate_response = AsyncMock(
        side_effect=[ValueError('not valid JSON'), ({'ok': True}, 10, 5)]
    )

    assert await client.generate_response(_messages()) == {'ok': True}
    assert client._generate_response.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(('status', 'name'), STATUS_ERRORS)
async def test_anthropic_client_error_is_sent_once(status, name):
    error = getattr(anthropic, name)('rejected', response=_response(status), body=None)
    mock_async = MagicMock()
    mock_async.messages.create = AsyncMock(side_effect=error)
    with patch('anthropic.AsyncAnthropic', return_value=mock_async):
        client = AnthropicClient(
            config=LLMConfig(api_key='test', model='test-model', max_tokens=1000), cache=False
        )
    client.client = mock_async

    with pytest.raises(getattr(anthropic, name)):
        await client.generate_response(_messages())

    assert mock_async.messages.create.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('code', [400, 401, 403, 404])
async def test_gemini_client_error_is_sent_once(code):
    mock_genai = MagicMock()
    mock_genai.aio.models.generate_content = AsyncMock(
        side_effect=genai_errors.ClientError(code, {'error': {'message': 'rejected'}})
    )
    with patch('google.genai.Client', return_value=mock_genai):
        client = GeminiClient(
            config=LLMConfig(api_key='test', model='test-model', max_tokens=1000), cache=False
        )
    client.client = mock_genai

    caught: BaseException | None = None
    try:
        await client.generate_response(_messages())
    except Exception as exc:  # the client wraps provider errors in a bare Exception
        caught = exc

    assert caught is not None
    assert isinstance(caught.__cause__, genai_errors.ClientError)
    assert mock_genai.aio.models.generate_content.await_count == 1
