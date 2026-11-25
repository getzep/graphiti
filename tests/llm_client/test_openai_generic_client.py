import json
from types import SimpleNamespace

import openai
import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.errors import EmptyResponseError, RateLimitError
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message


class DummyChatCompletions:
    def __init__(self, content: str = '{}', error: Exception | None = None):
        self.create_calls: list[dict] = []
        self._content = content
        self._error = error

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if self._error is not None:
            raise self._error
        message = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class DummyChat:
    def __init__(self, completions: DummyChatCompletions):
        self.completions = completions


class DummyClient:
    def __init__(self, completions: DummyChatCompletions):
        self.chat = DummyChat(completions)


class ResponseModel(BaseModel):
    foo: str


def _messages() -> list[Message]:
    return [
        Message(role='system', content='system message'),
        Message(role='user', content='user message'),
    ]


def _make_client(content: str = '{"foo": "bar"}', error: Exception | None = None, **kwargs):
    completions = DummyChatCompletions(content=content, error=error)
    client = OpenAIGenericClient(
        config=LLMConfig(api_key='test', model='test-model'),
        client=DummyClient(completions),
        **kwargs,
    )
    return client, completions


@pytest.mark.asyncio
async def test_defaults_to_json_schema_response_format():
    client, completions = _make_client()

    await client.generate_response(_messages(), response_model=ResponseModel)

    response_format = completions.create_calls[0]['response_format']
    assert response_format['type'] == 'json_schema'
    assert response_format['json_schema']['name'] == 'ResponseModel'
    assert response_format['json_schema']['schema'] == ResponseModel.model_json_schema()


@pytest.mark.asyncio
async def test_json_schema_mode_does_not_inject_schema_into_prompt():
    client, completions = _make_client()
    messages = _messages()

    await client.generate_response(messages, response_model=ResponseModel)

    sent_user_content = completions.create_calls[0]['messages'][-1]['content']
    assert 'Respond with a JSON object in the following format' not in sent_user_content


@pytest.mark.asyncio
async def test_json_object_mode_uses_json_object_and_injects_schema():
    client, completions = _make_client(structured_output_mode='json_object')

    await client.generate_response(_messages(), response_model=ResponseModel)

    call = completions.create_calls[0]
    assert call['response_format'] == {'type': 'json_object'}
    # The schema must be injected into the prompt since the API will not enforce it.
    sent_user_content = call['messages'][-1]['content']
    assert 'Respond with a JSON object in the following format' in sent_user_content
    assert json.dumps(ResponseModel.model_json_schema()) in sent_user_content


@pytest.mark.asyncio
async def test_no_response_model_uses_json_object_without_injection():
    client, completions = _make_client(content='{"any": "thing"}')

    result = await client.generate_response(_messages())

    call = completions.create_calls[0]
    assert call['response_format'] == {'type': 'json_object'}
    assert (
        'Respond with a JSON object in the following format' not in call['messages'][-1]['content']
    )
    assert result == {'any': 'thing'}


@pytest.mark.asyncio
async def test_rate_limit_error_is_translated():
    rate_limit = openai.RateLimitError(
        message='slow down',
        response=SimpleNamespace(status_code=429, headers={}, request=None),
        body=None,
    )
    client, _ = _make_client(error=rate_limit)

    # Assert translation at the _generate_response level. Going through generate_response
    # would invoke the inherited tenacity retry wrapper (RateLimitError is retryable), which
    # adds real backoff sleeps and would make this unit test slow.
    with pytest.raises(RateLimitError):
        await client._generate_response(_messages(), response_model=ResponseModel)


@pytest.mark.asyncio
async def test_empty_content_raises_empty_response_error():
    # Empty body (flaky endpoint / refusal / length cutoff) must raise a clear error,
    # not a cryptic JSONDecodeError from json.loads(''). Asserted at _generate_response
    # level: EmptyResponseError is retryable, so generate_response would invoke the real
    # backoff retry and slow this unit test.
    client, _ = _make_client(content='')

    with pytest.raises(EmptyResponseError):
        await client._generate_response(_messages(), response_model=ResponseModel)


def test_empty_response_error_is_retryable():
    # An empty body is treated as a transient provider hiccup (common on local/compatible
    # endpoints), so the base retry wrapper retries it rather than failing on first try.
    from graphiti_core.llm_client.client import is_server_or_retry_error

    assert is_server_or_retry_error(EmptyResponseError('empty')) is True


@pytest.mark.asyncio
async def test_strips_markdown_code_fence_before_parsing():
    # Local/compatible models (e.g. Ollama/gemma) often wrap JSON in a ```json fence even
    # under a structured response_format; the client must strip it before json.loads.
    fenced = '```json\n{"foo": "bar"}\n```'
    client, _ = _make_client(content=fenced)

    result = await client.generate_response(_messages(), response_model=ResponseModel)

    assert result == {'foo': 'bar'}


@pytest.mark.asyncio
async def test_non_retryable_error_is_not_retried():
    # The old hand-rolled re-prompt loop is gone. Retry is now delegated to the base
    # tenacity wrapper, which only retries transient errors (RateLimitError /
    # JSONDecodeError). A non-retryable error (e.g. ValueError) propagates after a
    # single create call.
    client, completions = _make_client(error=ValueError('bad response'))

    with pytest.raises(ValueError):
        await client.generate_response(_messages(), response_model=ResponseModel)

    assert len(completions.create_calls) == 1
