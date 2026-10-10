"""Regression tests for issue #1496.

Two independent defects in ``AzureOpenAILLMClient``:

1. ``_handle_structured_response`` returns a bare ``dict`` while the
   ``BaseOpenAIClient`` contract (and ``_generate_response``'s own return
   annotation) requires ``tuple[dict, int, int]``.  Every structured Azure
   call therefore raises ``ValueError: not enough values to unpack``.
2. The non-reasoning request paths send ``max_tokens``, which Azure OpenAI
   rejects on API version 2024-10-21 and later in favour of
   ``max_completion_tokens``.
"""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.prompts.models import Message


class DummyResponseModel(BaseModel):
    foo: str


class DummyResponses:
    def __init__(self):
        self.parse_calls: list[dict] = []

    async def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        return SimpleNamespace(
            output_text='{"foo": "bar"}',
            usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        )


class DummyChatCompletions:
    def __init__(self):
        self.create_calls: list[dict] = []
        self.parse_calls: list[dict] = []

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        message = SimpleNamespace(content='{"foo": "bar"}')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),
        )

    async def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        parsed_model = kwargs.get('response_format')
        message = SimpleNamespace(parsed=parsed_model(foo='bar'))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=13, completion_tokens=17),
        )


class DummyChat:
    def __init__(self):
        self.completions = DummyChatCompletions()


class DummyBeta:
    def __init__(self):
        self.chat = DummyChat()


class DummyAzureClient:
    def __init__(self):
        self.responses = DummyResponses()
        self.chat = DummyChat()
        self.beta = DummyBeta()


def _client(dummy: DummyAzureClient) -> AzureOpenAILLMClient:
    return AzureOpenAILLMClient(azure_client=dummy, config=LLMConfig(api_key='test'))


# --- Defect 1: return-type contract violation -------------------------------


@pytest.mark.asyncio
async def test_handle_structured_response_returns_usage_tuple_for_parsed_completion():
    dummy = DummyAzureClient()
    client = _client(dummy)

    response = await dummy.beta.chat.completions.parse(
        model='gpt-4o', messages=[], response_format=DummyResponseModel
    )
    result = client._handle_structured_response(response)

    assert isinstance(result, tuple), (
        'Azure _handle_structured_response must honour the BaseOpenAIClient '
        'tuple[dict, int, int] contract'
    )
    parsed, input_tokens, output_tokens = result
    assert parsed == {'foo': 'bar'}
    assert input_tokens == 13
    assert output_tokens == 17


@pytest.mark.asyncio
async def test_handle_structured_response_returns_usage_tuple_for_responses_api():
    dummy = DummyAzureClient()
    client = _client(dummy)

    response = await dummy.responses.parse(model='gpt-5', input=[])
    result = client._handle_structured_response(response)

    assert isinstance(result, tuple)
    parsed, input_tokens, output_tokens = result
    assert parsed == {'foo': 'bar'}
    assert input_tokens == 11
    assert output_tokens == 7


@pytest.mark.asyncio
async def test_generate_response_does_not_raise_unpack_error():
    """End-to-end repro of the reported 'expected 3, got 1' crash."""
    dummy = DummyAzureClient()
    client = _client(dummy)
    client.model = 'gpt-4o'

    parsed = await client.generate_response(
        messages=[Message(role='user', content='hello')],
        response_model=DummyResponseModel,
    )

    assert parsed == {'foo': 'bar'}


@pytest.mark.asyncio
async def test_handle_structured_response_missing_usage_defaults_to_zero():
    dummy = DummyAzureClient()
    client = _client(dummy)

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                parsed=None, message=SimpleNamespace(parsed=DummyResponseModel(foo='bar'))
            )
        ]
    )
    parsed, input_tokens, output_tokens = client._handle_structured_response(response)
    assert parsed == {'foo': 'bar'}
    assert (input_tokens, output_tokens) == (0, 0)


# --- Defect 2: max_tokens rejected by Azure 2024-10-21+ ---------------------


@pytest.mark.asyncio
async def test_structured_completion_uses_max_completion_tokens():
    dummy = DummyAzureClient()
    client = _client(dummy)

    await client._create_structured_completion(
        model='gpt-4o',
        messages=[],
        temperature=0.4,
        max_tokens=64,
        response_model=DummyResponseModel,
        reasoning=None,
        verbosity=None,
    )

    call_args = dummy.beta.chat.completions.parse_calls[0]
    assert 'max_tokens' not in call_args, (
        "Azure OpenAI 2024-10-21+ rejects 'max_tokens'; use 'max_completion_tokens'"
    )
    assert call_args['max_completion_tokens'] == 64


@pytest.mark.asyncio
async def test_json_completion_uses_max_completion_tokens():
    dummy = DummyAzureClient()
    client = _client(dummy)

    await client._create_completion(
        model='gpt-4o',
        messages=[],
        temperature=0.4,
        max_tokens=128,
    )

    call_args = dummy.chat.completions.create_calls[0]
    assert 'max_tokens' not in call_args
    assert call_args['max_completion_tokens'] == 128
