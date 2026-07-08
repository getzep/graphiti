from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_base_client import DEFAULT_MODEL, DEFAULT_REASONING
from graphiti_core.llm_client.openai_client import OpenAIClient


class DummyResponses:
    def __init__(self):
        self.parse_calls: list[dict] = []

    async def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        return SimpleNamespace(output_text='{}')


class DummyChatCompletions:
    def __init__(self):
        self.create_calls: list[dict] = []

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        message = SimpleNamespace(content='{}')
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class DummyChat:
    def __init__(self):
        self.completions = DummyChatCompletions()


class DummyOpenAIClient:
    def __init__(self):
        self.responses = DummyResponses()
        self.chat = DummyChat()


class DummyResponseModel(BaseModel):
    foo: str


def test_default_model_and_reasoning_sentinel():
    assert DEFAULT_MODEL == 'gpt-5.5'
    assert DEFAULT_REASONING == 'auto'


@pytest.mark.parametrize(
    ('model', 'reasoning', 'expected'),
    [
        # gpt-5.5 -> reasoning off; the dated snapshot must match the prefix too
        ('gpt-5.5', 'auto', 'none'),
        ('gpt-5.5-2026-04-23', 'auto', 'none'),
        # every other model: 'minimal' (cheapest tier, prior default) — NOT None,
        # so non-gpt-5.5 reasoning models don't regress to the API's medium default
        ('gpt-5', 'auto', 'minimal'),
        ('gpt-5.4-mini', 'auto', 'minimal'),
        ('o1', 'auto', 'minimal'),
        # non-reasoning model: value is ignored by the caller (not a reasoning model)
        ('gpt-4.1', 'auto', 'minimal'),
        # explicit values always pass through unchanged
        ('gpt-5.5', 'high', 'high'),
        ('gpt-5', 'low', 'low'),
        ('gpt-5.5', None, None),
    ],
)
def test_resolve_reasoning_effort(model, reasoning, expected):
    assert OpenAIClient._resolve_reasoning_effort(model, reasoning) == expected


def test_default_model_for_size_is_gpt_5_5():
    client = OpenAIClient(config=LLMConfig(), client=DummyOpenAIClient())
    from graphiti_core.llm_client.config import ModelSize

    assert client._get_model_for_size(ModelSize.medium) == 'gpt-5.5'


@pytest.mark.asyncio
async def test_gpt_5_5_structured_completion_sends_none_effort_and_no_temperature():
    dummy = DummyOpenAIClient()
    client = OpenAIClient(config=LLMConfig(), client=dummy)

    await client._create_structured_completion(
        model='gpt-5.5',
        messages=[],
        temperature=0.5,
        max_tokens=64,
        response_model=DummyResponseModel,
        reasoning=DEFAULT_REASONING,  # 'auto'
        verbosity='low',
    )

    assert len(dummy.responses.parse_calls) == 1
    call_args = dummy.responses.parse_calls[0]
    assert call_args['model'] == 'gpt-5.5'
    # Reasoning models do not accept temperature
    assert 'temperature' not in call_args
    # 'auto' resolves to 'none' for gpt-5.5
    assert call_args['reasoning'] == {'effort': 'none'}
    assert call_args['text'] == {'verbosity': 'low'}


@pytest.mark.asyncio
async def test_explicit_reasoning_overrides_auto():
    dummy = DummyOpenAIClient()
    client = OpenAIClient(config=LLMConfig(), client=dummy)

    await client._create_structured_completion(
        model='gpt-5.5',
        messages=[],
        temperature=None,
        max_tokens=64,
        response_model=DummyResponseModel,
        reasoning='high',
        verbosity='low',
    )

    call_args = dummy.responses.parse_calls[0]
    assert call_args['reasoning'] == {'effort': 'high'}


@pytest.mark.asyncio
async def test_non_reasoning_model_omits_reasoning_and_keeps_temperature():
    dummy = DummyOpenAIClient()
    client = OpenAIClient(config=LLMConfig(), client=dummy)

    await client._create_structured_completion(
        model='gpt-4.1',
        messages=[],
        temperature=0.4,
        max_tokens=64,
        response_model=DummyResponseModel,
        reasoning=DEFAULT_REASONING,
        verbosity='low',
    )

    call_args = dummy.responses.parse_calls[0]
    assert call_args.get('temperature') == 0.4
    assert 'reasoning' not in call_args
    assert 'text' not in call_args


@pytest.mark.asyncio
async def test_empty_string_reasoning_is_not_sent():
    # A stray reasoning='' must not produce an invalid {'effort': ''} on the wire.
    dummy = DummyOpenAIClient()
    client = OpenAIClient(config=LLMConfig(), client=dummy)

    await client._create_structured_completion(
        model='gpt-5.5',
        messages=[],
        temperature=None,
        max_tokens=64,
        response_model=DummyResponseModel,
        reasoning='',
        verbosity='low',
    )

    call_args = dummy.responses.parse_calls[0]
    assert 'reasoning' not in call_args


@pytest.mark.asyncio
async def test_non_5_5_reasoning_model_defaults_to_minimal_not_api_default():
    # Regression guard: non-gpt-5.5 reasoning models must keep the cheap 'minimal'
    # effort (the prior default), not fall through to the API's medium default.
    dummy = DummyOpenAIClient()
    client = OpenAIClient(config=LLMConfig(), client=dummy)

    await client._create_structured_completion(
        model='gpt-5',
        messages=[],
        temperature=0.5,
        max_tokens=64,
        response_model=DummyResponseModel,
        reasoning=DEFAULT_REASONING,  # 'auto'
        verbosity='low',
    )

    call_args = dummy.responses.parse_calls[0]
    assert call_args['reasoning'] == {'effort': 'minimal'}
    assert 'temperature' not in call_args


@pytest.mark.asyncio
async def test_create_completion_omits_temperature_for_reasoning_model():
    # The JSON-object fallback path must not send temperature (not even None) to
    # a reasoning model.
    dummy = DummyOpenAIClient()
    client = OpenAIClient(config=LLMConfig(), client=dummy)

    await client._create_completion(
        model='gpt-5.5',
        messages=[],
        temperature=0.5,
        max_tokens=64,
    )

    assert len(dummy.chat.completions.create_calls) == 1
    call_args = dummy.chat.completions.create_calls[0]
    assert 'temperature' not in call_args
    assert call_args['response_format'] == {'type': 'json_object'}


@pytest.mark.asyncio
async def test_create_completion_keeps_temperature_for_non_reasoning_model():
    dummy = DummyOpenAIClient()
    client = OpenAIClient(config=LLMConfig(), client=dummy)

    await client._create_completion(
        model='gpt-4.1',
        messages=[],
        temperature=0.4,
        max_tokens=64,
    )

    call_args = dummy.chat.completions.create_calls[0]
    assert call_args['temperature'] == 0.4
