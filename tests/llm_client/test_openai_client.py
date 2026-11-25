from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from graphiti_core.cross_encoder.openai_reranker_client import (
    DEFAULT_MODEL as DEFAULT_RERANKER_MODEL,
)
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.openai_base_client import (
    DEFAULT_MODEL,
    DEFAULT_REASONING,
    BaseOpenAIClient,
)
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.prompts.models import Message


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
    assert DEFAULT_MODEL == 'gpt-6-luna'
    assert DEFAULT_REASONING == 'auto'


@pytest.mark.parametrize(
    ('model', 'reasoning', 'expected'),
    [
        # GPT 5.1+ and later major versions use 'none'.
        ('gpt-5.1', 'auto', 'none'),
        ('gpt-5.4-mini', 'auto', 'none'),
        ('gpt-5.5', 'auto', 'none'),
        ('gpt-5.5-2026-04-23', 'auto', 'none'),
        ('gpt-5.6-luna', 'auto', 'none'),
        ('gpt-6-luna', 'auto', 'none'),
        ('gpt-6.1-sol', 'auto', 'none'),
        # The original GPT-5 family and o-series models use 'minimal'.
        ('gpt-5', 'auto', 'minimal'),
        ('gpt-5-mini', 'auto', 'minimal'),
        ('gpt-5-nano', 'auto', 'minimal'),
        ('gpt-5-2025-08-07', 'auto', 'minimal'),
        ('o3', 'auto', 'minimal'),
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


@pytest.mark.parametrize(
    ('model', 'expected'),
    [
        ('gpt-5', True),
        ('gpt-5-nano', True),
        ('gpt-5.5', True),
        ('gpt-6-luna', True),
        ('gpt-6.1-sol', True),
        ('o1', True),
        ('o3-mini', True),
        ('gpt-4.1-mini', False),
        ('gpt-4o-mini', False),
    ],
)
def test_is_reasoning_model(model, expected):
    assert BaseOpenAIClient._is_reasoning_model(model) is expected


def test_default_model_for_size_is_gpt_6_luna():
    client = OpenAIClient(config=LLMConfig(), client=DummyOpenAIClient())

    assert client._get_model_for_size(ModelSize.medium) == 'gpt-6-luna'
    assert client._get_model_for_size(ModelSize.small) == 'gpt-6-luna'


def test_default_reranker_model_is_gpt_4_1_mini():
    assert DEFAULT_RERANKER_MODEL == 'gpt-4.1-mini'


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
    assert 'extra_body' not in call_args


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('model', 'expected_effort'),
    [
        ('gpt-6-luna', 'none'),
        ('gpt-5.5', 'none'),
        ('gpt-4.1-mini', None),
    ],
)
async def test_structured_completion_uses_reasoning_fields_for_model(model, expected_effort):
    dummy = DummyOpenAIClient()
    client = OpenAIClient(config=LLMConfig(), client=dummy)

    await client._create_structured_completion(
        model=model,
        messages=[],
        temperature=0.5,
        max_tokens=64,
        response_model=DummyResponseModel,
        reasoning='auto',
    )

    call_args = dummy.responses.parse_calls[0]
    if expected_effort is None:
        assert 'reasoning' not in call_args
        assert call_args['temperature'] == 0.5
        assert 'extra_body' not in call_args
    else:
        assert call_args['reasoning'] == {'effort': expected_effort}
        assert 'temperature' not in call_args
        if model == 'gpt-6-luna':
            assert call_args['extra_body'] == {'prompt_cache_options': {'mode': 'explicit'}}
        else:
            assert 'extra_body' not in call_args


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
async def test_original_gpt_5_family_defaults_to_minimal():
    # Regression guard: original GPT-5 models keep the 'minimal' effort.
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
@pytest.mark.parametrize(
    ('model', 'expected_token_limit', 'expected_extra_body'),
    [
        ('gpt-6-luna', 'max_completion_tokens', {'prompt_cache_options': {'mode': 'explicit'}}),
        ('gpt-5.5', 'max_tokens', None),
        ('gpt-4.1-mini', 'max_tokens', None),
    ],
)
async def test_create_completion_model_specific_options(
    model, expected_token_limit, expected_extra_body
):
    dummy = DummyOpenAIClient()
    client = OpenAIClient(config=LLMConfig(), client=dummy)

    await client._create_completion(
        model=model,
        messages=[],
        temperature=0.5,
        max_tokens=64,
    )

    assert len(dummy.chat.completions.create_calls) == 1
    call_args = dummy.chat.completions.create_calls[0]
    if model == 'gpt-4.1-mini':
        assert call_args['temperature'] == 0.5
    else:
        assert 'temperature' not in call_args
    assert call_args['response_format'] == {'type': 'json_object'}
    assert expected_token_limit in call_args
    other_token_limit = (
        'max_tokens' if expected_token_limit == 'max_completion_tokens' else 'max_completion_tokens'
    )
    assert other_token_limit not in call_args
    if expected_extra_body is None:
        assert 'extra_body' not in call_args
    else:
        assert call_args['extra_body'] == expected_extra_body


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


@pytest.mark.asyncio
async def test_generate_response_does_not_mutate_caller_messages():
    dummy = DummyOpenAIClient()
    client = OpenAIClient(config=LLMConfig(), client=dummy)
    messages = [
        Message(role='system', content='System message'),
        Message(role='user', content='User message\x00'),
    ]
    original = [message.model_dump() for message in messages]

    await client.generate_response(
        messages,
        group_id='test-group',
        attribute_extraction=True,
    )

    assert [message.model_dump() for message in messages] == original
    call_args = dummy.chat.completions.create_calls[0]
    assert 'ATTRIBUTE EXTRACTION:' in call_args['messages'][0]['content']
    assert 'same language' in call_args['messages'][0]['content']
    assert '\x00' not in call_args['messages'][-1]['content']
