from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
from graphiti_core.llm_client.config import LLMConfig


class DummyResponses:
    def __init__(self):
        self.parse_calls: list[dict] = []

    async def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        return SimpleNamespace(output_text='{}')


class DummyChatCompletions:
    def __init__(self):
        self.create_calls: list[dict] = []
        self.parse_calls: list[dict] = []

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        message = SimpleNamespace(content='{}')
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    async def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        parsed_model = kwargs.get('response_format')
        message = SimpleNamespace(parsed=parsed_model(foo='bar'))
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


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


class DummyResponseModel(BaseModel):
    foo: str


@pytest.mark.asyncio
async def test_structured_completion_strips_reasoning_for_unsupported_models():
    dummy_client = DummyAzureClient()
    client = AzureOpenAILLMClient(
        azure_client=dummy_client,
        config=LLMConfig(),
        reasoning='minimal',
        verbosity='low',
    )

    await client._create_structured_completion(
        model='gpt-4.1',
        messages=[],
        temperature=0.4,
        max_tokens=64,
        response_model=DummyResponseModel,
        reasoning='minimal',
        verbosity='low',
    )

    # For non-reasoning models, uses beta.chat.completions.parse
    assert len(dummy_client.beta.chat.completions.parse_calls) == 1
    call_args = dummy_client.beta.chat.completions.parse_calls[0]
    assert call_args['model'] == 'gpt-4.1'
    assert call_args['messages'] == []
    assert call_args['max_tokens'] == 64
    assert call_args['response_format'] is DummyResponseModel
    assert call_args['temperature'] == 0.4
    # Reasoning and verbosity parameters should not be passed for non-reasoning models
    assert 'reasoning' not in call_args
    assert 'verbosity' not in call_args
    assert 'text' not in call_args


@pytest.mark.asyncio
async def test_reasoning_fields_forwarded_for_supported_models():
    dummy_client = DummyAzureClient()
    client = AzureOpenAILLMClient(
        azure_client=dummy_client,
        config=LLMConfig(),
        reasoning='intense',
        verbosity='high',
    )

    await client._create_structured_completion(
        model='o1-custom',
        messages=[],
        temperature=0.7,
        max_tokens=128,
        response_model=DummyResponseModel,
        reasoning='intense',
        verbosity='high',
    )

    call_args = dummy_client.responses.parse_calls[0]
    assert 'temperature' not in call_args
    assert call_args['reasoning'] == {'effort': 'intense'}
    assert call_args['text'] == {'verbosity': 'high'}

    await client._create_completion(
        model='o1-custom',
        messages=[],
        temperature=0.7,
        max_tokens=128,
    )

    create_args = dummy_client.chat.completions.create_calls[0]
    assert 'temperature' not in create_args


def test_handle_structured_response_returns_token_triple_for_parsed_chat_completion():
    """``_handle_structured_response`` must return ``(dict, input_tokens, output_tokens)``
    to match ``BaseOpenAIClient._generate_response``'s unpack site (#1298). The
    ``beta.chat.completions.parse`` path exposes ``prompt_tokens``/``completion_tokens``.
    """
    client = AzureOpenAILLMClient.__new__(AzureOpenAILLMClient)

    class ParsedModel:
        @staticmethod
        def model_dump():
            return {'foo': 'bar'}

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=ParsedModel()))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
    )

    parsed, input_tokens, output_tokens = client._handle_structured_response(response)
    assert parsed == {'foo': 'bar'}
    assert input_tokens == 10
    assert output_tokens == 20


def test_handle_structured_response_returns_token_triple_for_responses_parse():
    """The reasoning-model path (``responses.parse``) exposes token usage as
    ``input_tokens``/``output_tokens`` and must also return the 3-tuple (#1298)."""
    client = AzureOpenAILLMClient.__new__(AzureOpenAILLMClient)

    response = SimpleNamespace(
        output_text='{"foo": "bar"}',
        usage=SimpleNamespace(input_tokens=15, output_tokens=25),
    )

    parsed, input_tokens, output_tokens = client._handle_structured_response(response)
    assert parsed == {'foo': 'bar'}
    assert input_tokens == 15
    assert output_tokens == 25


def test_handle_structured_response_tolerates_missing_usage():
    """When the upstream body omits ``usage`` (some empty/error responses do),
    token counts fall back to 0 instead of raising."""
    client = AzureOpenAILLMClient.__new__(AzureOpenAILLMClient)

    response = SimpleNamespace(output_text='{"foo": "bar"}')

    parsed, input_tokens, output_tokens = client._handle_structured_response(response)
    assert parsed == {'foo': 'bar'}
    assert input_tokens == 0
    assert output_tokens == 0
