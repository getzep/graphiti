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
