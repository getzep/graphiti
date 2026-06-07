from types import SimpleNamespace

import pytest

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient


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
        self.chat = DummyChat()


@pytest.mark.asyncio
async def test_reasoning_completion_omits_temperature_key():
    dummy_client = DummyOpenAIClient()
    client = OpenAIClient(client=dummy_client, config=LLMConfig())

    await client._create_completion(
        model='gpt-5-mini',
        messages=[],
        temperature=0.7,
        max_tokens=128,
    )

    create_args = dummy_client.chat.completions.create_calls[0]
    assert 'temperature' not in create_args


@pytest.mark.asyncio
async def test_non_reasoning_completion_keeps_temperature_key():
    dummy_client = DummyOpenAIClient()
    client = OpenAIClient(client=dummy_client, config=LLMConfig())

    await client._create_completion(
        model='gpt-4.1-mini',
        messages=[],
        temperature=0.4,
        max_tokens=64,
    )

    create_args = dummy_client.chat.completions.create_calls[0]
    assert create_args['temperature'] == 0.4
