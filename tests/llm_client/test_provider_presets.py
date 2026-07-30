"""Tests for the OpenAI-compatible provider presets (DeepSeek, Kimi, Qwen).

Each is a thin preset over OpenAIGenericClient that only fills in a default
base_url / model and defaults to the json_object structured-output fallback; the
request/parse behavior is inherited and covered by test_openai_generic_client.
"""

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.deepseek_client import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DeepSeekClient,
)
from graphiti_core.llm_client.kimi_client import (
    DEFAULT_KIMI_BASE_URL,
    DEFAULT_KIMI_MODEL,
    KimiClient,
)
from graphiti_core.llm_client.qwen_client import (
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_QWEN_MODEL,
    QwenClient,
)
from graphiti_core.prompts.models import Message

PRESETS = [
    pytest.param(DeepSeekClient, DEFAULT_DEEPSEEK_BASE_URL, DEFAULT_DEEPSEEK_MODEL, id='deepseek'),
    pytest.param(KimiClient, DEFAULT_KIMI_BASE_URL, DEFAULT_KIMI_MODEL, id='kimi'),
    pytest.param(QwenClient, DEFAULT_QWEN_BASE_URL, DEFAULT_QWEN_MODEL, id='qwen'),
]


class DummyChatCompletions:
    def __init__(self, content: str = '{"foo": "bar"}'):
        self.create_calls: list[dict] = []
        self._content = content

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class DummyClient:
    def __init__(self, completions: DummyChatCompletions):
        self.chat = SimpleNamespace(completions=completions)


class ResponseModel(BaseModel):
    foo: str


def _messages() -> list[Message]:
    return [
        Message(role='system', content='system message'),
        Message(role='user', content='user message'),
    ]


@pytest.mark.parametrize(('client_cls', 'base_url', 'model'), PRESETS)
def test_fills_default_endpoint_and_model(client_cls, base_url, model):
    # No client injected -> a real AsyncOpenAI is built from the filled config.
    client = client_cls(config=LLMConfig(api_key='test'))
    # httpx normalizes the base_url with a trailing slash.
    assert str(client.client.base_url).rstrip('/') == base_url.rstrip('/')
    assert client.model == model


@pytest.mark.parametrize(('client_cls', 'base_url', 'model'), PRESETS)
def test_preserves_explicit_config(client_cls, base_url, model):
    client = client_cls(config=LLMConfig(api_key='k', base_url='http://custom:9/v1', model='m'))
    assert str(client.client.base_url).startswith('http://custom:9')
    assert client.model == 'm'


@pytest.mark.parametrize(('client_cls', 'base_url', 'model'), PRESETS)
@pytest.mark.asyncio
async def test_defaults_to_json_object_fallback(client_cls, base_url, model):
    completions = DummyChatCompletions()
    client = client_cls(config=LLMConfig(api_key='test'), client=DummyClient(completions))

    await client.generate_response(_messages(), response_model=ResponseModel)

    call = completions.create_calls[0]
    assert call['response_format'] == {'type': 'json_object'}
    injected = call['messages'][-1]['content']
    assert 'Respond with a JSON object in the following format' in injected
    assert json.dumps(ResponseModel.model_json_schema()) in injected


@pytest.mark.parametrize(('client_cls', 'base_url', 'model'), PRESETS)
@pytest.mark.asyncio
async def test_parses_response_body(client_cls, base_url, model):
    completions = DummyChatCompletions(content='{"foo": "baz"}')
    client = client_cls(config=LLMConfig(api_key='test'), client=DummyClient(completions))

    result = await client.generate_response(_messages(), response_model=ResponseModel)

    assert result == {'foo': 'baz'}
