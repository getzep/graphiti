import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.ollama_client import (
    DEFAULT_OLLAMA_API_KEY,
    DEFAULT_OLLAMA_BASE_URL,
    OllamaClient,
)
from graphiti_core.prompts.models import Message


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


class TestOllamaDefaults:
    def test_fills_local_endpoint_and_placeholder_key(self):
        # No client injected -> a real AsyncOpenAI is built from the filled config.
        client = OllamaClient(config=LLMConfig())
        assert str(client.client.base_url).startswith('http://localhost:11434')
        assert client.client.api_key == DEFAULT_OLLAMA_API_KEY

    def test_preserves_explicit_base_url_and_key(self):
        client = OllamaClient(
            config=LLMConfig(base_url='http://remote-ollama:1234/v1', api_key='secret')
        )
        assert str(client.client.base_url).startswith('http://remote-ollama:1234')
        assert client.client.api_key == 'secret'

    def test_default_base_url_constant(self):
        assert DEFAULT_OLLAMA_BASE_URL == 'http://localhost:11434/v1'


@pytest.mark.asyncio
class TestOllamaStructuredOutput:
    async def test_defaults_to_json_object_fallback_with_schema_injection(self):
        completions = DummyChatCompletions()
        client = OllamaClient(config=LLMConfig(model='llama3.1'), client=DummyClient(completions))

        await client.generate_response(_messages(), response_model=ResponseModel)

        call = completions.create_calls[0]
        # Local models default to the json_object fallback (schema not API-enforced).
        assert call['response_format'] == {'type': 'json_object'}
        injected = call['messages'][-1]['content']
        assert 'Respond with a JSON object in the following format' in injected
        assert json.dumps(ResponseModel.model_json_schema()) in injected

    async def test_can_opt_into_native_json_schema(self):
        completions = DummyChatCompletions()
        client = OllamaClient(
            config=LLMConfig(model='llama3.1'),
            client=DummyClient(completions),
            structured_output_mode='json_schema',
        )

        await client.generate_response(_messages(), response_model=ResponseModel)

        call = completions.create_calls[0]
        assert call['response_format']['type'] == 'json_schema'
        assert (
            'Respond with a JSON object in the following format'
            not in call['messages'][-1]['content']
        )

    async def test_parses_response_body(self):
        completions = DummyChatCompletions(content='{"foo": "baz"}')
        client = OllamaClient(config=LLMConfig(model='llama3.1'), client=DummyClient(completions))

        result = await client.generate_response(_messages(), response_model=ResponseModel)

        assert result == {'foo': 'baz'}
