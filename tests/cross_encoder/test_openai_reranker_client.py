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

# Running tests: pytest -xvs tests/cross_encoder/test_openai_reranker_client.py

from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import AsyncOpenAI

from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient


def make_config() -> LLMConfig:
    return LLMConfig(api_key='test-key', model='test-model', base_url='http://localhost:1/v1')


def make_response(token: str, logprob: float) -> MagicMock:
    """Build a chat completion response whose first token has one top logprob."""
    top_logprob = MagicMock()
    top_logprob.token = token
    top_logprob.logprob = logprob
    response = MagicMock()
    response.choices[0].logprobs.content[0].top_logprobs = [top_logprob]
    return response


class TestOpenAIRerankerClientInitialization:
    """The reranker must reach the AsyncOpenAI client inside any Graphiti LLM client."""

    @pytest.mark.parametrize(
        'wrap',
        [
            lambda inner: OpenAIClient(config=make_config(), client=inner),
            lambda inner: OpenAIGenericClient(config=make_config(), client=inner),
            lambda inner: AzureOpenAILLMClient(azure_client=inner, config=make_config()),
        ],
        ids=['OpenAIClient', 'OpenAIGenericClient', 'AzureOpenAILLMClient'],
    )
    def test_init_unwraps_graphiti_llm_clients(self, wrap):
        inner = AsyncOpenAI(api_key='test-key', base_url='http://localhost:1/v1')

        reranker = OpenAIRerankerClient(client=wrap(inner), config=make_config())

        assert reranker.client is inner

    def test_init_keeps_async_openai_client(self):
        inner = AsyncOpenAI(api_key='test-key', base_url='http://localhost:1/v1')

        reranker = OpenAIRerankerClient(client=inner, config=make_config())

        assert reranker.client is inner


class TestOpenAIRerankerClientRank:
    """rank() must work when the reranker is built the way the README shows for local models."""

    async def test_rank_with_generic_client_uses_wrapped_client(self):
        inner = MagicMock()

        async def fake_create(**kwargs):
            # The relevant passage answers "True"; the other answers "False".
            if 'Austin' in str(kwargs['messages']):
                return make_response('True', -0.1)
            return make_response('False', -0.2)

        inner.chat.completions.create = AsyncMock(side_effect=fake_create)
        llm_client = OpenAIGenericClient(config=make_config(), client=inner)
        reranker = OpenAIRerankerClient(client=llm_client, config=make_config())

        results = await reranker.rank(
            'Where does Kendra live now?',
            ['The Eiffel Tower is in Paris.', 'Kendra moved to Austin in March 2026.'],
        )

        assert inner.chat.completions.create.await_count == 2
        assert results[0][0] == 'Kendra moved to Austin in March 2026.'
        assert results[0][1] > results[1][1]
