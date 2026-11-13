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

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.llm_client import LLMConfig, RateLimitError


def _make_logprob_entry(token: str, logprob: float) -> MagicMock:
    entry = MagicMock()
    entry.token = token
    entry.logprob = logprob
    return entry


def _make_response(top_logprobs: list) -> MagicMock:
    """Build a mock ChatCompletion response with the given top_logprobs list."""
    response = MagicMock()
    content_item = MagicMock()
    content_item.top_logprobs = top_logprobs
    response.choices[0].logprobs.content = [content_item]
    return response


def _make_response_no_logprobs() -> MagicMock:
    """Build a mock response where logprobs is None (triggers empty top_logprobs path)."""
    response = MagicMock()
    response.choices[0].logprobs = None
    return response


@pytest.fixture
def client():
    config = LLMConfig(api_key='test-key', model='gpt-4.1-nano')
    with patch('openai.AsyncOpenAI'):
        c = OpenAIRerankerClient(config=config)
        c.client = MagicMock()
        c.client.chat = MagicMock()
        c.client.chat.completions = MagicMock()
        c.client.chat.completions.create = AsyncMock()
        return c


class TestOpenAIRerankerClientRank:
    @pytest.mark.asyncio
    async def test_rank_returns_sorted_scores(self, client):
        """Passages are returned sorted by descending relevance score."""
        low_logprob = math.log(0.2)   # token=True → score 0.2
        high_logprob = math.log(0.9)  # token=True → score 0.9

        client.client.chat.completions.create.side_effect = [
            _make_response([_make_logprob_entry('True', low_logprob)]),
            _make_response([_make_logprob_entry('True', high_logprob)]),
        ]

        result = await client.rank('query', ['low passage', 'high passage'])

        assert len(result) == 2
        assert result[0][0] == 'high passage'
        assert result[1][0] == 'low passage'
        assert result[0][1] > result[1][1]

    @pytest.mark.asyncio
    async def test_rank_false_token_inverts_score(self, client):
        """When the top token is 'False', score = 1 - exp(logprob)."""
        logprob = math.log(0.8)
        client.client.chat.completions.create.return_value = _make_response(
            [_make_logprob_entry('False', logprob)]
        )

        result = await client.rank('query', ['passage'])

        assert len(result) == 1
        assert pytest.approx(result[0][1], abs=1e-6) == 1 - 0.8

    @pytest.mark.asyncio
    async def test_rank_empty_top_logprobs_appends_zero(self, client):
        """When top_logprobs is empty, score 0.0 is appended and the list stays aligned.

        Regression test for the bug where 'continue' without appending caused a
        length mismatch between scores and passages.
        """
        logprob = math.log(0.9)
        client.client.chat.completions.create.side_effect = [
            _make_response([_make_logprob_entry('True', logprob)]),  # normal passage
            _make_response_no_logprobs(),                             # empty top_logprobs
            _make_response([_make_logprob_entry('True', logprob)]),  # normal passage
        ]

        result = await client.rank('query', ['p1', 'p2', 'p3'])

        # Must return all 3 passages without raising (zip strict=True would fail on mismatch)
        assert len(result) == 3
        scores = {passage: score for passage, score in result}
        assert scores['p2'] == 0.0

    @pytest.mark.asyncio
    async def test_rank_all_empty_logprobs(self, client):
        """All responses with no logprobs → all scores 0.0, no exception."""
        client.client.chat.completions.create.side_effect = [
            _make_response_no_logprobs(),
            _make_response_no_logprobs(),
        ]

        result = await client.rank('query', ['a', 'b'])

        assert len(result) == 2
        assert all(score == 0.0 for _, score in result)

    @pytest.mark.asyncio
    async def test_rank_empty_passages(self, client):
        result = await client.rank('query', [])
        assert result == []

    @pytest.mark.asyncio
    async def test_rank_rate_limit_error_is_reraised(self, client):
        import openai as _openai
        client.client.chat.completions.create.side_effect = _openai.RateLimitError(
            message='rate limit', response=MagicMock(), body={}
        )

        with pytest.raises(RateLimitError):
            await client.rank('query', ['passage'])
