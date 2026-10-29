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

from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.llm_client import LLMConfig


def _scored_response(token: str, logprob: float) -> MagicMock:
    """A chat-completion response carrying a single top-logprob token."""
    tl = MagicMock()
    tl.token = token
    tl.logprob = logprob
    content_item = MagicMock()
    content_item.top_logprobs = [tl]
    logprobs = MagicMock()
    logprobs.content = [content_item]
    choice = MagicMock()
    choice.logprobs = logprobs
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _empty_logprobs_response() -> MagicMock:
    """A response with no logprobs (e.g. an empty/truncated completion)."""
    choice = MagicMock()
    choice.logprobs = None
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_reranker(responses: list) -> OpenAIRerankerClient:
    reranker = OpenAIRerankerClient(config=LLMConfig(api_key='test', model='m'))
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=responses)
    reranker.client = mock_client
    return reranker


@pytest.mark.asyncio
async def test_rank_skips_passage_with_empty_logprobs_without_crashing() -> None:
    """A response missing top_logprobs must be skipped gracefully, not crash the
    whole rank() via a passage/score length mismatch in a strict zip."""
    reranker = _make_reranker(
        [
            _scored_response('true', -0.1),
            _empty_logprobs_response(),  # previously desynced scores -> strict-zip ValueError
            _scored_response('false', -0.2),
        ]
    )

    results = await reranker.rank('query', ['p0', 'p1', 'p2'])

    assert {p for p, _ in results} == {'p0', 'p2'}
    assert all(0.0 <= s <= 1.0 for _, s in results)


@pytest.mark.asyncio
async def test_rank_scores_all_passages_when_logprobs_present() -> None:
    reranker = _make_reranker(
        [
            _scored_response('true', -0.05),
            _scored_response('false', -0.05),
        ]
    )

    results = await reranker.rank('query', ['relevant', 'irrelevant'])

    assert {p for p, _ in results} == {'relevant', 'irrelevant'}
    # 'true' (relevant) should outrank 'false' (irrelevant).
    assert results[0][0] == 'relevant'
