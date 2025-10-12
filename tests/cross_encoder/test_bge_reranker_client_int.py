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

import pytest

from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    return BGERerankerClient()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_rank_basic_functionality(client):
    query = 'What is the capital of France?'
    passages = [
        'Paris is the capital and most populous city of France.',
        'London is the capital city of England and the United Kingdom.',
        'Berlin is the capital and largest city of Germany.',
    ]

    ranked_passages = await client.rank(query, passages)

    # Check if the output is a list of tuples
    assert isinstance(ranked_passages, list)
    assert all(isinstance(item, tuple) for item in ranked_passages)

    # Check if the output has the correct length
    assert len(ranked_passages) == len(passages)

    # Check if the scores are floats and passages are strings
    for passage, score in ranked_passages:
        assert isinstance(passage, str)
        assert isinstance(score, float)

    # Check if the results are sorted in descending order
    scores = [score for _, score in ranked_passages]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_rank_empty_input(client):
    query = 'Empty test'
    passages = []

    ranked_passages = await client.rank(query, passages)

    # Check if the output is an empty list
    assert ranked_passages == []


@pytest.mark.asyncio
@pytest.mark.integration
async def test_rank_single_passage(client):
    query = 'Test query'
    passages = ['Single test passage']

    ranked_passages = await client.rank(query, passages)

    # Check if the output has one item
    assert len(ranked_passages) == 1

    # Check if the passage is correct and the score is a float
    assert ranked_passages[0][0] == passages[0]
    assert isinstance(ranked_passages[0][1], float)
