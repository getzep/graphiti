"""Unit tests for graphiti_core.search.search_utils.episode_mentions_reranker.

Regression coverage for #1342: the reranker sorted ascending by mention
count, so the least-mentioned nodes ranked first - the opposite of what the
function name and every other count-based reranker implies.
"""

from unittest.mock import AsyncMock

import pytest

from graphiti_core.search.search_utils import episode_mentions_reranker


def _mock_driver(records):
    driver = AsyncMock()
    # driver.search_interface must be falsy to exercise the direct-query
    # fallback path exercised by these tests, rather than a driver-specific
    # override.
    driver.search_interface = None
    driver.execute_query = AsyncMock(return_value=(records, None, None))
    return driver


@pytest.mark.asyncio
async def test_episode_mentions_reranker_orders_by_descending_mentions():
    driver = _mock_driver(
        [
            {'uuid': 'low', 'score': 1},
            {'uuid': 'high', 'score': 10},
            {'uuid': 'mid', 'score': 5},
        ]
    )

    sorted_uuids, scores = await episode_mentions_reranker(driver, [['low', 'high', 'mid']])

    assert sorted_uuids == ['high', 'mid', 'low']
    assert scores == [10, 5, 1]


@pytest.mark.asyncio
async def test_episode_mentions_reranker_unmentioned_nodes_rank_last():
    driver = _mock_driver([{'uuid': 'mentioned', 'score': 3}])

    sorted_uuids, scores = await episode_mentions_reranker(
        driver, [['mentioned', 'unmentioned']], min_score=0
    )

    assert sorted_uuids == ['mentioned', 'unmentioned']
    assert scores == [3, 0]


@pytest.mark.asyncio
async def test_episode_mentions_reranker_min_score_filters_unmentioned_nodes():
    driver = _mock_driver([{'uuid': 'mentioned', 'score': 3}])

    # A positive min_score should filter out nodes with zero mentions -
    # under the old float('inf') sentinel this could never happen, since
    # infinity always satisfied `score >= min_score`.
    sorted_uuids, scores = await episode_mentions_reranker(
        driver, [['mentioned', 'unmentioned']], min_score=1
    )

    assert sorted_uuids == ['mentioned']
    assert scores == [3]


@pytest.mark.asyncio
async def test_episode_mentions_reranker_all_unmentioned_returns_empty_when_filtered():
    driver = _mock_driver([])

    sorted_uuids, scores = await episode_mentions_reranker(driver, [['a', 'b']], min_score=1)

    assert sorted_uuids == []
    assert scores == []
