"""Unit tests for the FalkorDB episode fulltext search query shape.

Regression guard for https://github.com/getzep/graphiti/issues/1819: the
fulltext procedure already yields the Episodic node, but the query used to
re-match it via ``MATCH (e:Episodic) WHERE e.uuid = episode.uuid``. That
predicate cannot be answered from an index (the right-hand side is a per-row
value), so every fulltext hit triggered a scan over all Episodic nodes —
O(hits x episodes) — which pins a FalkorDB core and wedges any search recipe
that includes the episode scope once the graph is non-trivial.

The tests assert on the generated Cypher with a mocked executor, so they need
no live FalkorDB and run in CI regardless of whether the ``falkordb`` package
is installed. They are deliberately scoped to the FalkorDB operations class:
the Neo4j/Kuzu/Neptune implementations are not coupled to this query-plan
workaround.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.driver.falkordb.operations.search_ops import FalkorSearchOperations
from graphiti_core.search.search_filters import SearchFilters

pytestmark = pytest.mark.asyncio


def _make_executor():
    executor = MagicMock(spec=GraphDriver)
    executor.provider = GraphProvider.FALKORDB
    executor.execute_query = AsyncMock(return_value=([], None, None))
    return executor


async def _run_search(executor, group_ids):
    ops = FalkorSearchOperations()
    return await ops.episode_fulltext_search(
        executor, 'Duetto revenue', SearchFilters(), group_ids, limit=10
    )


async def test_episode_fulltext_uses_yielded_node_directly():
    """The yielded node must be used as-is — no per-hit re-MATCH by uuid."""
    executor = _make_executor()

    await _run_search(executor, ['group-a'])

    executor.execute_query.assert_awaited_once()
    cypher = executor.execute_query.await_args.args[0]

    # The O(hits x episodes) shape must not come back.
    assert 'MATCH (e:Episodic)' not in cypher
    assert 'episode.uuid' not in cypher
    # The procedure's yielded node is the episode.
    assert 'YIELD node AS e' in cypher


async def test_episode_fulltext_group_filter_applies_to_yielded_node():
    executor = _make_executor()

    await _run_search(executor, ['group-a'])

    cypher = executor.execute_query.await_args.args[0]
    kwargs = executor.execute_query.await_args.kwargs

    assert 'WHERE e.group_id IN $group_ids' in cypher
    assert kwargs['group_ids'] == ['group-a']


async def test_episode_fulltext_no_group_filter_when_group_ids_none():
    executor = _make_executor()

    await _run_search(executor, None)

    cypher = executor.execute_query.await_args.args[0]
    kwargs = executor.execute_query.await_args.kwargs

    assert 'WHERE e.group_id IN $group_ids' not in cypher
    assert 'group_ids' not in kwargs


async def test_episode_fulltext_preserves_ordering_and_limit():
    executor = _make_executor()

    await _run_search(executor, ['group-a'])

    cypher = executor.execute_query.await_args.args[0]
    kwargs = executor.execute_query.await_args.kwargs

    assert 'ORDER BY score DESC' in cypher
    assert 'LIMIT $limit' in cypher
    assert kwargs['limit'] == 10


async def test_episode_fulltext_empty_query_short_circuits():
    """An all-stopword/empty query must return before executing any Cypher."""
    executor = _make_executor()
    ops = FalkorSearchOperations()

    result = await ops.episode_fulltext_search(
        executor, 'the and or', SearchFilters(), ['group-a'], limit=10
    )

    assert result == []
    executor.execute_query.assert_not_called()
