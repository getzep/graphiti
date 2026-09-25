"""Query-shape regression tests for FalkorSearchOperations (no live database).

``edge_fulltext_search`` must read a matched relationship's endpoints off the
relationship the fulltext index already returned (``startNode``/``endNode``)
instead of re-matching them by uuid
(``MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(m:Entity)``). FalkorDB
plans the re-match as a full ``:Entity`` label scan for every row the index
yields, so the cost becomes O(hits x entities).

The generic path in ``graphiti_core/search/search_utils.py`` was fixed for this;
the driver-level copy here was not, and it becomes the executing path once
search migrates onto ``driver.search_ops``.

Mirrors the recording-driver approach of
tests/utils/search/test_edge_bfs_query_shape.py: a recording executor captures
the emitted Cypher, so no database connection is required.
"""

from typing import Any

import pytest

from graphiti_core.driver.falkordb.operations.search_ops import FalkorSearchOperations
from graphiti_core.search.search_filters import SearchFilters


class RecordingExecutor:
    """Captures the Cypher and params a search operation emits, returning no rows."""

    def __init__(self):
        self.cypher_query = ''
        self.params: dict[str, Any] = {}

    async def execute_query(self, cypher_query_: str, **kwargs: Any):
        self.cypher_query = cypher_query_
        self.params = kwargs
        return [], None, None


@pytest.mark.asyncio
async def test_edge_fulltext_search_resolves_endpoints_without_re_matching():
    executor = RecordingExecutor()

    await FalkorSearchOperations().edge_fulltext_search(
        executor,  # pyright: ignore[reportArgumentType]
        'anything',
        SearchFilters(),
        group_ids=['group-a'],
        limit=10,
    )

    cypher = executor.cypher_query

    # The endpoints come from the yielded relationship.
    assert 'startNode(rel)' in cypher
    assert 'endNode(rel)' in cypher

    # And are never re-matched by uuid, which is what triggers the label scan.
    assert 'RELATES_TO {uuid: rel.uuid}' not in cypher

    # The label predicate must survive, so the result set stays identical to the
    # pattern's: the old MATCH required both endpoints to be :Entity.
    assert 'n:Entity' in cypher
    assert 'm:Entity' in cypher


@pytest.mark.asyncio
async def test_edge_fulltext_search_still_applies_group_and_limit():
    executor = RecordingExecutor()

    await FalkorSearchOperations().edge_fulltext_search(
        executor,  # pyright: ignore[reportArgumentType]
        'anything',
        SearchFilters(),
        group_ids=['group-a'],
        limit=7,
    )

    assert 'e.group_id IN $group_ids' in executor.cypher_query
    assert 'ORDER BY score DESC' in executor.cypher_query
    assert executor.params['group_ids'] == ['group-a']
    assert executor.params['limit'] == 7
