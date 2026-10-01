"""Query-shape regression tests for episode_fulltext_search (no live database).

Covers #1819 on the generic path in ``graphiti_core/search/search_utils.py``,
which is what executes today for Neo4j, FalkorDB and Kuzu
(``driver.search_interface`` is never assigned, so the driver-level operations
modules are not reached at runtime).

``episode_fulltext_search`` re-matched every fulltext hit against all
``Episodic`` nodes by unindexed equality::

    CALL db.idx.fulltext.queryNodes('Episodic', $query) YIELD node AS episode, score
    MATCH (e:Episodic)
    WHERE e.uuid = episode.uuid

The ``e.uuid = episode.uuid`` predicate cannot be answered from an index (the
right-hand side is a per-row value), so each yielded hit scans all ``Episodic``
nodes - O(hits x episodes). Episode nodes carry large ``content`` properties, so
every row is expensive; the reporter measured searches that stopped completing
within a 30s timeout on a 2,163-episode graph.

The yielded node *is* the target, and ``episode_content`` is only ever indexed
over ``Episodic`` (``CREATE FULLTEXT INDEX episode_content FOR (e:Episodic)`` on
Neo4j/FalkorDB, ``CREATE_FTS_INDEX('Episodic', 'episode_content', ...)`` on
Kuzu), so the re-MATCH is redundant. The sibling fulltext searches in the same
module already consume the yielded node directly::

    YIELD node AS n, score          # node_fulltext_search / community_fulltext_search
    WITH n, score
    WHERE n.group_id IN $group_ids

Kuzu does not accept a ``YIELD`` clause after ``CALL QUERY_FTS_INDEX``; both
siblings swap it for ``WITH node AS <alias>, score``. ``episode_fulltext_search``
hardcoded ``YIELD``, so it needs the same provider-aware clause.

Mirrors the RecordingDriver approach of tests/utils/search/test_edge_bfs_query_shape.py
(#1500): a recording driver captures the emitted Cypher, so no database
connection is required.
"""

from typing import Any

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb.operations.search_ops import (
    _build_falkor_fulltext_query,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import episode_fulltext_search


class RecordingDriver:
    """Captures the Cypher and params a search function emits, returning no rows."""

    search_interface = None
    fulltext_syntax = ''

    def __init__(self, provider: GraphProvider = GraphProvider.NEO4J):
        self.provider = provider
        self.cypher_query = ''
        self.params: dict[str, Any] = {}

    async def execute_query(self, cypher_query_: str, **kwargs: Any):
        self.cypher_query = cypher_query_
        self.params = kwargs
        return [], None, None

    def build_fulltext_query(
        self, query: str, group_ids: list[str] | None = None, max_query_length: int = 128
    ):
        return _build_falkor_fulltext_query(query, group_ids, max_query_length)


@pytest.mark.asyncio
async def test_episode_fulltext_search_consumes_yielded_node_directly():
    driver = RecordingDriver()

    await episode_fulltext_search(
        driver,  # type: ignore[arg-type]
        'alpha beta',
        SearchFilters(),
        group_ids=['group-a'],
    )

    # The procedure already yields the Episodic node: alias it and use it.
    assert 'YIELD node AS e, score' in driver.cypher_query
    # The O(hits x episodes) re-MATCH must be gone.
    assert 'MATCH (e:Episodic)' not in driver.cypher_query
    assert 'episode.uuid' not in driver.cypher_query
    assert 'e.uuid = ' not in driver.cypher_query
    # Downstream filters still see the node.
    assert 'WHERE e.group_id IN $group_ids' in driver.cypher_query
    assert driver.params['group_ids'] == ['group-a']


@pytest.mark.asyncio
async def test_episode_fulltext_search_without_group_ids_has_no_dangling_where():
    driver = RecordingDriver()

    await episode_fulltext_search(driver, 'alpha beta', SearchFilters())  # type: ignore[arg-type]

    assert 'WHERE e.group_id' not in driver.cypher_query
    assert 'MATCH (e:Episodic)' not in driver.cypher_query
    assert 'episode.uuid' not in driver.cypher_query
    assert 'group_ids' not in driver.params


@pytest.mark.asyncio
async def test_episode_fulltext_search_kuzu_uses_with_not_yield():
    """Kuzu's QUERY_FTS_INDEX is a table function; a YIELD clause is not valid."""
    driver = RecordingDriver(GraphProvider.KUZU)

    await episode_fulltext_search(
        driver,  # type: ignore[arg-type]
        'alpha beta',
        SearchFilters(),
        group_ids=['group-a'],
    )

    assert 'WITH node AS e, score' in driver.cypher_query
    assert 'YIELD' not in driver.cypher_query
    assert 'MATCH (e:Episodic)' not in driver.cypher_query
    assert 'episode.uuid' not in driver.cypher_query
    assert 'WHERE e.group_id IN $group_ids' in driver.cypher_query


@pytest.mark.asyncio
async def test_episode_fulltext_search_falkordb_targets_episodic_index():
    driver = RecordingDriver(GraphProvider.FALKORDB)
    driver.fulltext_syntax = '@'

    await episode_fulltext_search(
        driver,  # type: ignore[arg-type]
        'alpha beta',
        SearchFilters(),
        group_ids=['group-a'],
    )

    assert "db.idx.fulltext.queryNodes('Episodic', $query)" in driver.cypher_query
    assert 'MATCH (e:Episodic)' not in driver.cypher_query
    assert 'episode.uuid' not in driver.cypher_query
    assert 'WHERE e.group_id IN $group_ids' in driver.cypher_query


@pytest.mark.asyncio
async def test_episode_fulltext_search_orders_and_limits():
    """The rewrite must keep score ordering and the limit."""
    driver = RecordingDriver()

    await episode_fulltext_search(
        driver,  # type: ignore[arg-type]
        'alpha beta',
        SearchFilters(),
        group_ids=['group-a'],
    )

    assert 'ORDER BY score DESC' in driver.cypher_query
    assert 'LIMIT $limit' in driver.cypher_query
    assert driver.params['limit'] == 10
