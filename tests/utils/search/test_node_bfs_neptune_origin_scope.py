"""Query-shape regression tests for the Neptune branch of ``node_bfs_search``.

``node_bfs_search`` in ``graphiti_core/search/search_utils.py`` emits a
Neptune-only traversal query whose ``WHERE`` clause is supposed to (a) restrict
the origin to an ``Entity``/``Episodic`` node and (b) keep the traversal inside
the origin's group.

The clause used to be emitted as::

    WHERE origin:Entity OR origin.Episode
    AND n.group_id = origin.group_id

Two independent defects follow from that text:

* Cypher's ``AND`` binds tighter than ``OR``, so the group predicates only apply
  to the right-hand disjunct: ``origin:Entity OR (origin.Episode AND
  n.group_id = ...)``.  A traversal started from an ``Entity`` origin therefore
  ignores ``n.group_id = origin.group_id`` *and* the ``n.group_id IN
  $group_ids`` filter that ``filter_query`` appends, and can walk into another
  group.
* ``origin.Episode`` is a property access, not a label predicate (the label test
  is ``origin:Episodic``), so it is ``null`` for every row: an ``Episodic``
  origin can never satisfy the clause and returns no rows at all, while the
  sibling Neptune branch in ``edge_bfs_search`` correctly spells the label test
  as ``origin:Entity OR origin:Episodic``.

No database is required: a recording driver captures the emitted Cypher.
Mirrors ``tests/utils/search/test_edge_bfs_query_shape.py``.
"""

import re
from typing import Any

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import node_bfs_search


class RecordingNeptuneDriver:
    """Capture the Cypher a search function emits, returning no rows."""

    provider = GraphProvider.NEPTUNE
    search_interface = None
    graph_operations_interface = None
    fulltext_syntax = ''

    def __init__(self):
        self.cypher_query = ''
        self.params: dict[str, Any] = {}

    async def execute_query(self, cypher_query_: str, **kwargs: Any):
        self.cypher_query = cypher_query_
        self.params = kwargs
        return [], None, None


def _normalized_where_clause(query: str) -> str:
    where = query.split('WHERE', 1)[1].split('RETURN', 1)[0]
    return re.sub(r'\s+', ' ', where).strip()


@pytest.mark.asyncio
async def test_neptune_node_bfs_restricts_origin_to_entity_or_episodic():
    driver = RecordingNeptuneDriver()

    await node_bfs_search(
        driver,  # type: ignore[arg-type]
        ['origin-uuid'],
        SearchFilters(),
        bfs_max_depth=2,
        group_ids=['group-a'],
    )

    where = _normalized_where_clause(driver.cypher_query)

    # A label predicate, not a property access on the origin node.
    assert 'origin.Episode' not in where
    assert 'origin:Episodic' in where


@pytest.mark.asyncio
async def test_neptune_node_bfs_scopes_group_for_every_origin_label():
    driver = RecordingNeptuneDriver()

    await node_bfs_search(
        driver,  # type: ignore[arg-type]
        ['origin-uuid'],
        SearchFilters(),
        bfs_max_depth=2,
        group_ids=['group-a'],
    )

    where = _normalized_where_clause(driver.cypher_query)

    # The label disjunction must be grouped so that both group predicates apply
    # to Entity *and* Episodic origins (AND binds tighter than OR in Cypher).
    assert where.startswith('(origin:Entity OR origin:Episodic) AND n.group_id = origin.group_id')
    assert 'n.group_id IN $group_ids' in where


@pytest.mark.asyncio
async def test_neptune_node_bfs_keeps_same_group_guard_without_group_ids():
    driver = RecordingNeptuneDriver()

    await node_bfs_search(
        driver,  # type: ignore[arg-type]
        ['origin-uuid'],
        SearchFilters(),
        bfs_max_depth=3,
        group_ids=None,
    )

    where = _normalized_where_clause(driver.cypher_query)

    assert where == '(origin:Entity OR origin:Episodic) AND n.group_id = origin.group_id'
