"""Query-shape regression tests for edge_bfs_search (no live database).

Completes #1500 on the generic path in ``graphiti_core/search/search_utils.py``,
which is what executes today for FalkorDB and Neo4j (``driver.search_interface``
is never assigned, so the driver-level operations modules patched by #1500 are
not reached at runtime).

The BFS query must consume the relationships already produced by
``UNWIND relationships(path)`` directly (``startNode``/``endNode``) instead of
re-matching every hit by uuid against the whole graph
(``MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]-(m:Entity)``), which caused
an O(matches x graph) scan per row. Because the BFS path traverses
``RELATES_TO|MENTIONS``, the explicit ``type(e) = 'RELATES_TO'`` guard must be
kept: the old re-MATCH filtered MENTIONS edges out implicitly.

Mirrors the RecordingExecutor approach of tests/test_falkordb_search_ops.py
(#1500): a recording driver captures the emitted Cypher, so no database
connection is required.
"""

from typing import Any

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import edge_bfs_search, edge_fulltext_search


class RecordingDriver:
    """Captures the Cypher and params a search function emits, returning no rows."""

    provider = GraphProvider.NEO4J
    search_interface = None
    fulltext_syntax = ''

    def __init__(self):
        self.cypher_query = ''
        self.params: dict[str, Any] = {}

    async def execute_query(self, cypher_query_: str, **kwargs: Any):
        self.cypher_query = cypher_query_
        self.params = kwargs
        return [], None, None


@pytest.mark.asyncio
async def test_edge_bfs_search_consumes_path_relationships_directly():
    driver = RecordingDriver()

    await edge_bfs_search(
        driver,  # type: ignore[arg-type]
        ['origin-uuid'],
        2,
        SearchFilters(),
        group_ids=['group-a'],
    )

    assert 'WITH rel AS e, startNode(rel) AS n, endNode(rel) AS m' in driver.cypher_query
    # The path traverses RELATES_TO|MENTIONS; the old re-MATCH filtered MENTIONS
    # implicitly, so the rewrite must keep an explicit type guard.
    assert "WHERE type(e) = 'RELATES_TO'" in driver.cypher_query
    # The expensive per-row re-MATCH must be gone.
    assert 'uuid: rel.uuid' not in driver.cypher_query
    # Downstream bindings stay intact: filters and the return still see e, n, m.
    assert 'e.group_id IN $group_ids' in driver.cypher_query
    assert 'n.uuid AS source_node_uuid' in driver.cypher_query
    assert 'm.uuid AS target_node_uuid' in driver.cypher_query


@pytest.mark.asyncio
async def test_edge_fulltext_search_not_rewritten_by_bfs_fix():
    """The fulltext variant of this rewrite belongs to #1500; this change is
    scoped to the BFS query and must leave the fulltext query alone."""
    driver = RecordingDriver()

    await edge_fulltext_search(
        driver,  # type: ignore[arg-type]
        'api test system',
        SearchFilters(),
        group_ids=['group-a'],
    )

    assert 'WITH rel AS e, startNode(rel) AS n, endNode(rel) AS m' not in driver.cypher_query
    assert "type(e) = 'RELATES_TO'" not in driver.cypher_query
