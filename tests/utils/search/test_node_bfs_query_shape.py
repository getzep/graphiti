"""Query-shape tests for node_bfs_search (no live database).

The BFS MATCH gives one row for each path from an origin to a node. A node that two
paths reach, or that two origins reach, must come back once. Otherwise the duplicate
rows take places under the LIMIT and add rank to the node in RRF.
"""

from typing import Any

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb.operations.search_ops import FalkorSearchOperations
from graphiti_core.driver.neo4j.operations.search_ops import Neo4jSearchOperations
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import node_bfs_search


class RecordingDriver:
    """Captures the Cypher that a search function emits and returns no rows."""

    search_interface = None

    def __init__(self, provider: GraphProvider):
        self.provider = provider
        self.cypher_query = ''

    async def execute_query(self, cypher_query_: str, **kwargs: Any):
        self.cypher_query = cypher_query_
        return [], [], None


def _assert_distinct_before_return(cypher: str) -> None:
    assert 'WITH DISTINCT n' in cypher
    assert cypher.index('WITH DISTINCT n') < cypher.index('RETURN')
    # The group filters use `origin`, so they must run before the rows collapse.
    assert cypher.index('origin.group_id IN $group_ids') < cypher.index('WITH DISTINCT n')
    assert cypher.index('WITH DISTINCT n') < cypher.index('LIMIT $limit')


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', [GraphProvider.NEO4J, GraphProvider.FALKORDB])
async def test_node_bfs_search_returns_each_node_once(provider):
    driver = RecordingDriver(provider)

    await node_bfs_search(
        driver,  # type: ignore[arg-type]
        ['origin-a', 'origin-b'],
        SearchFilters(),
        2,
        group_ids=['group-a'],
    )

    _assert_distinct_before_return(driver.cypher_query)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'operations', [Neo4jSearchOperations(), FalkorSearchOperations()], ids=['neo4j', 'falkordb']
)
async def test_driver_node_bfs_search_returns_each_node_once(operations):
    driver = RecordingDriver(GraphProvider.NEO4J)

    await operations.node_bfs_search(
        driver,  # type: ignore[arg-type]
        ['origin-a', 'origin-b'],
        SearchFilters(),
        2,
        group_ids=['group-a'],
    )

    _assert_distinct_before_return(driver.cypher_query)
