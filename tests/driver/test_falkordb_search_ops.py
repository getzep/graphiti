from typing import Any

import pytest

from graphiti_core.driver.falkordb.operations.search_ops import FalkorSearchOperations
from graphiti_core.search.search_filters import SearchFilters


class RecordingExecutor:
    def __init__(self):
        self.cypher_query = ''
        self.params: dict[str, Any] = {}

    async def execute_query(self, cypher_query_: str, **kwargs: Any):
        self.cypher_query = cypher_query_
        self.params = kwargs
        return [], None, None


@pytest.mark.asyncio
async def test_edge_fulltext_search_uses_returned_relationship_endpoints():
    executor = RecordingExecutor()
    operations = FalkorSearchOperations()

    await operations.edge_fulltext_search(
        executor,
        'api test system',
        SearchFilters(),
        group_ids=['group-a'],
    )

    assert 'YIELD relationship AS e, score' in executor.cypher_query
    assert 'WITH e, score, startNode(e) AS n, endNode(e) AS m' in executor.cypher_query
    assert 'uuid: rel.uuid' not in executor.cypher_query


@pytest.mark.asyncio
async def test_edge_bfs_search_uses_path_relationship_endpoints():
    executor = RecordingExecutor()
    operations = FalkorSearchOperations()

    await operations.edge_bfs_search(
        executor,
        ['origin-uuid'],
        2,
        SearchFilters(),
        group_ids=['group-a'],
    )

    assert "WHERE type(rel) = 'RELATES_TO'" in executor.cypher_query
    assert 'WITH rel AS e, startNode(rel) AS n, endNode(rel) AS m' in executor.cypher_query
    assert 'uuid: rel.uuid' not in executor.cypher_query
