"""Endpoint restrictions must not depend on optional group restrictions."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb.operations.search_ops import FalkorSearchOperations
from graphiti_core.driver.neo4j.operations.search_ops import Neo4jSearchOperations
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import edge_similarity_search


@pytest.mark.parametrize(
    'path,provider',
    [
        ('operations', GraphProvider.NEO4J),
        ('operations', GraphProvider.FALKORDB),
        ('fallback', GraphProvider.NEO4J),
        ('fallback', GraphProvider.FALKORDB),
        ('fallback', GraphProvider.KUZU),
        ('fallback', GraphProvider.NEPTUNE),
    ],
)
@pytest.mark.parametrize('group_ids', [None, [], ['memory']])
@pytest.mark.parametrize(
    'source,target', [(None, None), ('alice', None), (None, 'acme'), ('alice', 'acme')]
)
async def test_edge_similarity_endpoint_filters(path, provider, group_ids, source, target):
    executor = MagicMock()
    executor.provider = provider
    executor.search_interface = None
    executor.execute_query = AsyncMock(return_value=([], None, None))
    filters = SearchFilters(edge_uuids=['fact'])
    vector = [1.0, 0.0]

    if path == 'operations':
        operations = (
            Neo4jSearchOperations() if provider == GraphProvider.NEO4J else FalkorSearchOperations()
        )
        search = operations.edge_similarity_search
    else:
        search = edge_similarity_search

    assert await search(executor, vector, source, target, filters, group_ids, 7, 0.2) == []

    executor.execute_query.assert_awaited_once()
    call = executor.execute_query.await_args
    query = call.args[0]
    params = call.kwargs
    # Filtering must happen before scoring/limiting, so unrelated facts cannot
    # consume the requested result budget.
    match_filters = query.split('WITH DISTINCT', 1)[0].split('RETURN', 1)[0]
    for value, predicate, parameter in [
        (source, 'n.uuid = $source_uuid', 'source_uuid'),
        (target, 'm.uuid = $target_uuid', 'target_uuid'),
        (group_ids, 'e.group_id IN $group_ids', 'group_ids'),
    ]:
        if value is not None:
            assert predicate in match_filters
            assert params[parameter] == value
        else:
            assert predicate not in query
            assert parameter not in params

    assert 'e.uuid in $edge_uuids' in match_filters
    assert params['edge_uuids'] == ['fact']
    assert params['search_vector'] == vector
    assert params['limit'] == 7
    assert params['min_score'] == 0.2
