"""Test for BFS max_depth parameter fix."""

from unittest.mock import AsyncMock

import pytest

from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import edge_bfs_search, node_bfs_search


@pytest.mark.asyncio
async def test_edge_bfs_search_uses_depth_parameter():
    """Test that edge_bfs_search uses the bfs_max_depth parameter in the query."""
    # Mock driver
    mock_driver = AsyncMock()
    mock_driver.execute_query.return_value = ([], None, None)

    # Mock search filter
    search_filter = SearchFilters()

    # Call edge_bfs_search with depth=2
    await edge_bfs_search(
        driver=mock_driver,
        bfs_origin_node_uuids=['test-uuid'],
        bfs_max_depth=2,
        search_filter=search_filter,
        group_ids=['test-group'],
        limit=10,
    )

    # Verify the query was called
    assert mock_driver.execute_query.called
    call_args = mock_driver.execute_query.call_args

    # Check that depth parameter is passed
    assert 'depth' in call_args.kwargs
    assert call_args.kwargs['depth'] == 2

    # Check that the query contains the variable depth pattern
    query = call_args.args[0]
    assert '{1,$depth}' in query, f"Query should contain '{{1,$depth}}' but got: {query}"


@pytest.mark.asyncio
async def test_node_bfs_search_uses_depth_parameter():
    """Test that node_bfs_search uses the bfs_max_depth parameter in the query."""
    # Mock driver
    mock_driver = AsyncMock()
    mock_driver.execute_query.return_value = ([], None, None)

    # Mock search filter
    search_filter = SearchFilters()

    # Call node_bfs_search with depth=1
    await node_bfs_search(
        driver=mock_driver,
        bfs_origin_node_uuids=['test-uuid'],
        search_filter=search_filter,
        bfs_max_depth=1,
        group_ids=['test-group'],
        limit=10,
    )

    # Verify the query was called
    assert mock_driver.execute_query.called
    call_args = mock_driver.execute_query.call_args

    # Check that depth parameter is passed
    assert 'depth' in call_args.kwargs
    assert call_args.kwargs['depth'] == 1

    # Check that the query contains the variable depth pattern
    query = call_args.args[0]
    assert '{1,$depth}' in query, f"Query should contain '{{1,$depth}}' but got: {query}"


@pytest.mark.asyncio
async def test_different_depth_values():
    """Test that different bfs_max_depth values are correctly passed."""
    mock_driver = AsyncMock()
    mock_driver.execute_query.return_value = ([], None, None)
    search_filter = SearchFilters()

    # Test depth=5
    await edge_bfs_search(
        driver=mock_driver,
        bfs_origin_node_uuids=['test-uuid'],
        bfs_max_depth=5,
        search_filter=search_filter,
        group_ids=['test-group'],
    )

    call_args = mock_driver.execute_query.call_args
    assert call_args.kwargs['depth'] == 5
