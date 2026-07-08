from unittest.mock import AsyncMock, patch

import pytest

from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import hybrid_node_search


@pytest.mark.asyncio
async def test_hybrid_node_search_deduplication():
    # Mock the database driver
    mock_driver = AsyncMock()

    # Mock the node_fulltext_search and entity_similarity_search functions
    with (
        patch('graphiti_core.search.search_utils.node_fulltext_search') as mock_fulltext_search,
        patch('graphiti_core.search.search_utils.node_similarity_search') as mock_similarity_search,
    ):
        # Set up mock return values
        mock_fulltext_search.side_effect = [
            [EntityNode(uuid='1', name='Alice', labels=['Entity'], group_id='1')],
            [EntityNode(uuid='2', name='Bob', labels=['Entity'], group_id='1')],
        ]
        mock_similarity_search.side_effect = [
            [EntityNode(uuid='1', name='Alice', labels=['Entity'], group_id='1')],
            [EntityNode(uuid='3', name='Charlie', labels=['Entity'], group_id='1')],
        ]

        # Call the function with test data
        queries = ['Alice', 'Bob']
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        results = await hybrid_node_search(queries, embeddings, mock_driver, SearchFilters())

        # Assertions
        assert len(results) == 3
        assert set(node.uuid for node in results) == {'1', '2', '3'}
        assert set(node.name for node in results) == {'Alice', 'Bob', 'Charlie'}

        # Verify that the mock functions were called correctly
        assert mock_fulltext_search.call_count == 2
        assert mock_similarity_search.call_count == 2


@pytest.mark.asyncio
async def test_hybrid_node_search_empty_results():
    mock_driver = AsyncMock()

    with (
        patch('graphiti_core.search.search_utils.node_fulltext_search') as mock_fulltext_search,
        patch('graphiti_core.search.search_utils.node_similarity_search') as mock_similarity_search,
    ):
        mock_fulltext_search.return_value = []
        mock_similarity_search.return_value = []

        queries = ['NonExistent']
        embeddings = [[0.1, 0.2, 0.3]]
        results = await hybrid_node_search(queries, embeddings, mock_driver, SearchFilters())

        assert len(results) == 0


@pytest.mark.asyncio
async def test_hybrid_node_search_only_fulltext():
    mock_driver = AsyncMock()

    with (
        patch('graphiti_core.search.search_utils.node_fulltext_search') as mock_fulltext_search,
        patch('graphiti_core.search.search_utils.node_similarity_search') as mock_similarity_search,
    ):
        mock_fulltext_search.return_value = [
            EntityNode(uuid='1', name='Alice', labels=['Entity'], group_id='1')
        ]
        mock_similarity_search.return_value = []

        queries = ['Alice']
        embeddings = []
        results = await hybrid_node_search(queries, embeddings, mock_driver, SearchFilters())

        assert len(results) == 1
        assert results[0].name == 'Alice'
        assert mock_fulltext_search.call_count == 1
        assert mock_similarity_search.call_count == 0


@pytest.mark.asyncio
async def test_hybrid_node_search_with_limit():
    mock_driver = AsyncMock()

    with (
        patch('graphiti_core.search.search_utils.node_fulltext_search') as mock_fulltext_search,
        patch('graphiti_core.search.search_utils.node_similarity_search') as mock_similarity_search,
    ):
        mock_fulltext_search.return_value = [
            EntityNode(uuid='1', name='Alice', labels=['Entity'], group_id='1'),
            EntityNode(uuid='2', name='Bob', labels=['Entity'], group_id='1'),
        ]
        mock_similarity_search.return_value = [
            EntityNode(uuid='3', name='Charlie', labels=['Entity'], group_id='1'),
            EntityNode(
                uuid='4',
                name='David',
                labels=['Entity'],
                group_id='1',
            ),
        ]

        queries = ['Test']
        embeddings = [[0.1, 0.2, 0.3]]
        limit = 1
        results = await hybrid_node_search(
            queries, embeddings, mock_driver, SearchFilters(), ['1'], limit
        )

        # We expect 4 results because the limit is applied per search method
        # before deduplication, and we're not actually limiting the results
        # in the hybrid_node_search function itself
        assert len(results) == 4
        assert mock_fulltext_search.call_count == 1
        assert mock_similarity_search.call_count == 1
        # Verify that the limit was passed to the search functions
        mock_fulltext_search.assert_called_with(mock_driver, 'Test', SearchFilters(), ['1'], 2)
        mock_similarity_search.assert_called_with(
            mock_driver, [0.1, 0.2, 0.3], SearchFilters(), ['1'], 2
        )


@pytest.mark.asyncio
async def test_hybrid_node_search_with_limit_and_duplicates():
    mock_driver = AsyncMock()

    with (
        patch('graphiti_core.search.search_utils.node_fulltext_search') as mock_fulltext_search,
        patch('graphiti_core.search.search_utils.node_similarity_search') as mock_similarity_search,
    ):
        mock_fulltext_search.return_value = [
            EntityNode(uuid='1', name='Alice', labels=['Entity'], group_id='1'),
            EntityNode(uuid='2', name='Bob', labels=['Entity'], group_id='1'),
        ]
        mock_similarity_search.return_value = [
            EntityNode(uuid='1', name='Alice', labels=['Entity'], group_id='1'),  # Duplicate
            EntityNode(uuid='3', name='Charlie', labels=['Entity'], group_id='1'),
        ]

        queries = ['Test']
        embeddings = [[0.1, 0.2, 0.3]]
        limit = 2
        results = await hybrid_node_search(
            queries, embeddings, mock_driver, SearchFilters(), ['1'], limit
        )

        # We expect 3 results because:
        # 1. The limit of 2 is applied to each search method
        # 2. We get 2 results from fulltext and 2 from similarity
        # 3. One result is a duplicate (Alice), so it's only included once
        assert len(results) == 3
        assert set(node.name for node in results) == {'Alice', 'Bob', 'Charlie'}
        assert mock_fulltext_search.call_count == 1
        assert mock_similarity_search.call_count == 1
        mock_fulltext_search.assert_called_with(mock_driver, 'Test', SearchFilters(), ['1'], 4)
        mock_similarity_search.assert_called_with(
            mock_driver, [0.1, 0.2, 0.3], SearchFilters(), ['1'], 4
        )
