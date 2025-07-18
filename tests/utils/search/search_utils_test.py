from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    hybrid_node_search,
    maximal_marginal_relevance,
    normalize_embeddings_batch,
    normalize_l2_fast,
)


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


def test_normalize_embeddings_batch():
    """Test batch normalization of embeddings."""
    # Test normal case
    embeddings = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])
    normalized = normalize_embeddings_batch(embeddings)

    # Check that vectors are normalized
    assert np.allclose(normalized[0], [0.6, 0.8], rtol=1e-5)  # 3/5, 4/5
    assert np.allclose(normalized[1], [1.0, 0.0], rtol=1e-5)  # Already normalized
    assert np.allclose(normalized[2], [0.0, 0.0], rtol=1e-5)  # Zero vector stays zero

    # Check that norms are 1 (except for zero vector)
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms[0], 1.0, rtol=1e-5)
    assert np.allclose(norms[1], 1.0, rtol=1e-5)
    assert np.allclose(norms[2], 0.0, rtol=1e-5)  # Zero vector

    # Check that output is float32
    assert normalized.dtype == np.float32


def test_normalize_l2_fast():
    """Test fast single vector normalization."""
    # Test normal case
    vector = [3.0, 4.0]
    normalized = normalize_l2_fast(vector)

    # Check that vector is normalized
    assert np.allclose(normalized, [0.6, 0.8], rtol=1e-5)  # 3/5, 4/5

    # Check that norm is 1
    norm = np.linalg.norm(normalized)
    assert np.allclose(norm, 1.0, rtol=1e-5)

    # Check that output is float32
    assert normalized.dtype == np.float32

    # Test zero vector
    zero_vector = [0.0, 0.0]
    normalized_zero = normalize_l2_fast(zero_vector)
    assert np.allclose(normalized_zero, [0.0, 0.0], rtol=1e-5)


def test_maximal_marginal_relevance_empty_candidates():
    """Test MMR with empty candidates."""
    query = [1.0, 0.0, 0.0]
    candidates = {}

    result = maximal_marginal_relevance(query, candidates)
    assert result == []


def test_maximal_marginal_relevance_single_candidate():
    """Test MMR with single candidate."""
    query = [1.0, 0.0, 0.0]
    candidates = {'doc1': [1.0, 0.0, 0.0]}

    result = maximal_marginal_relevance(query, candidates)
    assert result == ['doc1']


def test_maximal_marginal_relevance_basic_functionality():
    """Test basic MMR functionality with multiple candidates."""
    query = [1.0, 0.0, 0.0]
    candidates = {
        'doc1': [1.0, 0.0, 0.0],  # Most relevant to query
        'doc2': [0.0, 1.0, 0.0],  # Orthogonal to query
        'doc3': [0.8, 0.0, 0.0],  # Similar to query but less relevant
    }

    result = maximal_marginal_relevance(query, candidates, mmr_lambda=1.0)  # Only relevance
    # Should select most relevant first
    assert result[0] == 'doc1'

    result = maximal_marginal_relevance(query, candidates, mmr_lambda=0.0)  # Only diversity
    # With pure diversity, should still select most relevant first, then most diverse
    assert result[0] == 'doc1'  # First selection is always most relevant
    assert result[1] == 'doc2'  # Most diverse from doc1


def test_maximal_marginal_relevance_diversity_effect():
    """Test that MMR properly balances relevance and diversity."""
    query = [1.0, 0.0, 0.0]
    candidates = {
        'doc1': [1.0, 0.0, 0.0],  # Most relevant
        'doc2': [0.9, 0.0, 0.0],  # Very similar to doc1, high relevance
        'doc3': [0.0, 1.0, 0.0],  # Orthogonal, lower relevance but high diversity
    }

    # With high lambda (favor relevance), should select doc1, then doc2
    result_relevance = maximal_marginal_relevance(query, candidates, mmr_lambda=0.9)
    assert result_relevance[0] == 'doc1'
    assert result_relevance[1] == 'doc2'

    # With low lambda (favor diversity), should select doc1, then doc3
    result_diversity = maximal_marginal_relevance(query, candidates, mmr_lambda=0.1)
    assert result_diversity[0] == 'doc1'
    assert result_diversity[1] == 'doc3'


def test_maximal_marginal_relevance_min_score_threshold():
    """Test MMR with minimum score threshold."""
    query = [1.0, 0.0, 0.0]
    candidates = {
        'doc1': [1.0, 0.0, 0.0],  # High relevance
        'doc2': [0.0, 1.0, 0.0],  # Low relevance
        'doc3': [-1.0, 0.0, 0.0],  # Negative relevance
    }

    # With high min_score, should only return highly relevant documents
    result = maximal_marginal_relevance(query, candidates, min_score=0.5)
    assert len(result) == 1
    assert result[0] == 'doc1'

    # With low min_score, should return more documents
    result = maximal_marginal_relevance(query, candidates, min_score=-0.5)
    assert len(result) >= 2


def test_maximal_marginal_relevance_max_results():
    """Test MMR with maximum results limit."""
    query = [1.0, 0.0, 0.0]
    candidates = {
        'doc1': [1.0, 0.0, 0.0],
        'doc2': [0.8, 0.0, 0.0],
        'doc3': [0.6, 0.0, 0.0],
        'doc4': [0.4, 0.0, 0.0],
    }

    # Limit to 2 results
    result = maximal_marginal_relevance(query, candidates, max_results=2)
    assert len(result) == 2
    assert result[0] == 'doc1'  # Most relevant

    # Limit to more than available
    result = maximal_marginal_relevance(query, candidates, max_results=10)
    assert len(result) == 4  # Should return all available


def test_maximal_marginal_relevance_deterministic():
    """Test that MMR returns deterministic results."""
    query = [1.0, 0.0, 0.0]
    candidates = {
        'doc1': [1.0, 0.0, 0.0],
        'doc2': [0.0, 1.0, 0.0],
        'doc3': [0.0, 0.0, 1.0],
    }

    # Run multiple times to ensure deterministic behavior
    results = []
    for _ in range(5):
        result = maximal_marginal_relevance(query, candidates)
        results.append(result)

    # All results should be identical
    for result in results[1:]:
        assert result == results[0]


def test_maximal_marginal_relevance_normalized_inputs():
    """Test that MMR handles both normalized and non-normalized inputs correctly."""
    query = [3.0, 4.0]  # Non-normalized
    candidates = {
        'doc1': [6.0, 8.0],  # Same direction as query, non-normalized
        'doc2': [0.6, 0.8],  # Same direction as query, normalized
        'doc3': [0.0, 1.0],  # Orthogonal, normalized
    }

    result = maximal_marginal_relevance(query, candidates)

    # Both doc1 and doc2 should be equally relevant (same direction)
    # The algorithm should handle normalization internally
    assert result[0] in ['doc1', 'doc2']
    assert len(result) == 3


def test_maximal_marginal_relevance_edge_cases():
    """Test MMR with edge cases."""
    query = [0.0, 0.0, 0.0]  # Zero query vector
    candidates = {
        'doc1': [1.0, 0.0, 0.0],
        'doc2': [0.0, 1.0, 0.0],
    }

    # Should still work with zero query (all similarities will be 0)
    result = maximal_marginal_relevance(query, candidates)
    assert len(result) == 2

    # Test with identical candidates
    candidates_identical = {
        'doc1': [1.0, 0.0, 0.0],
        'doc2': [1.0, 0.0, 0.0],
        'doc3': [1.0, 0.0, 0.0],
    }
    query = [1.0, 0.0, 0.0]

    result = maximal_marginal_relevance(query, candidates_identical, mmr_lambda=0.5)
    # Should select only one due to high similarity penalty
    assert len(result) >= 1
