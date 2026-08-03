"""Unit tests for Neo4jSearchOperations similarity search query construction.

Regression coverage for #1328: edge_similarity_search / node_similarity_search
/ community_similarity_search computed vector.similarity.cosine() over every
matched row with no guard against a null or invalid embedding. Neo4j's
vector.similarity.cosine() raises Neo.ClientError.Statement.ArgumentError on a
null vector, so a single row with a missing/invalid embedding failed the
entire search. These tests assert the generated Cypher excludes such rows
before the cosine call, using a mocked executor so no live Neo4j instance is
required (mirrors the existing _build_neo4j_fulltext_query query-shape tests).
"""

from unittest.mock import AsyncMock

import pytest

from graphiti_core.driver.neo4j.operations.search_ops import Neo4jSearchOperations
from graphiti_core.search.search_filters import SearchFilters


def _mock_executor():
    executor = AsyncMock()
    executor.execute_query = AsyncMock(return_value=([], None, None))
    return executor


@pytest.mark.asyncio
async def test_node_similarity_search_excludes_null_embeddings():
    executor = _mock_executor()
    ops = Neo4jSearchOperations()

    await ops.node_similarity_search(executor, [0.1, 0.2, 0.3], SearchFilters())

    cypher = executor.execute_query.call_args.args[0]
    assert 'n.name_embedding IS NOT NULL' in cypher
    # The null-check must gate the MATCH before the cosine call, not just
    # appear anywhere in the query string.
    assert cypher.index('n.name_embedding IS NOT NULL') < cypher.index(
        'vector.similarity.cosine'
    )


@pytest.mark.asyncio
async def test_node_similarity_search_combines_null_check_with_group_ids():
    executor = _mock_executor()
    ops = Neo4jSearchOperations()

    await ops.node_similarity_search(
        executor, [0.1, 0.2, 0.3], SearchFilters(), group_ids=['group-1']
    )

    cypher = executor.execute_query.call_args.args[0]
    assert 'n.name_embedding IS NOT NULL' in cypher
    assert 'n.group_id IN $group_ids' in cypher


@pytest.mark.asyncio
async def test_edge_similarity_search_excludes_null_embeddings():
    executor = _mock_executor()
    ops = Neo4jSearchOperations()

    await ops.edge_similarity_search(
        executor, [0.1, 0.2, 0.3], None, None, SearchFilters()
    )

    cypher = executor.execute_query.call_args.args[0]
    assert 'e.fact_embedding IS NOT NULL' in cypher
    assert cypher.index('e.fact_embedding IS NOT NULL') < cypher.index(
        'vector.similarity.cosine'
    )


@pytest.mark.asyncio
async def test_edge_similarity_search_combines_null_check_with_group_ids():
    executor = _mock_executor()
    ops = Neo4jSearchOperations()

    await ops.edge_similarity_search(
        executor,
        [0.1, 0.2, 0.3],
        'source-uuid',
        'target-uuid',
        SearchFilters(),
        group_ids=['group-1'],
    )

    cypher = executor.execute_query.call_args.args[0]
    assert 'e.fact_embedding IS NOT NULL' in cypher
    assert 'e.group_id IN $group_ids' in cypher
    assert 'n.uuid = $source_uuid' in cypher
    assert 'm.uuid = $target_uuid' in cypher


@pytest.mark.asyncio
async def test_community_similarity_search_excludes_null_embeddings():
    executor = _mock_executor()
    ops = Neo4jSearchOperations()

    await ops.community_similarity_search(executor, [0.1, 0.2, 0.3])

    cypher = executor.execute_query.call_args.args[0]
    assert 'c.name_embedding IS NOT NULL' in cypher
    assert cypher.index('c.name_embedding IS NOT NULL') < cypher.index(
        'vector.similarity.cosine'
    )


@pytest.mark.asyncio
async def test_community_similarity_search_combines_null_check_with_group_ids():
    executor = _mock_executor()
    ops = Neo4jSearchOperations()

    await ops.community_similarity_search(executor, [0.1, 0.2, 0.3], group_ids=['group-1'])

    cypher = executor.execute_query.call_args.args[0]
    assert 'c.name_embedding IS NOT NULL' in cypher
    assert 'c.group_id IN $group_ids' in cypher
