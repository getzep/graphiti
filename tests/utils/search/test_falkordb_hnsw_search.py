"""Tests for FalkorDB HNSW vector index search branches in search_utils.py."""

from unittest.mock import AsyncMock, PropertyMock

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    community_similarity_search,
    edge_similarity_search,
    node_similarity_search,
)


def _make_falkordb_driver():
    """Create a mock GraphDriver configured as FalkorDB."""
    driver = AsyncMock()
    type(driver).provider = PropertyMock(return_value=GraphProvider.FALKORDB)
    driver.search_interface = None
    driver.fulltext_syntax = '@'
    return driver


def _make_edge_record():
    """Create a mock edge record matching get_entity_edge_return_query output."""
    return {
        'uuid': 'edge-1',
        'group_id': 'group-1',
        'source_node_uuid': 'node-1',
        'target_node_uuid': 'node-2',
        'source_node_name': 'Alice',
        'target_node_name': 'Bob',
        'created_at': '2024-01-01T00:00:00',
        'expired_at': None,
        'valid_at': None,
        'invalid_at': None,
        'name': 'knows',
        'fact': 'Alice knows Bob',
        'fact_embedding': [0.1] * 768,
        'episodes': ['ep-1'],
        'source_name': 'test',
        'source_description': 'test source',
        'attributes': {},
    }


def _make_node_record():
    """Create a mock node record matching get_entity_node_return_query output."""
    return {
        'uuid': 'node-1',
        'group_id': 'group-1',
        'name': 'Alice',
        'name_embedding': [0.1] * 768,
        'labels': ['Entity'],
        'created_at': '2024-01-01T00:00:00',
        'summary': 'A person named Alice',
        'attributes': {},
    }


def _make_community_record():
    """Create a mock community record matching COMMUNITY_NODE_RETURN output."""
    return {
        'uuid': 'comm-1',
        'group_id': 'group-1',
        'name': 'Test Community',
        'name_embedding': [0.1] * 768,
        'created_at': '2024-01-01T00:00:00',
        'summary': 'A test community',
    }


class TestFalkorDBEdgeSimilaritySearch:
    """Tests for FalkorDB HNSW branch in edge_similarity_search."""

    @pytest.mark.asyncio
    async def test_uses_hnsw_index_query(self):
        """Test that FalkorDB branch uses db.idx.vector.queryRelationships."""
        driver = _make_falkordb_driver()
        driver.execute_query.return_value = ([_make_edge_record()], ['uuid'], None)

        search_vector = [0.1] * 768
        await edge_similarity_search(
            driver,
            search_vector,
            source_node_uuid=None,
            target_node_uuid=None,
            search_filter=SearchFilters(),
            group_ids=['group-1'],
        )

        driver.execute_query.assert_called_once()
        query = driver.execute_query.call_args[0][0]
        assert 'db.idx.vector.queryRelationships' in query
        assert 'RELATES_TO' in query
        assert 'fact_embedding' in query

    @pytest.mark.asyncio
    async def test_uses_start_end_node_not_match(self):
        """Test that FalkorDB branch uses startNode/endNode instead of MATCH re-scan."""
        driver = _make_falkordb_driver()
        driver.execute_query.return_value = ([_make_edge_record()], ['uuid'], None)

        search_vector = [0.1] * 768
        await edge_similarity_search(
            driver,
            search_vector,
            source_node_uuid=None,
            target_node_uuid=None,
            search_filter=SearchFilters(),
            group_ids=['group-1'],
        )

        query = driver.execute_query.call_args[0][0]
        assert 'startNode(e)' in query
        assert 'endNode(e)' in query
        # Should NOT use MATCH to re-scan edges
        assert 'MATCH (n:Entity)-[e]->(m:Entity)' not in query

    @pytest.mark.asyncio
    async def test_over_fetches_for_post_filtering(self):
        """Test that the over-fetch limit is passed for post-filtering compensation."""
        driver = _make_falkordb_driver()
        driver.execute_query.return_value = ([], [], None)

        search_vector = [0.1] * 768
        limit = 10
        await edge_similarity_search(
            driver,
            search_vector,
            source_node_uuid=None,
            target_node_uuid=None,
            search_filter=SearchFilters(),
            group_ids=['group-1'],
            limit=limit,
        )

        call_kwargs = driver.execute_query.call_args[1]
        assert call_kwargs['over_fetch_limit'] == limit * 10
        assert call_kwargs['limit'] == limit

    @pytest.mark.asyncio
    async def test_applies_group_id_filter(self):
        """Test that group_id filter is applied in the query."""
        driver = _make_falkordb_driver()
        driver.execute_query.return_value = ([], [], None)

        search_vector = [0.1] * 768
        await edge_similarity_search(
            driver,
            search_vector,
            source_node_uuid=None,
            target_node_uuid=None,
            search_filter=SearchFilters(),
            group_ids=['group-1'],
        )

        query = driver.execute_query.call_args[0][0]
        assert 'group_id' in query

    @pytest.mark.asyncio
    async def test_applies_min_score_filter(self):
        """Test that min_score filter is included in the query."""
        driver = _make_falkordb_driver()
        driver.execute_query.return_value = ([], [], None)

        search_vector = [0.1] * 768
        await edge_similarity_search(
            driver,
            search_vector,
            source_node_uuid=None,
            target_node_uuid=None,
            search_filter=SearchFilters(),
            group_ids=['group-1'],
            min_score=0.5,
        )

        query = driver.execute_query.call_args[0][0]
        assert 'score > $min_score' in query
        call_kwargs = driver.execute_query.call_args[1]
        assert call_kwargs['min_score'] == 0.5


class TestFalkorDBNodeSimilaritySearch:
    """Tests for FalkorDB HNSW branch in node_similarity_search."""

    @pytest.mark.asyncio
    async def test_uses_hnsw_index_query(self):
        """Test that FalkorDB branch uses db.idx.vector.queryNodes for Entity."""
        driver = _make_falkordb_driver()
        driver.execute_query.return_value = ([_make_node_record()], ['uuid'], None)

        search_vector = [0.1] * 768
        await node_similarity_search(
            driver,
            search_vector,
            search_filter=SearchFilters(),
            group_ids=['group-1'],
        )

        driver.execute_query.assert_called_once()
        query = driver.execute_query.call_args[0][0]
        assert 'db.idx.vector.queryNodes' in query
        assert "'Entity'" in query
        assert 'name_embedding' in query

    @pytest.mark.asyncio
    async def test_over_fetches_for_post_filtering(self):
        """Test that the over-fetch limit is passed."""
        driver = _make_falkordb_driver()
        driver.execute_query.return_value = ([], [], None)

        search_vector = [0.1] * 768
        limit = 5
        await node_similarity_search(
            driver,
            search_vector,
            search_filter=SearchFilters(),
            group_ids=['group-1'],
            limit=limit,
        )

        call_kwargs = driver.execute_query.call_args[1]
        assert call_kwargs['over_fetch_limit'] == limit * 10

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_no_results(self):
        """Test that empty results are handled correctly."""
        driver = _make_falkordb_driver()
        driver.execute_query.return_value = ([], [], None)

        search_vector = [0.1] * 768
        results = await node_similarity_search(
            driver,
            search_vector,
            search_filter=SearchFilters(),
            group_ids=['group-1'],
        )

        assert results == []


class TestFalkorDBCommunitySimilaritySearch:
    """Tests for FalkorDB HNSW branch in community_similarity_search."""

    @pytest.mark.asyncio
    async def test_uses_hnsw_index_query(self):
        """Test that FalkorDB branch uses db.idx.vector.queryNodes for Community."""
        driver = _make_falkordb_driver()
        driver.execute_query.return_value = ([_make_community_record()], ['uuid'], None)

        search_vector = [0.1] * 768
        await community_similarity_search(
            driver,
            search_vector,
            group_ids=['group-1'],
        )

        driver.execute_query.assert_called_once()
        query = driver.execute_query.call_args[0][0]
        assert 'db.idx.vector.queryNodes' in query
        assert "'Community'" in query
        assert 'name_embedding' in query

    @pytest.mark.asyncio
    async def test_applies_group_id_filter(self):
        """Test that group_id filter is applied."""
        driver = _make_falkordb_driver()
        driver.execute_query.return_value = ([], [], None)

        search_vector = [0.1] * 768
        await community_similarity_search(
            driver,
            search_vector,
            group_ids=['group-1'],
        )

        query = driver.execute_query.call_args[0][0]
        assert 'group_id' in query

    @pytest.mark.asyncio
    async def test_over_fetches_for_post_filtering(self):
        """Test that the over-fetch limit is passed."""
        driver = _make_falkordb_driver()
        driver.execute_query.return_value = ([], [], None)

        search_vector = [0.1] * 768
        limit = 8
        await community_similarity_search(
            driver,
            search_vector,
            group_ids=['group-1'],
            limit=limit,
        )

        call_kwargs = driver.execute_query.call_args[1]
        assert call_kwargs['over_fetch_limit'] == limit * 10
