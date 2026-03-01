"""
Tests for group_ids access semantics in search functions.

Security invariants:
  - group_ids=None  → no group restriction (global/all records returned)
  - group_ids=[]    → no groups accessible → ZERO results (fail-closed)
  - group_ids=['g'] → only records in group 'g' returned
  - group_ids=[]    must NOT be silently coerced to None (that would be fail-open)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import fulltext_query


# ---------------------------------------------------------------------------
# Helper: minimal mock driver for unit tests
# ---------------------------------------------------------------------------

def _make_neo4j_driver() -> MagicMock:
    """Return a mock GraphDriver with Neo4j provider (default fulltext syntax)."""
    driver = MagicMock(spec=GraphDriver)
    driver.provider = GraphProvider.NEO4J
    driver.fulltext_syntax = ''
    driver.search_interface = None
    return driver


# ---------------------------------------------------------------------------
# fulltext_query() — Lucene query builder
# ---------------------------------------------------------------------------

class TestFulltextQueryGroupSemantics:
    """Unit tests for fulltext_query() group_ids handling."""

    def test_none_group_ids_produces_no_group_filter(self):
        """group_ids=None → no group restriction in Lucene query."""
        driver = _make_neo4j_driver()
        result = fulltext_query('memory', None, driver)
        # Should produce a query, without any group_id: filter
        assert result != ''
        assert 'group_id' not in result

    def test_empty_group_ids_returns_empty_string(self):
        """group_ids=[] → no query at all → caller returns [] (fail-closed)."""
        driver = _make_neo4j_driver()
        result = fulltext_query('memory', [], driver)
        assert result == '', (
            'fulltext_query must return "" for group_ids=[] so the caller '
            'short-circuits to empty results (fail-closed).'
        )

    def test_single_group_id_produces_filter(self):
        """group_ids=['g1'] → Lucene query includes group filter."""
        driver = _make_neo4j_driver()
        result = fulltext_query('memory', ['g1'], driver)
        assert result != ''
        assert 'group_id:"g1"' in result

    def test_multiple_group_ids_produce_or_filter(self):
        """group_ids=['g1','g2'] → filter includes both groups."""
        driver = _make_neo4j_driver()
        result = fulltext_query('memory', ['g1', 'g2'], driver)
        assert 'group_id:"g1"' in result
        assert 'group_id:"g2"' in result
        assert ' OR ' in result

    def test_empty_list_not_silently_widened_to_none(self):
        """
        Critical security invariant: [] must NOT produce the same output as None.
        If it does, an empty group list silently grants full access.
        """
        driver = _make_neo4j_driver()
        result_none = fulltext_query('memory', None, driver)
        result_empty = fulltext_query('memory', [], driver)
        assert result_none != result_empty, (
            'SECURITY: group_ids=[] and group_ids=None must produce different '
            'fulltext queries. Empty list must not silently grant global access.'
        )
        # Specifically: [] must return '', None must return something non-empty
        assert result_empty == ''
        assert result_none != ''

    def test_kuzu_empty_group_ids_returns_empty_string(self):
        """Kuzu provider: group_ids=[] must still return ''."""
        driver = _make_neo4j_driver()
        driver.provider = GraphProvider.KUZU
        result = fulltext_query('memory', [], driver)
        assert result == '', (
            'Kuzu provider: fulltext_query must return "" for group_ids=[] '
            '(fail-closed).'
        )


# ---------------------------------------------------------------------------
# search() — top-level search function
# ---------------------------------------------------------------------------

class TestSearchGroupIdsSemantics:
    """Unit tests for the search() function group_ids handling."""

    def _make_search_results_with_edges(self):
        from graphiti_core.edges import EntityEdge
        from graphiti_core.nodes import EntityNode
        edge = MagicMock(spec=EntityEdge)
        return SearchResults(edges=[edge])

    @pytest.mark.asyncio
    async def test_empty_group_ids_returns_empty_search_results(self):
        """
        group_ids=[] must immediately return empty SearchResults (fail-closed).
        No search backends should be called.
        """
        from graphiti_core.graphiti_types import GraphitiClients
        from graphiti_core.search.search import search
        from graphiti_core.search.search_config import SearchConfig

        clients = MagicMock(spec=GraphitiClients)
        clients.driver = _make_neo4j_driver()
        clients.embedder = AsyncMock()
        clients.cross_encoder = None

        config = SearchConfig()

        with (
            patch('graphiti_core.search.search.edge_search') as mock_edge,
            patch('graphiti_core.search.search.node_search') as mock_node,
            patch('graphiti_core.search.search.episode_search') as mock_episode,
            patch('graphiti_core.search.search.community_search') as mock_community,
        ):
            result = await search(
                clients=clients,
                query='memory leak',
                group_ids=[],
                config=config,
                search_filter=SearchFilters(),
            )

        assert isinstance(result, SearchResults)
        assert result.edges == []
        assert result.nodes == []
        assert result.episodes == []
        assert result.communities == []

        # Critically: no search backend should have been called
        mock_edge.assert_not_called()
        mock_node.assert_not_called()
        mock_episode.assert_not_called()
        mock_community.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_group_ids_calls_search_backends(self):
        """
        group_ids=None → global access → search backends ARE called.
        """
        from graphiti_core.graphiti_types import GraphitiClients
        from graphiti_core.search.search import search
        from graphiti_core.search.search_config import SearchConfig

        clients = MagicMock(spec=GraphitiClients)
        clients.driver = _make_neo4j_driver()
        clients.embedder = AsyncMock()
        clients.cross_encoder = None

        config = SearchConfig()
        empty_results = ([], [])

        with (
            patch('graphiti_core.search.search.edge_search', new_callable=AsyncMock,
                  return_value=empty_results) as mock_edge,
            patch('graphiti_core.search.search.node_search', new_callable=AsyncMock,
                  return_value=empty_results) as mock_node,
            patch('graphiti_core.search.search.episode_search', new_callable=AsyncMock,
                  return_value=empty_results) as mock_episode,
            patch('graphiti_core.search.search.community_search', new_callable=AsyncMock,
                  return_value=empty_results) as mock_community,
        ):
            result = await search(
                clients=clients,
                query='memory leak',
                group_ids=None,
                config=config,
                search_filter=SearchFilters(),
            )

        # Backends should have been called (global access)
        mock_edge.assert_called_once()
        mock_node.assert_called_once()
        mock_episode.assert_called_once()
        mock_community.assert_called_once()

    @pytest.mark.asyncio
    async def test_group_ids_list_not_coerced_to_none(self):
        """
        SECURITY: group_ids=[] must NOT be silently converted to None before
        calling search backends. The early-return must fire.
        """
        from graphiti_core.graphiti_types import GraphitiClients
        from graphiti_core.search.search import search
        from graphiti_core.search.search_config import SearchConfig

        clients = MagicMock(spec=GraphitiClients)
        clients.driver = _make_neo4j_driver()
        clients.embedder = AsyncMock()
        clients.cross_encoder = None

        config = SearchConfig()

        call_log: list[str] = []

        async def _spy_edge(*args, **kwargs):
            call_log.append('edge_search')
            return ([], [])

        with patch('graphiti_core.search.search.edge_search', side_effect=_spy_edge):
            await search(
                clients=clients,
                query='query',
                group_ids=[],
                config=config,
                search_filter=SearchFilters(),
            )

        assert 'edge_search' not in call_log, (
            'SECURITY: edge_search must not be called when group_ids=[]. '
            'The coercion group_ids=[] -> None must NOT exist.'
        )

    @pytest.mark.asyncio
    async def test_specific_group_ids_passed_to_backends(self):
        """
        group_ids=['gid-1'] → backends are called with that exact list.
        No widening to None.
        """
        from graphiti_core.graphiti_types import GraphitiClients
        from graphiti_core.search.search import search
        from graphiti_core.search.search_config import SearchConfig

        clients = MagicMock(spec=GraphitiClients)
        clients.driver = _make_neo4j_driver()
        clients.embedder = AsyncMock()
        clients.cross_encoder = None

        config = SearchConfig()
        captured_group_ids: list = []

        async def _capture_edge(driver, cross_encoder, query, search_vector,
                                 group_ids, config, search_filter, *args, **kwargs):
            captured_group_ids.extend(group_ids or [])
            return ([], [])

        async def _noop(*args, **kwargs):
            return ([], [])

        with (
            patch('graphiti_core.search.search.edge_search', side_effect=_capture_edge),
            patch('graphiti_core.search.search.node_search', side_effect=_noop),
            patch('graphiti_core.search.search.episode_search', side_effect=_noop),
            patch('graphiti_core.search.search.community_search', side_effect=_noop),
        ):
            await search(
                clients=clients,
                query='query',
                group_ids=['gid-1'],
                config=config,
                search_filter=SearchFilters(),
            )

        assert captured_group_ids == ['gid-1'], (
            'group_ids must be passed through to backends unchanged.'
        )


# ---------------------------------------------------------------------------
# Cypher filter safety — WHERE clause with empty list
# ---------------------------------------------------------------------------

class TestCypherFilterGroupIdsSafety:
    """
    Verify that the Cypher filter pattern `e.group_id IN $group_ids`
    is safe when group_ids is an empty list (should match nothing).

    These tests use mock drivers to verify the query construction,
    not actual Neo4j execution.
    """

    @pytest.mark.asyncio
    async def test_edge_fulltext_search_empty_group_ids_returns_empty(self):
        """
        edge_fulltext_search with group_ids=[] must return [] via fulltext_query
        short-circuit (returns ''), not by executing a widened Cypher query.
        """
        from graphiti_core.search.search_utils import edge_fulltext_search

        driver = _make_neo4j_driver()
        # execute_query should NOT be called — fulltext_query returns '' for []
        driver.execute_query = AsyncMock()

        result = await edge_fulltext_search(
            driver=driver,
            query='test query',
            search_filter=SearchFilters(),
            group_ids=[],
        )

        assert result == [], 'edge_fulltext_search must return [] for group_ids=[]'
        driver.execute_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_node_fulltext_search_empty_group_ids_returns_empty(self):
        """
        node_fulltext_search with group_ids=[] must return [] (fail-closed).
        """
        from graphiti_core.search.search_utils import node_fulltext_search

        driver = _make_neo4j_driver()
        driver.execute_query = AsyncMock()

        result = await node_fulltext_search(
            driver=driver,
            query='test query',
            search_filter=SearchFilters(),
            group_ids=[],
        )

        assert result == [], 'node_fulltext_search must return [] for group_ids=[]'
        driver.execute_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_edge_fulltext_search_none_group_ids_calls_db(self):
        """
        edge_fulltext_search with group_ids=None must call the DB (global access).
        """
        from graphiti_core.search.search_utils import edge_fulltext_search

        driver = _make_neo4j_driver()
        # Return (records, summary, keys) as expected by the driver
        driver.execute_query = AsyncMock(return_value=([], MagicMock(), []))

        await edge_fulltext_search(
            driver=driver,
            query='test query',
            search_filter=SearchFilters(),
            group_ids=None,
        )

        driver.execute_query.assert_called()

    @pytest.mark.asyncio
    async def test_node_fulltext_search_none_group_ids_calls_db(self):
        """
        node_fulltext_search with group_ids=None must call the DB (global access).
        """
        from graphiti_core.search.search_utils import node_fulltext_search

        driver = _make_neo4j_driver()
        driver.execute_query = AsyncMock(return_value=([], MagicMock(), []))

        await node_fulltext_search(
            driver=driver,
            query='test query',
            search_filter=SearchFilters(),
            group_ids=None,
        )

        driver.execute_query.assert_called()
