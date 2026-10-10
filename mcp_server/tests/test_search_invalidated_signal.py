#!/usr/bin/env python3
"""Unit tests for the search observability additions to the MCP search tools.

Covers `exclude_invalidated` / `suppressed_invalidated` on `search_memory_facts`,
the `ranker` label on both search tools, and the per-result `score`. These drive
the tool functions against a mocked Graphiti client, so they need no database,
no LLM, and run in the default (non-integration) suite.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_config import SearchResults

# Add the src directory to the path (mirrors the other unit tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import graphiti_mcp_server as server  # noqa: E402
from utils.formatting import format_fact_result, is_invalidated, to_node_result  # noqa: E402

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


def make_edge(name: str, *, invalid_at=None, expired_at=None) -> EntityEdge:
    edge = EntityEdge(
        uuid=f'edge-{name}',
        group_id='g',
        source_node_uuid='src',
        target_node_uuid='tgt',
        name=name,
        fact=f'fact about {name}',
        created_at=NOW,
        valid_at=NOW,
        invalid_at=invalid_at,
    )
    # expired_at is set by the invalidation write path rather than at construction.
    edge.expired_at = expired_at
    return edge


def make_node(name: str) -> EntityNode:
    return EntityNode(uuid=f'node-{name}', name=name, group_id='g', created_at=NOW)


@pytest.fixture
def fake_service(monkeypatch):
    """Install a graphiti_service whose client returns canned SearchResults."""
    client = AsyncMock(spec=Graphiti)
    service = AsyncMock()
    service.get_client = AsyncMock(return_value=client)
    monkeypatch.setattr(server, 'graphiti_service', service)
    return client


class TestIsInvalidated:
    """Either temporal marker retires a fact; they are set independently."""

    def test_live_edge(self):
        assert is_invalidated(make_edge('live')) is False

    def test_invalid_at_only(self):
        assert is_invalidated(make_edge('a', invalid_at=NOW)) is True

    def test_expired_at_only(self):
        # A fact superseded by dedupe carries expired_at with no invalid_at.
        assert is_invalidated(make_edge('b', expired_at=NOW)) is True


class TestSuppressedInvalidated:
    @pytest.mark.asyncio
    async def test_off_by_default_returns_everything(self, fake_service):
        edges = [make_edge('live'), make_edge('dead', invalid_at=NOW)]
        fake_service.search_.return_value = SearchResults(
            edges=edges, edge_reranker_scores=[0.9, 0.8]
        )

        result = await server.search_memory_facts(query='q', group_ids='g')

        assert len(result['facts']) == 2
        assert result['suppressed_invalidated'] == 0

    @pytest.mark.asyncio
    async def test_opt_in_drops_and_counts(self, fake_service):
        edges = [
            make_edge('live'),
            make_edge('dead1', invalid_at=NOW),
            make_edge('dead2', expired_at=NOW),
        ]
        fake_service.search_.return_value = SearchResults(
            edges=edges, edge_reranker_scores=[0.9, 0.8, 0.7]
        )

        result = await server.search_memory_facts(
            query='q', group_ids='g', exclude_invalidated=True
        )

        assert [f['name'] for f in result['facts']] == ['live']
        assert result['suppressed_invalidated'] == 2

    @pytest.mark.asyncio
    async def test_all_suppressed_says_so(self, fake_service):
        """An empty result after filtering must not read as absence of memory."""
        fake_service.search_.return_value = SearchResults(
            edges=[make_edge('dead', invalid_at=NOW)], edge_reranker_scores=[0.9]
        )

        result = await server.search_memory_facts(
            query='q', group_ids='g', exclude_invalidated=True
        )

        assert result['facts'] == []
        assert result['suppressed_invalidated'] == 1
        assert 'superseded' in result['message']
        assert result['message'] != 'No relevant facts found'

    @pytest.mark.asyncio
    async def test_genuinely_empty_keeps_original_message(self, fake_service):
        fake_service.search_.return_value = SearchResults(edges=[], edge_reranker_scores=[])

        result = await server.search_memory_facts(
            query='q', group_ids='g', exclude_invalidated=True
        )

        assert result['facts'] == []
        assert result['suppressed_invalidated'] == 0
        assert result['message'] == 'No relevant facts found'


class TestRankerAndScores:
    @pytest.mark.asyncio
    async def test_facts_report_rrf_without_center_node(self, fake_service):
        fake_service.search_.return_value = SearchResults(
            edges=[make_edge('a')], edge_reranker_scores=[0.42]
        )

        result = await server.search_memory_facts(query='q', group_ids='g')

        assert result['ranker'] == 'rrf'
        assert result['facts'][0]['score'] == 0.42

    @pytest.mark.asyncio
    async def test_facts_report_node_distance_with_center_node(self, fake_service):
        fake_service.search_.return_value = SearchResults(
            edges=[make_edge('a')], edge_reranker_scores=[0.42]
        )

        result = await server.search_memory_facts(
            query='q', group_ids='g', center_node_uuid='node-1'
        )

        assert result['ranker'] == 'node_distance'

    @pytest.mark.asyncio
    async def test_nodes_report_ranker_and_scores(self, fake_service):
        fake_service.search_.return_value = SearchResults(
            nodes=[make_node('a')], node_reranker_scores=[0.31]
        )

        result = await server.search_nodes(query='q', group_ids='g')

        assert result['ranker'] == 'rrf'
        assert result['nodes'][0]['score'] == 0.31

    @pytest.mark.asyncio
    async def test_ranker_present_on_empty_results(self, fake_service):
        """A client thresholding on score needs the ranker even with no hits."""
        fake_service.search_.return_value = SearchResults(edges=[], nodes=[])

        facts = await server.search_memory_facts(query='q', group_ids='g')
        nodes = await server.search_nodes(query='q', group_ids='g')

        assert facts['ranker'] == 'rrf'
        assert nodes['ranker'] == 'rrf'

    @pytest.mark.asyncio
    async def test_missing_scores_are_omitted_not_faked(self, fake_service):
        """A reranker returning fewer scores than results must not invent zeros."""
        fake_service.search_.return_value = SearchResults(
            edges=[make_edge('a'), make_edge('b')], edge_reranker_scores=[0.9]
        )

        result = await server.search_memory_facts(query='q', group_ids='g')

        assert result['facts'][0]['score'] == 0.9
        assert 'score' not in result['facts'][1]


class TestLimitIsolation:
    @pytest.mark.asyncio
    async def test_limit_does_not_leak_into_the_shared_recipe(self, fake_service):
        """The recipes are module-level singletons; max_facts must not mutate them."""
        from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

        before = EDGE_HYBRID_SEARCH_RRF.limit
        fake_service.search_.return_value = SearchResults(edges=[], edge_reranker_scores=[])

        await server.search_memory_facts(query='q', group_ids='g', max_facts=99)

        assert EDGE_HYBRID_SEARCH_RRF.limit == before
        assert fake_service.search_.call_args.kwargs['config'].limit == 99


class TestFormattingHelpers:
    def test_score_omitted_when_none(self):
        assert 'score' not in format_fact_result(make_edge('a'))
        assert 'score' not in to_node_result(make_node('a'))

    def test_score_included_when_given(self):
        assert format_fact_result(make_edge('a'), 0.5)['score'] == 0.5
        assert to_node_result(make_node('a'), 0.5)['score'] == 0.5
