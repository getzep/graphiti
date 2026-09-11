#!/usr/bin/env python3
"""Unit tests for search observability: ranker/score in search responses and
the invalidated-fact signal (issue #1645)."""

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

# Add the src directory to the path (mirrors the other unit tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

import graphiti_mcp_server
from config.schema import GraphitiConfig

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_edge(uuid: str, invalid_at=None, expired_at=None) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        group_id='test-group',
        source_node_uuid='source-uuid',
        target_node_uuid='target-uuid',
        created_at=NOW,
        name='WORKS_AT',
        fact=f'fact {uuid}',
        invalid_at=invalid_at,
        expired_at=expired_at,
    )


def make_node(uuid: str) -> EntityNode:
    return EntityNode(uuid=uuid, name=f'node {uuid}', group_id='test-group')


@pytest.fixture
def mock_service(monkeypatch):
    """Install a mocked GraphitiService and config; yields the mocked client."""
    client = Mock()
    client.search_ = AsyncMock()
    service = Mock()
    service.get_client = AsyncMock(return_value=client)
    monkeypatch.setattr(graphiti_mcp_server, 'graphiti_service', service)
    monkeypatch.setattr(graphiti_mcp_server, 'config', GraphitiConfig(), raising=False)
    return client


@pytest.mark.asyncio
async def test_facts_response_reports_ranker_and_scores(mock_service):
    mock_service.search_.return_value = SearchResults(
        edges=[make_edge('e1'), make_edge('e2')],
        edge_reranker_scores=[0.9, 0.4],
    )

    result = await graphiti_mcp_server.search_memory_facts(query='where does X work?')

    assert result['ranker'] == 'rrf'
    assert [fact['score'] for fact in result['facts']] == [0.9, 0.4]
    assert result['invalidated_count'] == 0
    assert result['invalidated_uuids'] == []


@pytest.mark.asyncio
async def test_facts_response_counts_invalidated_and_superseded(mock_service):
    mock_service.search_.return_value = SearchResults(
        edges=[
            make_edge('live'),
            make_edge('invalidated', invalid_at=NOW),
            make_edge('superseded', expired_at=NOW),
        ],
        edge_reranker_scores=[0.9, 0.5, 0.1],
    )

    result = await graphiti_mcp_server.search_memory_facts(query='where does X work?')

    # No filtering: all facts still returned (graphiti is bi-temporal by design)
    assert len(result['facts']) == 3
    assert result['invalidated_count'] == 2
    assert result['invalidated_uuids'] == ['invalidated', 'superseded']


@pytest.mark.asyncio
async def test_facts_empty_result_still_reports_ranker(mock_service):
    mock_service.search_.return_value = SearchResults()

    result = await graphiti_mcp_server.search_memory_facts(query='unknown topic')

    assert result['message'] == 'No relevant facts found'
    assert result['ranker'] == 'rrf'
    assert result['invalidated_count'] == 0


@pytest.mark.asyncio
async def test_facts_center_node_selects_node_distance_ranker(mock_service):
    mock_service.search_.return_value = SearchResults()

    result = await graphiti_mcp_server.search_memory_facts(
        query='q', center_node_uuid='center-uuid'
    )

    assert result['ranker'] == 'node_distance'


@pytest.mark.asyncio
async def test_facts_limit_set_on_copy_not_shared_recipe(mock_service):
    """The shared module-level recipe must not be mutated (concurrency race)."""
    original_limit = EDGE_HYBRID_SEARCH_RRF.limit
    mock_service.search_.return_value = SearchResults()

    await graphiti_mcp_server.search_memory_facts(query='q', max_facts=3)

    passed_config = mock_service.search_.call_args.kwargs['config']
    assert passed_config.limit == 3
    assert passed_config is not EDGE_HYBRID_SEARCH_RRF
    assert EDGE_HYBRID_SEARCH_RRF.limit == original_limit


@pytest.mark.asyncio
async def test_nodes_response_reports_ranker_and_scores(mock_service):
    mock_service.search_.return_value = SearchResults(
        nodes=[make_node('n1'), make_node('n2')],
        node_reranker_scores=[0.8, 0.3],
    )

    result = await graphiti_mcp_server.search_nodes(query='who is X?')

    assert result['ranker'] == 'rrf'
    assert [node['score'] for node in result['nodes']] == [0.8, 0.3]


@pytest.mark.asyncio
async def test_nodes_empty_result_still_reports_ranker(mock_service):
    mock_service.search_.return_value = SearchResults()

    result = await graphiti_mcp_server.search_nodes(query='unknown topic')

    assert result['message'] == 'No relevant nodes found'
    assert result['ranker'] == 'rrf'


@pytest.mark.asyncio
async def test_nodes_center_node_selects_node_distance_ranker(mock_service):
    mock_service.search_.return_value = SearchResults()

    result = await graphiti_mcp_server.search_nodes(query='q', center_node_uuid='center-uuid')

    assert result['ranker'] == 'node_distance'
