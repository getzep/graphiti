"""Tests for search() behavior with empty query + BFS origin UUIDs.

Regression test: search() must not bail on empty query when bfs_origin_node_uuids
is provided. BFS traversal doesn't need a text query — only origin UUIDs.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search import search
from graphiti_core.search.search_config import (
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    NodeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    SearchConfig,
)
from graphiti_core.search.search_filters import SearchFilters


def _make_clients():
    """Build a minimal GraphitiClients mock."""
    clients = MagicMock()
    clients.driver = MagicMock()
    clients.driver.provider = MagicMock()
    clients.driver.search_interface = None
    clients.embedder = AsyncMock()
    clients.cross_encoder = AsyncMock()
    return clients


def _bfs_only_config():
    """SearchConfig that only uses BFS — no embedding needed."""
    return SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bfs],
            reranker=EdgeReranker.rrf,
            bfs_max_depth=2,
        ),
        node_config=NodeSearchConfig(
            search_methods=[NodeSearchMethod.bfs],
            reranker=NodeReranker.rrf,
            bfs_max_depth=2,
        ),
        limit=10,
    )


@pytest.mark.asyncio
async def test_empty_query_no_bfs_returns_empty():
    """Empty query with no BFS origins should still return empty."""
    clients = _make_clients()
    result = await search(clients, '', None, _bfs_only_config(), SearchFilters())
    assert result.nodes == []
    assert result.edges == []


@pytest.mark.asyncio
async def test_empty_query_with_bfs_origins_runs_bfs():
    """Empty query with bfs_origin_node_uuids must NOT bail — BFS should execute."""
    clients = _make_clients()

    bfs_node = EntityNode(uuid='node-1', name='Vueling Airlines', labels=['Operator'], group_id='g1')
    bfs_edge = EntityEdge(
        uuid='edge-1',
        name='OPERATED_BY',
        fact='Vueling Airlines operates EC-MYT',
        source_node_uuid='node-2',
        target_node_uuid='node-1',
        group_id='g1',
        created_at=datetime.now(timezone.utc),
    )

    with (
        patch('graphiti_core.search.search.node_bfs_search', new_callable=AsyncMock) as mock_node_bfs,
        patch('graphiti_core.search.search.edge_bfs_search', new_callable=AsyncMock) as mock_edge_bfs,
        patch('graphiti_core.search.search.node_fulltext_search', new_callable=AsyncMock) as mock_node_ft,
        patch('graphiti_core.search.search.edge_fulltext_search', new_callable=AsyncMock) as mock_edge_ft,
        patch('graphiti_core.search.search.node_similarity_search', new_callable=AsyncMock) as mock_node_sim,
        patch('graphiti_core.search.search.edge_similarity_search', new_callable=AsyncMock) as mock_edge_sim,
        patch('graphiti_core.search.search.episode_fulltext_search', new_callable=AsyncMock) as mock_ep_ft,
        patch('graphiti_core.search.search.community_fulltext_search', new_callable=AsyncMock) as mock_comm_ft,
        patch('graphiti_core.search.search.community_similarity_search', new_callable=AsyncMock) as mock_comm_sim,
    ):
        mock_node_bfs.return_value = [bfs_node]
        mock_edge_bfs.return_value = [bfs_edge]
        # All other search methods return empty (no text to search)
        for m in [mock_node_ft, mock_edge_ft, mock_node_sim, mock_edge_sim,
                  mock_ep_ft, mock_comm_ft, mock_comm_sim]:
            m.return_value = []

        result = await search(
            clients,
            '',  # empty query
            ['g1'],
            _bfs_only_config(),
            SearchFilters(),
            center_node_uuid='node-1',
            bfs_origin_node_uuids=['node-1'],
        )

        # BFS must have been called
        mock_node_bfs.assert_called_once()
        mock_edge_bfs.assert_called_once()

        # Results must include BFS-discovered nodes and edges
        assert len(result.nodes) >= 1
        assert result.nodes[0].uuid == 'node-1'
        assert len(result.edges) >= 1
        assert result.edges[0].uuid == 'edge-1'


@pytest.mark.asyncio
async def test_empty_query_with_bfs_skips_embedding_call():
    """When query is empty, should NOT call the embedder API — use zero vector instead."""
    clients = _make_clients()

    with (
        patch('graphiti_core.search.search.node_bfs_search', new_callable=AsyncMock, return_value=[]),
        patch('graphiti_core.search.search.edge_bfs_search', new_callable=AsyncMock, return_value=[]),
        patch('graphiti_core.search.search.node_fulltext_search', new_callable=AsyncMock, return_value=[]),
        patch('graphiti_core.search.search.edge_fulltext_search', new_callable=AsyncMock, return_value=[]),
        patch('graphiti_core.search.search.node_similarity_search', new_callable=AsyncMock, return_value=[]),
        patch('graphiti_core.search.search.edge_similarity_search', new_callable=AsyncMock, return_value=[]),
        patch('graphiti_core.search.search.episode_fulltext_search', new_callable=AsyncMock, return_value=[]),
        patch('graphiti_core.search.search.community_fulltext_search', new_callable=AsyncMock, return_value=[]),
        patch('graphiti_core.search.search.community_similarity_search', new_callable=AsyncMock, return_value=[]),
    ):
        await search(
            clients,
            '',  # empty query
            None,
            _bfs_only_config(),
            SearchFilters(),
            bfs_origin_node_uuids=['node-1'],
        )

        # Embedder must NOT be called — zero vector should be used
        clients.embedder.create.assert_not_called()
