from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graphiti_core.nodes import EntityNode
from graphiti_core.search.search import edge_search, search
from graphiti_core.search.search_config import (
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    NodeSearchConfig,
    NodeSearchMethod,
    SearchConfig,
)
from graphiti_core.search.search_filters import SearchFilters


@pytest.mark.asyncio
async def test_edge_search_skips_all_work_for_empty_edge_uuid_filter(monkeypatch):
    fulltext_search = AsyncMock(return_value=[])
    similarity_search = AsyncMock(return_value=[])
    bfs_search = AsyncMock(return_value=[])
    cross_encoder = SimpleNamespace(rank=AsyncMock(return_value=[]))

    monkeypatch.setattr(
        'graphiti_core.search.search.edge_fulltext_search',
        fulltext_search,
    )
    monkeypatch.setattr(
        'graphiti_core.search.search.edge_similarity_search',
        similarity_search,
    )
    monkeypatch.setattr(
        'graphiti_core.search.search.edge_bfs_search',
        bfs_search,
    )

    edges, scores = await edge_search(
        driver=SimpleNamespace(),
        cross_encoder=cross_encoder,
        query='known edge',
        query_vector=[0.1, 0.2, 0.3],
        group_ids=['group_1'],
        config=EdgeSearchConfig(
            search_methods=[
                EdgeSearchMethod.bm25,
                EdgeSearchMethod.cosine_similarity,
                EdgeSearchMethod.bfs,
            ],
            reranker=EdgeReranker.cross_encoder,
        ),
        search_filter=SearchFilters(edge_uuids=[]),
    )

    assert edges == []
    assert scores == []
    fulltext_search.assert_not_awaited()
    similarity_search.assert_not_awaited()
    bfs_search.assert_not_awaited()
    cross_encoder.rank.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'edge_uuids',
    [
        None,
        ['edge-1'],
    ],
)
async def test_edge_search_keeps_non_empty_filter_semantics(monkeypatch, edge_uuids):
    fulltext_search = AsyncMock(return_value=[])
    monkeypatch.setattr(
        'graphiti_core.search.search.edge_fulltext_search',
        fulltext_search,
    )
    driver = SimpleNamespace()
    search_filter = SearchFilters(edge_uuids=edge_uuids)

    edges, scores = await edge_search(
        driver=driver,
        cross_encoder=SimpleNamespace(),
        query='known edge',
        query_vector=[0.0],
        group_ids=None,
        config=EdgeSearchConfig(search_methods=[EdgeSearchMethod.bm25]),
        search_filter=search_filter,
    )

    assert edges == []
    assert scores == []
    fulltext_search.assert_awaited_once_with(
        driver,
        'known edge',
        search_filter,
        None,
        20,
    )


@pytest.mark.asyncio
async def test_empty_edge_uuid_filter_does_not_skip_node_search(monkeypatch):
    edge_fulltext_search = AsyncMock(return_value=[])
    node = EntityNode(
        uuid='node-1',
        name='Alice',
        labels=['Entity'],
        group_id='group_1',
    )
    node_fulltext_search = AsyncMock(return_value=[node])

    monkeypatch.setattr(
        'graphiti_core.search.search.edge_fulltext_search',
        edge_fulltext_search,
    )
    monkeypatch.setattr(
        'graphiti_core.search.search.node_fulltext_search',
        node_fulltext_search,
    )

    clients = SimpleNamespace(
        driver=SimpleNamespace(),
        embedder=SimpleNamespace(),
        cross_encoder=SimpleNamespace(),
    )
    search_filter = SearchFilters(edge_uuids=[])

    results = await search(
        clients,
        query='Alice',
        group_ids=['group_1'],
        config=SearchConfig(
            edge_config=EdgeSearchConfig(search_methods=[EdgeSearchMethod.bm25]),
            node_config=NodeSearchConfig(search_methods=[NodeSearchMethod.bm25]),
        ),
        search_filter=search_filter,
    )

    assert results.edges == []
    assert results.nodes == [node]
    edge_fulltext_search.assert_not_awaited()
    node_fulltext_search.assert_awaited_once()
