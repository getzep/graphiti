from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graphiti_core.search.search import edge_search
from graphiti_core.search.search_config import EdgeReranker, EdgeSearchConfig, EdgeSearchMethod
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
