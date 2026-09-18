from types import SimpleNamespace

import pytest

from graphiti_core.search.search import edge_search
from graphiti_core.search.search_config import (
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
)
from graphiti_core.search.search_filters import SearchFilters


@pytest.mark.asyncio
async def test_edge_search_episode_mentions_keeps_scores_aligned(monkeypatch):
    """The episode_mentions reranker re-sorts edges by episode count. Its scores
    must be re-sorted in lockstep, otherwise the parallel (edges, scores) result
    pairs every edge with a neighbour's score."""
    # edge_b has more episodes, so episode_mentions ranks it first; rrf ranks
    # edge_a first (higher score). The two orderings differ, exposing the bug.
    edge_a = SimpleNamespace(uuid='uuid_a', episodes=['ep1'])
    edge_b = SimpleNamespace(uuid='uuid_b', episodes=['ep1', 'ep2', 'ep3'])

    async def fake_fulltext(*args, **kwargs):
        return [edge_a, edge_b]

    def fake_rrf(search_result_uuids, min_score=0, rank_const=1):
        return ['uuid_a', 'uuid_b'], [0.9, 0.5]

    monkeypatch.setattr('graphiti_core.search.search.edge_fulltext_search', fake_fulltext)
    monkeypatch.setattr('graphiti_core.search.search.rrf', fake_rrf)

    edges, scores = await edge_search(
        driver=SimpleNamespace(),
        cross_encoder=SimpleNamespace(),
        query='q',
        query_vector=[],
        group_ids=None,
        config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25],
            reranker=EdgeReranker.episode_mentions,
        ),
        search_filter=SearchFilters(),
    )

    # episode_mentions orders by descending episode count: edge_b first.
    assert [edge.uuid for edge in edges] == ['uuid_b', 'uuid_a']
    # Each edge must keep its own rrf score: edge_b -> 0.5, edge_a -> 0.9.
    assert scores == [0.5, 0.9]
