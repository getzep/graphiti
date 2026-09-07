from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.search.search import edge_search
from graphiti_core.search.search_config import EdgeReranker, EdgeSearchConfig, EdgeSearchMethod
from graphiti_core.search.search_filters import SearchFilters


def _edge(uuid: str, fact: str) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        source_node_uuid='source',
        target_node_uuid='target',
        name='relates_to',
        group_id='group_1',
        fact=fact,
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_rrf_shortlist_includes_cosine_top_hit(monkeypatch):
    """RRF + 2*limit shortlist: cosine rank-1 reaches CE even when BM25 list is long."""
    limit = 2
    bm25_hits = [_edge(f'bm25-{i}', f'bm25-fact-{i}') for i in range(6)]
    cosine_hit = _edge('cosine-1', 'cosine-top-fact')
    ranked_passages: list[str] = []

    async def fake_fulltext(*args, **kwargs):
        return bm25_hits

    async def fake_similarity(*args, **kwargs):
        return [cosine_hit]

    class RecordingCrossEncoder:
        async def rank(self, query: str, passages: list[str]):
            ranked_passages.extend(passages)
            scored = [(p, 0.99 if p == cosine_hit.fact else 0.1) for p in passages]
            scored.sort(key=lambda item: item[1], reverse=True)
            return scored

    monkeypatch.setattr('graphiti_core.search.search.edge_fulltext_search', fake_fulltext)
    monkeypatch.setattr('graphiti_core.search.search.edge_similarity_search', fake_similarity)

    edges, scores = await edge_search(
        driver=SimpleNamespace(),
        cross_encoder=RecordingCrossEncoder(),
        query='find the cosine top hit',
        query_vector=[0.1, 0.2, 0.3],
        group_ids=None,
        config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.cross_encoder,
        ),
        search_filter=SearchFilters(),
        limit=limit,
    )

    assert cosine_hit.fact in ranked_passages
    assert len(ranked_passages) == 2 * limit
    assert len(ranked_passages) > limit
    assert edges[0].uuid == cosine_hit.uuid
    assert len(edges) == limit
    assert len(scores) == limit


@pytest.mark.asyncio
async def test_smoke_alice_works_at_zep_reaches_cross_encoder(monkeypatch):
    """Smoke: BM25 fills weak hits; cosine has the real answer.

    Query: "Where does Alice work?"
    BM25:  "Alice likes coffee", "Alice lives in SF"  (and more)
    Cosine: "Alice works at Zep"

    Before the fix, CE only saw the first `limit` BM25 facts.
    After, RRF shortlists up to `2 * limit` candidates so Zep reaches CE and can rank #1.
    """
    limit = 2
    coffee = _edge('bm25-1', 'Alice likes coffee')
    lives_in_sf = _edge('bm25-2', 'Alice lives in SF')
    extra_bm25 = [_edge(f'bm25-{i}', f'Alice unrelated {i}') for i in range(3, 5)]
    works_at_zep = _edge('cosine-1', 'Alice works at Zep')
    ranked_passages: list[str] = []

    async def fake_fulltext(*args, **kwargs):
        return [coffee, lives_in_sf, *extra_bm25]

    async def fake_similarity(*args, **kwargs):
        return [works_at_zep]

    class RecordingCrossEncoder:
        async def rank(self, query: str, passages: list[str]):
            ranked_passages.extend(passages)
            scored = [(p, 0.99 if p == works_at_zep.fact else 0.1) for p in passages]
            scored.sort(key=lambda item: item[1], reverse=True)
            return scored

    monkeypatch.setattr('graphiti_core.search.search.edge_fulltext_search', fake_fulltext)
    monkeypatch.setattr('graphiti_core.search.search.edge_similarity_search', fake_similarity)

    edges, scores = await edge_search(
        driver=SimpleNamespace(),
        cross_encoder=RecordingCrossEncoder(),
        query='Where does Alice work?',
        query_vector=[0.1, 0.2, 0.3],
        group_ids=None,
        config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.cross_encoder,
        ),
        search_filter=SearchFilters(),
        limit=limit,
    )

    assert works_at_zep.fact in ranked_passages
    assert len(ranked_passages) <= 2 * limit
    assert edges[0].uuid == works_at_zep.uuid
    assert edges[0].fact == 'Alice works at Zep'
    assert len(edges) == limit
    assert len(scores) == limit
