from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.search.search import edge_search
from graphiti_core.search.search_config import EdgeReranker, EdgeSearchConfig, EdgeSearchMethod
from graphiti_core.search.search_filters import SearchFilters


def _edge(uuid: str, fact: str, source: str = 'source') -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        source_node_uuid=source,
        target_node_uuid=source + '_target',
        name='relates_to',
        group_id='group_1',
        fact=fact,
        created_at=datetime.now(timezone.utc),
    )


async def _run(monkeypatch, bm25_hits, cosine_hits, limit, score_map, min_score=0.0):
    passages_seen: list[str] = []

    async def fake_fulltext(*args, **kwargs):
        return bm25_hits

    async def fake_similarity(*args, **kwargs):
        return cosine_hits

    class RecordingCrossEncoder:
        async def rank(self, query: str, passages: list[str]):
            passages_seen.extend(passages)
            scored = [(p, score_map.get(p, 0.1)) for p in passages]
            scored.sort(key=lambda item: item[1], reverse=True)
            return scored

    monkeypatch.setattr('graphiti_core.search.search.edge_fulltext_search', fake_fulltext)
    monkeypatch.setattr('graphiti_core.search.search.edge_similarity_search', fake_similarity)

    edges, scores = await edge_search(
        driver=SimpleNamespace(),
        cross_encoder=RecordingCrossEncoder(),
        query='shared fact',
        query_vector=[0.1, 0.2, 0.3],
        group_ids=None,
        config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.cross_encoder,
        ),
        search_filter=SearchFilters(),
        limit=limit,
        reranker_min_score=min_score,
    )
    return edges, scores, passages_seen


SHARED = 'Alice works at Zep'


@pytest.mark.asyncio
async def test_distinct_edges_sharing_one_fact_are_all_returned(monkeypatch):
    """Two edges can carry identical `fact` text while being different edges. The
    cross-encoder rerank keyed its candidate map by fact text, so only the last
    uuid survived and the other edge disappeared from search results."""
    dup_a = _edge('edge-a', SHARED, source='node-a')
    dup_b = _edge('edge-b', SHARED, source='node-b')
    other = _edge('edge-c', 'Bob likes coffee', source='node-c')

    edges, scores, passages = await _run(
        monkeypatch,
        bm25_hits=[dup_a, dup_b, other],
        cosine_hits=[other],
        limit=10,
        score_map={SHARED: 0.9, other.fact: 0.2},
    )

    returned = {e.uuid for e in edges}
    assert returned == {'edge-a', 'edge-b', 'edge-c'}, (
        f'edges sharing one fact were dropped: got {sorted(returned)}'
    )
    assert len(edges) == len(scores), 'edge/score lists must stay index-aligned'
    # one rank() call per unique fact, not per edge
    assert passages == [SHARED, other.fact] or passages == [other.fact, SHARED]
    for edge, score in zip(edges, scores, strict=True):
        if edge.fact == SHARED:
            assert score == 0.9


@pytest.mark.asyncio
async def test_duplicate_fact_scores_stay_aligned_with_min_score(monkeypatch):
    """A shared fact must not shift the score list once the min-score filter runs."""
    dup_a = _edge('edge-a', SHARED, source='node-a')
    dup_b = _edge('edge-b', SHARED, source='node-b')
    weak = _edge('edge-c', 'weak fact', source='node-c')

    edges, scores, _ = await _run(
        monkeypatch,
        bm25_hits=[dup_a, dup_b, weak],
        cosine_hits=[weak],
        limit=10,
        score_map={SHARED: 0.9, weak.fact: 0.2},
        min_score=0.5,
    )

    assert {e.uuid for e in edges} == {'edge-a', 'edge-b'}
    assert scores == [0.9, 0.9]


@pytest.mark.asyncio
async def test_three_edges_sharing_one_fact_all_survive(monkeypatch):
    trio = [_edge(f'edge-{i}', SHARED, source=f'node-{i}') for i in range(3)]
    edges, scores, _ = await _run(
        monkeypatch,
        bm25_hits=trio,
        cosine_hits=[],
        limit=10,
        score_map={SHARED: 0.8},
    )
    assert {e.uuid for e in edges} == {'edge-0', 'edge-1', 'edge-2'}
    assert scores == [0.8, 0.8, 0.8]


@pytest.mark.asyncio
async def test_limit_still_caps_expanded_duplicates(monkeypatch):
    trio = [_edge(f'edge-{i}', SHARED, source=f'node-{i}') for i in range(3)]
    edges, scores, _ = await _run(
        monkeypatch,
        bm25_hits=trio,
        cosine_hits=[],
        limit=2,
        score_map={SHARED: 0.8},
    )
    assert len(edges) == 2
    assert len(scores) == 2
