from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.search.search import edge_search
from graphiti_core.search.search_config import EdgeReranker, EdgeSearchConfig, EdgeSearchMethod
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import balanced_merge


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


def test_balanced_merge_round_robin_and_bug_repro():
    # Interleaves methods and keeps later-method hits when an early list is long.
    bm25 = [_edge(f'b{i}', f'bm25-{i}') for i in range(4)]
    cosine = [_edge('c1', 'cosine-1')]
    bfs = [_edge('f1', 'bfs-1')]

    assert [e.uuid for e in balanced_merge([bm25, cosine, bfs], limit=3)] == ['b0', 'c1', 'f1']
    assert [e.uuid for e in balanced_merge([bm25, cosine], limit=2)] == ['b0', 'c1']


def test_balanced_merge_dedupe_and_empty_lists():
    shared = _edge('shared', 'shared-fact')
    bm25 = [shared, _edge('b2', 'bm25-2')]
    cosine = [_edge('shared', 'shared-fact'), _edge('c2', 'cosine-2')]

    assert [e.uuid for e in balanced_merge([bm25, cosine], limit=2)] == ['shared', 'b2']
    assert [e.uuid for e in balanced_merge([[], [_edge('e1', 'only')], []], limit=2)] == ['e1']
    assert balanced_merge([[], []], limit=5) == []
    assert balanced_merge([], limit=5) == []
    assert balanced_merge([[_edge('e1', 'fact')]], limit=0) == []


@pytest.mark.asyncio
async def test_smoke_alice_works_at_zep_reaches_cross_encoder(monkeypatch):
    """Smoke: BM25 fills `limit` with weak hits; cosine has the real answer.

    Query: "Where does Alice work?"
    BM25:  "Alice likes coffee", "Alice lives in SF"  (and more)
    Cosine: "Alice works at Zep"

    Before the fix, CE only saw the first `limit` BM25 facts.
    After, balanced_merge yields candidates [coffee, works at Zep]; CE can rank Zep #1.
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

    assert ranked_passages == [coffee.fact, works_at_zep.fact]
    assert edges[0].uuid == works_at_zep.uuid
    assert edges[0].fact == 'Alice works at Zep'
    assert len(edges) == limit
    assert len(scores) == limit
