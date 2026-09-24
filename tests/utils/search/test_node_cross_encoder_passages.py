from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from graphiti_core.nodes import EntityNode
from graphiti_core.search.search import node_search
from graphiti_core.search.search_config import NodeReranker, NodeSearchConfig, NodeSearchMethod
from graphiti_core.search.search_filters import SearchFilters


def _node(uuid: str, name: str, summary: str = '') -> EntityNode:
    return EntityNode(
        uuid=uuid,
        name=name,
        group_id='group_1',
        labels=['Entity'],
        summary=summary,
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_node_cross_encoder_sees_the_summary(monkeypatch):
    """The answer lives in a summary; the name alone would give the cross-encoder nothing."""
    leader = _node('n-leader', 'Kim Minsu', 'leads the search team')
    decoy = _node('n-decoy', 'Search Team', 'was founded in 2026')  # lexical decoy on the name
    nameless_context = _node('n-plain', 'Scope Labs')  # no summary at all
    ranked_passages: list[str] = []

    async def fake_fulltext(*args, **kwargs):
        return [decoy, nameless_context]

    async def fake_similarity(*args, **kwargs):
        return [leader]

    class RecordingCrossEncoder:
        async def rank(self, query: str, passages: list[str]):
            ranked_passages.extend(passages)
            scored = [(p, 0.99 if 'leads the search team' in p else 0.1) for p in passages]
            scored.sort(key=lambda item: item[1], reverse=True)
            return scored

    monkeypatch.setattr('graphiti_core.search.search.node_fulltext_search', fake_fulltext)
    monkeypatch.setattr('graphiti_core.search.search.node_similarity_search', fake_similarity)

    nodes, scores = await node_search(
        driver=SimpleNamespace(),
        cross_encoder=RecordingCrossEncoder(),
        query='who leads the search team?',
        query_vector=[0.1, 0.2, 0.3],
        group_ids=None,
        config=NodeSearchConfig(
            search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
            reranker=NodeReranker.cross_encoder,
        ),
        search_filter=SearchFilters(),
        limit=3,
    )

    assert 'Kim Minsu: leads the search team' in ranked_passages
    assert 'Search Team: was founded in 2026' in ranked_passages
    assert 'Scope Labs' in ranked_passages  # no summary -> name only, no dangling colon
    assert nodes[0].uuid == leader.uuid
    assert len(nodes) == 3
    assert len(scores) == 3
