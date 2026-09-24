from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from graphiti_core.nodes import CommunityNode
from graphiti_core.search.search import community_search
from graphiti_core.search.search_config import (
    CommunityReranker,
    CommunitySearchConfig,
    CommunitySearchMethod,
)


def _community(uuid: str, name: str, summary: str = '') -> CommunityNode:
    return CommunityNode(
        uuid=uuid,
        name=name,
        group_id='group_1',
        summary=summary,
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_community_cross_encoder_sees_the_summary(monkeypatch):
    target = _community('c-search', 'Cluster 7', 'people and work around the search team')
    decoy = _community('c-design', 'Search', 'the design team and its tooling')
    ranked_passages: list[str] = []

    async def fake_fulltext(*args, **kwargs):
        return [decoy]

    async def fake_similarity(*args, **kwargs):
        return [target]

    class RecordingCrossEncoder:
        async def rank(self, query: str, passages: list[str]):
            ranked_passages.extend(passages)
            scored = [(p, 0.99 if 'around the search team' in p else 0.1) for p in passages]
            scored.sort(key=lambda item: item[1], reverse=True)
            return scored

    monkeypatch.setattr('graphiti_core.search.search.community_fulltext_search', fake_fulltext)
    monkeypatch.setattr('graphiti_core.search.search.community_similarity_search', fake_similarity)

    communities, scores = await community_search(
        driver=SimpleNamespace(),
        cross_encoder=RecordingCrossEncoder(),
        query='what is going on around the search team?',
        query_vector=[0.1, 0.2, 0.3],
        group_ids=None,
        config=CommunitySearchConfig(
            search_methods=[CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
            reranker=CommunityReranker.cross_encoder,
        ),
        limit=2,
    )

    assert 'Cluster 7: people and work around the search team' in ranked_passages
    assert communities[0].uuid == target.uuid
    assert len(communities) == 2
    assert len(scores) == 2
