from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search import episode_search
from graphiti_core.search.search_config import (
    EpisodeReranker,
    EpisodeSearchConfig,
    EpisodeSearchMethod,
)
from graphiti_core.search.search_filters import SearchFilters


def _episode(uuid: str, content: str) -> EpisodicNode:
    return EpisodicNode(
        uuid=uuid,
        name=uuid,
        group_id='group_1',
        source=EpisodeType.text,
        source_description='test',
        content=content,
        valid_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('limit', [1, 2])
async def test_episode_cross_encoder_shortlist_preserves_extra_candidates(monkeypatch, limit):
    """Cross-encoder sees the full RRF shortlist before the final result limit."""
    episodes = [_episode(f'episode-{i}', f'candidate-{i}') for i in range(2 * limit)]
    best_episode = episodes[-1]
    ranked_contents: list[str] = []

    async def fake_fulltext(*args, **kwargs):
        return episodes

    class RecordingCrossEncoder:
        async def rank(self, query: str, passages: list[str]):
            ranked_contents.extend(passages)
            scored = [
                (content, 0.99 if content == best_episode.content else 0.1) for content in passages
            ]
            scored.sort(key=lambda item: item[1], reverse=True)
            return scored

    monkeypatch.setattr('graphiti_core.search.search.episode_fulltext_search', fake_fulltext)

    result, scores = await episode_search(
        driver=SimpleNamespace(),
        cross_encoder=RecordingCrossEncoder(),
        query='find the best episode',
        _query_vector=[0.1, 0.2, 0.3],
        group_ids=None,
        config=EpisodeSearchConfig(
            search_methods=[EpisodeSearchMethod.bm25],
            reranker=EpisodeReranker.cross_encoder,
        ),
        search_filter=SearchFilters(),
        limit=limit,
    )

    assert best_episode.content in ranked_contents
    assert len(ranked_contents) == 2 * limit
    assert result[0].uuid == best_episode.uuid
    assert len(result) == limit
    assert len(scores) == limit
