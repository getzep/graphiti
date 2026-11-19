"""Unit tests for graphiti_core.search.search_utils.get_episodes_by_mentions.

The helper fetches at most `limit` episodes for the given edges. An episode is
listed in `edge.episodes` of every fact that mentions it, so the collected UUIDs
repeat. The limit therefore has to count distinct episodes, not references:
applying it to the raw list fetches fewer episodes than the caller asked for and
silently drops the ones that only appear after the cut.
"""

from datetime import datetime, timezone

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_utils import get_episodes_by_mentions

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _episode(index: int) -> EpisodicNode:
    return EpisodicNode(
        uuid=f'episode-{index}',
        name=f'episode {index}',
        group_id='group',
        source=EpisodeType.text,
        source_description='test',
        content=f'content {index}',
        valid_at=NOW,
    )


def _edge(episodes: list[EpisodicNode]) -> EntityEdge:
    return EntityEdge(
        source_node_uuid='source',
        target_node_uuid='target',
        name='RELATES_TO',
        fact='a fact',
        group_id='group',
        created_at=NOW,
        episodes=[episode.uuid for episode in episodes],
    )


def _patch_get_by_uuids(monkeypatch, episodes: list[EpisodicNode], requested: list[str]) -> None:
    async def fake_get_by_uuids(driver, uuids):
        requested.extend(uuids)
        return [episode for episode in episodes if episode.uuid in uuids]

    monkeypatch.setattr(EpisodicNode, 'get_by_uuids', fake_get_by_uuids)


@pytest.mark.asyncio
async def test_limit_counts_distinct_episodes_not_mentions(monkeypatch):
    episodes = [_episode(index) for index in range(10)]
    # Two facts mention the same five episodes; five further episodes are mentioned once each.
    edges = [_edge(episodes[0:5]), _edge(episodes[0:5])] + [
        _edge([episode]) for episode in episodes[5:10]
    ]
    requested: list[str] = []
    _patch_get_by_uuids(monkeypatch, episodes, requested)

    found = await get_episodes_by_mentions(None, [], edges, limit=10)

    assert len(requested) == len(set(requested)) == 10
    assert [episode.uuid for episode in found] == [episode.uuid for episode in episodes]


@pytest.mark.asyncio
async def test_limit_truncates_after_dedup_and_keeps_first_seen_order(monkeypatch):
    episodes = [_episode(index) for index in range(4)]
    edges = [_edge([episodes[0], episodes[1]]), _edge([episodes[1], episodes[2]])]
    requested: list[str] = []
    _patch_get_by_uuids(monkeypatch, episodes, requested)

    found = await get_episodes_by_mentions(None, [], edges, limit=2)

    assert requested == ['episode-0', 'episode-1']
    assert [episode.uuid for episode in found] == ['episode-0', 'episode-1']


@pytest.mark.asyncio
async def test_no_edges_fetches_nothing(monkeypatch):
    requested: list[str] = []
    _patch_get_by_uuids(monkeypatch, [], requested)

    found = await get_episodes_by_mentions(None, [], [], limit=10)

    assert requested == []
    assert found == []
