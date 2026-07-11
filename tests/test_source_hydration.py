"""Tests for source hydration (redevops fork, G1).

The pure collection/attachment logic is exercised without a DB driver; the async
Graphiti.hydrate_edge_sources fetch is covered with a stubbed EpisodicNode.get_by_uuids so it runs
offline. Verifies the invariant: after hydration an edge exposes the raw source turns behind its
lossy extracted fact, and duplicate sources are surfaced once, in order.
"""

from datetime import datetime, timezone

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.source_hydration import (
    attach_source_episodes,
    collect_episode_uuids,
    deduped_source_episodes,
)

NOW = datetime.now(timezone.utc)
GROUP = 'test_group'


def _episode(name: str, content: str) -> EpisodicNode:
    return EpisodicNode(
        name=name,
        group_id=GROUP,
        source=EpisodeType.message,
        source_description='chat',
        content=content,
        valid_at=NOW,
        created_at=NOW,
    )


def _edge(fact: str, episode_uuids: list[str]) -> EntityEdge:
    return EntityEdge(
        source_node_uuid='src',
        target_node_uuid='tgt',
        created_at=NOW,
        name='relates_to',
        fact=fact,
        episodes=episode_uuids,
        group_id=GROUP,
    )


def test_collect_episode_uuids_dedupes_in_order():
    ep_a, ep_b, ep_c = 'a', 'b', 'c'
    edges = [_edge('f1', [ep_a, ep_b]), _edge('f2', [ep_b, ep_c]), _edge('f3', [])]
    assert collect_episode_uuids(edges) == [ep_a, ep_b, ep_c]


def test_attach_source_episodes_hydrates_and_ignores_missing():
    ep1 = _episode('s1', 'user: I moved to Berlin in March')
    ep2 = _episode('s2', 'user: actually it was April')
    edge = _edge('user moved to Berlin', [ep1.uuid, ep2.uuid, 'missing-uuid'])

    attach_source_episodes([edge], {ep1.uuid: ep1, ep2.uuid: ep2})

    assert edge.source_episodes is not None
    assert [e.uuid for e in edge.source_episodes] == [ep1.uuid, ep2.uuid]  # missing uuid skipped


def test_hydrated_text_recovers_raw_turns_else_falls_back():
    ep1 = _episode('s1', 'user: I moved to Berlin in March')
    ep2 = _episode('s2', 'user: actually it was April')
    edge = _edge('user moved to Berlin', [ep1.uuid, ep2.uuid])

    # before hydration → the lossy fact
    assert edge.hydrated_text() == 'user moved to Berlin'

    attach_source_episodes([edge], {ep1.uuid: ep1, ep2.uuid: ep2})
    text = edge.hydrated_text()
    assert 'March' in text and 'April' in text          # the raw, non-lossy turns
    assert edge.hydrated_text(max_chars=10) == text[:10]


def test_source_episodes_is_transient_not_persisted():
    ep1 = _episode('s1', 'user: hello')
    edge = _edge('greeting', [ep1.uuid])
    attach_source_episodes([edge], {ep1.uuid: ep1})
    # excluded from serialization so it never reaches the DB write path
    assert 'source_episodes' not in edge.model_dump()


def test_deduped_source_episodes_across_edges():
    ep1 = _episode('s1', 'turn one')
    ep2 = _episode('s2', 'turn two')
    e1 = _edge('f1', [ep1.uuid])
    e2 = _edge('f2', [ep1.uuid, ep2.uuid])          # ep1 shared
    attach_source_episodes([e1, e2], {ep1.uuid: ep1, ep2.uuid: ep2})

    sources = deduped_source_episodes([e1, e2])
    assert [s.uuid for s in sources] == [ep1.uuid, ep2.uuid]   # ep1 surfaced once, in order


@pytest.mark.asyncio
async def test_graphiti_hydrate_edge_sources_batch_fetches(monkeypatch):
    ep1 = _episode('s1', 'user: I moved to Berlin in March')
    ep2 = _episode('s2', 'user: actually it was April')
    edge = _edge('user moved to Berlin', [ep1.uuid, ep2.uuid])

    captured = {}

    async def fake_get_by_uuids(driver, uuids):
        captured['uuids'] = uuids                    # assert a single batched fetch
        pool = {ep1.uuid: ep1, ep2.uuid: ep2}
        return [pool[u] for u in uuids if u in pool]

    monkeypatch.setattr(EpisodicNode, 'get_by_uuids', staticmethod(fake_get_by_uuids))

    # a minimal stand-in for Graphiti — exercise the method without constructing a full client/driver
    from graphiti_core.graphiti import Graphiti

    class _Stub:
        driver = object()
        hydrate_edge_sources = Graphiti.hydrate_edge_sources

    out = await _Stub().hydrate_edge_sources([edge])
    assert captured['uuids'] == [ep1.uuid, ep2.uuid]          # one round-trip, deduped
    assert 'March' in out[0].hydrated_text() and 'April' in out[0].hydrated_text()
