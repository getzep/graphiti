"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Spanner node operations (entity, episode, community, saga) against the emulator; see conftest.py.
#
#   SPANNER_EMULATOR_HOST=localhost:9010 pytest tests/driver/spanner/test_node_ops_int.py -q

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import pytest

from graphiti_core.driver.spanner import SpannerDriver
from graphiti_core.driver.spanner.mutations import upsert
from graphiti_core.driver.spanner.rows import (
    ENTITY_EDGE_COLUMNS,
    ENTITY_NODE_COLUMNS,
    EPISODIC_EDGE_COLUMNS,
    HAS_EPISODE_EDGE_COLUMNS,
    entity_edge_to_row,
    entity_node_to_row,
    episodic_edge_to_row,
    has_episode_edge_to_row,
)
from graphiti_core.edges import EntityEdge, EpisodicEdge, HasEpisodeEdge
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode, SagaNode
from tests.driver.spanner.conftest import TEST_EMBEDDING_DIM

# Emulator-backed, like the repository's database-backed suites; skipped without the
# emulator by the fixtures, so no `integration` marker (which would exclude them from CI).
pytestmark = pytest.mark.asyncio(loop_scope='session')

T0 = datetime(2026, 10, 28, 12, 30, 15, 123456, tzinfo=timezone.utc)
# Spanner caps mutations per commit, so large seeds are applied in chunks of this many rows.
SEED_CHUNK = 250


def _embedding(offset: int = 0) -> list[float]:
    """A deterministic embedding of the pinned dimension, distinct per ``offset``."""
    return [float((i + offset) % 7) / 7 for i in range(TEST_EMBEDDING_DIM)]


def _group() -> str:
    return f'group_{uuid4().hex[:8]}'


def _entity(group_id: str, name: str = 'Alice', **kwargs: Any) -> EntityNode:
    return EntityNode(
        name=name, group_id=group_id, summary=f'{name} summary', created_at=T0, **kwargs
    )


def _episode(group_id: str, minutes: int = 0, **kwargs: Any) -> EpisodicNode:
    fields: dict[str, Any] = {
        'name': f'episode {minutes}',
        'group_id': group_id,
        'source': EpisodeType.message,
        'source_description': 'chat',
        'content': f'content {minutes}',
        'valid_at': T0 + timedelta(minutes=minutes),
        'created_at': T0,
        'entity_edges': ['edge-1'],
    }
    return EpisodicNode(**{**fields, **kwargs})


def _community(group_id: str, name: str = 'Team', **kwargs: Any) -> CommunityNode:
    return CommunityNode(
        name=name, group_id=group_id, summary=f'{name} summary', created_at=T0, **kwargs
    )


def _saga(group_id: str, name: str = 'saga') -> SagaNode:
    return SagaNode(name=name, group_id=group_id, summary='a saga', created_at=T0)


def _relates_to(group_id: str, source: EntityNode, target: EntityNode) -> EntityEdge:
    return EntityEdge(
        group_id=group_id,
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        name='KNOWS',
        fact=f'{source.name} knows {target.name}',
        created_at=T0,
    )


def _has_episode(group_id: str, saga: SagaNode, episode: EpisodicNode) -> HasEpisodeEdge:
    return HasEpisodeEdge(
        group_id=group_id, source_node_uuid=saga.uuid, target_node_uuid=episode.uuid, created_at=T0
    )


async def _seed(
    driver: SpannerDriver, table: str, rows: list[dict[str, Any]], columns: Any
) -> None:
    """Upsert ``rows`` directly in a few large commits, bypassing the ops under test."""
    for start in range(0, len(rows), SEED_CHUNK):
        await driver.apply_mutations([upsert(table, rows[start : start + SEED_CHUNK], columns)])


async def _count(driver: SpannerDriver, table: str, where: str = 'TRUE', **params: Any) -> int:
    rows = await driver.read(f'SELECT COUNT(*) AS c FROM {table} WHERE {where}', params)
    return rows[0]['c']


# --- save / get round trips -------------------------------------------------------------------


async def test_entity_round_trip(clean_driver: SpannerDriver):
    ops = clean_driver.entity_node_ops
    node = _entity(
        _group(),
        labels=['Person'],
        attributes={'age': 30, 'city': 'Paris'},
        name_embedding=_embedding(),
    )
    await ops.save(clean_driver, node)

    loaded = await ops.get_by_uuid(clean_driver, node.uuid)

    assert loaded.name == node.name
    assert loaded.group_id == node.group_id
    assert loaded.summary == node.summary
    assert 'Person' in loaded.labels
    assert loaded.attributes == {'age': 30, 'city': 'Paris'}
    assert loaded.created_at == T0
    assert loaded.name_embedding == pytest.approx(node.name_embedding, abs=1e-6)


async def test_episode_round_trip(clean_driver: SpannerDriver):
    ops = clean_driver.episode_node_ops
    node = _episode(_group())
    await ops.save(clean_driver, node)

    loaded = await ops.get_by_uuid(clean_driver, node.uuid)

    assert loaded.content == node.content
    assert loaded.source == EpisodeType.message
    assert loaded.source_description == 'chat'
    assert loaded.entity_edges == ['edge-1']
    assert loaded.valid_at == T0


async def test_community_round_trip(clean_driver: SpannerDriver):
    ops = clean_driver.community_node_ops
    node = _community(_group(), name_embedding=_embedding(1))
    await ops.save(clean_driver, node)

    loaded = await ops.get_by_uuid(clean_driver, node.uuid)

    assert loaded.name == 'Team'
    assert loaded.summary == 'Team summary'
    assert loaded.name_embedding == pytest.approx(node.name_embedding, abs=1e-6)


async def test_saga_round_trip(clean_driver: SpannerDriver):
    ops = clean_driver.saga_node_ops
    node = _saga(_group(), 'onboarding')
    await ops.save(clean_driver, node)

    loaded = await ops.get_by_uuid(clean_driver, node.uuid)

    assert loaded.name == 'onboarding'
    assert loaded.summary == 'a saga'
    assert loaded.first_episode_uuid is None


async def test_get_by_uuid_missing_raises(clean_driver: SpannerDriver):
    for ops in (
        clean_driver.entity_node_ops,
        clean_driver.episode_node_ops,
        clean_driver.community_node_ops,
        clean_driver.saga_node_ops,
    ):
        with pytest.raises(NodeNotFoundError):
            await ops.get_by_uuid(clean_driver, 'missing-uuid')


# --- bulk save, group reads, pagination ----------------------------------------------------------


async def test_entity_save_bulk_across_chunks_and_get_by_uuids(clean_driver: SpannerDriver):
    ops = clean_driver.entity_node_ops
    group = _group()
    nodes = [_entity(group, f'node-{i}') for i in range(250)]

    await ops.save_bulk(clean_driver, nodes, batch_size=100)

    assert await _count(clean_driver, 'Entity', 'group_id = @g', g=group) == 250
    wanted = [n.uuid for n in nodes[:10]]
    loaded = await ops.get_by_uuids(clean_driver, [*wanted, 'not-a-uuid'])
    assert sorted(n.uuid for n in loaded) == sorted(wanted)
    assert await ops.get_by_uuids(clean_driver, []) == []


async def test_entity_get_by_group_ids_limit_and_descending_cursor(clean_driver: SpannerDriver):
    ops = clean_driver.entity_node_ops
    group = _group()
    await ops.save_bulk(clean_driver, [_entity(group, f'n{i}') for i in range(250)])

    page1 = await ops.get_by_group_ids(clean_driver, [group], limit=100)
    page2 = await ops.get_by_group_ids(clean_driver, [group], limit=100, uuid_cursor=page1[-1].uuid)
    everything = await ops.get_by_group_ids(clean_driver, [group])

    ids1 = [n.uuid for n in page1]
    ids2 = [n.uuid for n in page2]
    assert len(ids1) == len(ids2) == 100
    assert ids1 == sorted(ids1, reverse=True)
    assert ids2 == sorted(ids2, reverse=True)
    assert max(ids2) < min(ids1)
    assert len(everything) == 250


async def test_episode_community_saga_group_reads(clean_driver: SpannerDriver):
    group = _group()
    episodes = [_episode(group, i) for i in range(4)]
    communities = [_community(group, f'c{i}') for i in range(3)]
    sagas = [_saga(group, f's{i}') for i in range(3)]
    episode_ops = clean_driver.episode_node_ops
    community_ops = clean_driver.community_node_ops
    saga_ops = clean_driver.saga_node_ops
    await episode_ops.save_bulk(clean_driver, episodes, batch_size=3)
    await community_ops.save_bulk(clean_driver, communities)
    await saga_ops.save_bulk(clean_driver, sagas)

    assert len(await episode_ops.get_by_group_ids(clean_driver, [group], limit=2)) == 2
    assert len(await episode_ops.get_by_uuids(clean_driver, [e.uuid for e in episodes])) == 4
    assert len(await community_ops.get_by_group_ids(clean_driver, [group], limit=2)) == 2
    assert len(await community_ops.get_by_uuids(clean_driver, [c.uuid for c in communities])) == 3
    assert len(await saga_ops.get_by_group_ids(clean_driver, [group], limit=1)) == 1
    assert len(await saga_ops.get_by_group_ids(clean_driver, [group])) == 3


# --- embeddings --------------------------------------------------------------------------------


async def test_entity_load_embeddings(clean_driver: SpannerDriver):
    ops = clean_driver.entity_node_ops
    node = _entity(_group(), name_embedding=_embedding(3))
    await ops.save(clean_driver, node)

    shell = EntityNode(uuid=node.uuid, name='shell', group_id=node.group_id, created_at=T0)
    await ops.load_embeddings(clean_driver, shell)

    assert shell.name_embedding == pytest.approx(_embedding(3), abs=1e-6)


async def test_entity_load_embeddings_bulk(clean_driver: SpannerDriver):
    ops = clean_driver.entity_node_ops
    group = _group()
    nodes = [_entity(group, f'e{i}', name_embedding=_embedding(i)) for i in range(5)]
    await ops.save_bulk(clean_driver, nodes)

    shells = [EntityNode(uuid=n.uuid, name='shell', group_id=group, created_at=T0) for n in nodes]
    await ops.load_embeddings_bulk(clean_driver, shells, batch_size=2)

    for i, shell in enumerate(shells):
        assert shell.name_embedding == pytest.approx(_embedding(i), abs=1e-6)


async def test_community_load_name_embedding(clean_driver: SpannerDriver):
    ops = clean_driver.community_node_ops
    node = _community(_group(), name_embedding=_embedding(2))
    await ops.save(clean_driver, node)

    shell = CommunityNode(uuid=node.uuid, name='shell', group_id=node.group_id, created_at=T0)
    await ops.load_name_embedding(clean_driver, shell)

    assert shell.name_embedding == pytest.approx(_embedding(2), abs=1e-6)


async def test_load_embeddings_missing_node_raises(clean_driver: SpannerDriver):
    group = _group()
    with pytest.raises(NodeNotFoundError):
        await clean_driver.entity_node_ops.load_embeddings(
            clean_driver, EntityNode(name='ghost', group_id=group, created_at=T0)
        )
    with pytest.raises(NodeNotFoundError):
        await clean_driver.community_node_ops.load_name_embedding(
            clean_driver, CommunityNode(name='ghost', group_id=group, created_at=T0)
        )


# --- deletes -----------------------------------------------------------------------------------


async def test_entity_delete_removes_incident_edges_only(clean_driver: SpannerDriver):
    group = _group()
    target, other = _entity(group, 'target'), _entity(group, 'other')
    episode = _episode(group)
    await clean_driver.entity_node_ops.save_bulk(clean_driver, [target, other])
    await clean_driver.episode_node_ops.save(clean_driver, episode)
    relates = _relates_to(group, other, target)
    mentions = EpisodicEdge(
        group_id=group, source_node_uuid=episode.uuid, target_node_uuid=target.uuid, created_at=T0
    )
    await _seed(clean_driver, 'RelatesTo', [entity_edge_to_row(relates)], ENTITY_EDGE_COLUMNS)
    await _seed(clean_driver, 'Mentions', [episodic_edge_to_row(mentions)], EPISODIC_EDGE_COLUMNS)

    await clean_driver.entity_node_ops.delete(clean_driver, target)

    assert await _count(clean_driver, 'RelatesTo', 'group_id = @g', g=group) == 0
    assert await _count(clean_driver, 'Mentions', 'group_id = @g', g=group) == 0
    assert await _count(clean_driver, 'Entity', 'uuid = @u', u=target.uuid) == 0
    assert await _count(clean_driver, 'Entity', 'uuid = @u', u=other.uuid) == 1
    assert await _count(clean_driver, 'Episodic', 'uuid = @u', u=episode.uuid) == 1


async def test_entity_delete_removes_outgoing_edges(clean_driver: SpannerDriver):
    group = _group()
    source, target = _entity(group, 'source'), _entity(group, 'target')
    await clean_driver.entity_node_ops.save_bulk(clean_driver, [source, target])
    edge = _relates_to(group, source, target)
    await _seed(clean_driver, 'RelatesTo', [entity_edge_to_row(edge)], ENTITY_EDGE_COLUMNS)

    await clean_driver.entity_node_ops.delete(clean_driver, source)

    assert await _count(clean_driver, 'RelatesTo', 'group_id = @g', g=group) == 0
    assert await _count(clean_driver, 'Entity', 'uuid = @u', u=target.uuid) == 1


async def test_delete_inside_transaction_is_deferred_until_exit(clean_driver: SpannerDriver):
    ops = clean_driver.entity_node_ops
    node = _entity(_group())
    await ops.save(clean_driver, node)

    async with clean_driver.transaction() as tx:
        await ops.delete(clean_driver, node, tx)
        assert await _count(clean_driver, 'Entity', 'uuid = @u', u=node.uuid) == 1

    assert await _count(clean_driver, 'Entity', 'uuid = @u', u=node.uuid) == 0


async def test_save_inside_transaction_is_deferred_until_exit(clean_driver: SpannerDriver):
    ops = clean_driver.entity_node_ops
    node = _entity(_group())

    async with clean_driver.transaction() as tx:
        await ops.save(clean_driver, node, tx)
        assert await _count(clean_driver, 'Entity', 'uuid = @u', u=node.uuid) == 0

    assert await _count(clean_driver, 'Entity', 'uuid = @u', u=node.uuid) == 1


async def test_delete_by_uuids(clean_driver: SpannerDriver):
    group = _group()
    entities = [_entity(group, f'e{i}') for i in range(5)]
    episodes = [_episode(group, i) for i in range(3)]
    communities = [_community(group, f'c{i}') for i in range(3)]
    sagas = [_saga(group, f's{i}') for i in range(3)]
    await clean_driver.entity_node_ops.save_bulk(clean_driver, entities)
    await clean_driver.episode_node_ops.save_bulk(clean_driver, episodes)
    await clean_driver.community_node_ops.save_bulk(clean_driver, communities)
    await clean_driver.saga_node_ops.save_bulk(clean_driver, sagas)

    await clean_driver.entity_node_ops.delete_by_uuids(
        clean_driver, [e.uuid for e in entities[:3]], batch_size=2
    )
    await clean_driver.entity_node_ops.delete_by_uuids(clean_driver, [])
    await clean_driver.episode_node_ops.delete_by_uuids(clean_driver, [episodes[0].uuid])
    await clean_driver.community_node_ops.delete_by_uuids(clean_driver, [communities[0].uuid])
    await clean_driver.community_node_ops.delete(clean_driver, communities[1])
    await clean_driver.saga_node_ops.delete_by_uuids(clean_driver, [sagas[0].uuid])

    assert await _count(clean_driver, 'Entity', 'group_id = @g', g=group) == 2
    assert await _count(clean_driver, 'Episodic', 'group_id = @g', g=group) == 2
    assert await _count(clean_driver, 'Community', 'group_id = @g', g=group) == 1
    assert await _count(clean_driver, 'Saga', 'group_id = @g', g=group) == 2


async def test_delete_by_group_id_leaves_other_groups_intact(clean_driver: SpannerDriver):
    doomed, kept = _group(), _group()
    for group in (doomed, kept):
        entities = [_entity(group, 'a'), _entity(group, 'b')]
        await clean_driver.entity_node_ops.save_bulk(clean_driver, entities)
        await clean_driver.episode_node_ops.save_bulk(
            clean_driver, [_episode(group, 0), _episode(group, 1)]
        )
        await clean_driver.community_node_ops.save_bulk(
            clean_driver, [_community(group, 'c1'), _community(group, 'c2')]
        )
        await clean_driver.saga_node_ops.save_bulk(
            clean_driver, [_saga(group, 's1'), _saga(group, 's2')]
        )
        edge = _relates_to(group, entities[0], entities[1])
        await _seed(clean_driver, 'RelatesTo', [entity_edge_to_row(edge)], ENTITY_EDGE_COLUMNS)

    await clean_driver.entity_node_ops.delete_by_group_id(clean_driver, doomed)
    await clean_driver.episode_node_ops.delete_by_group_id(clean_driver, doomed)
    await clean_driver.community_node_ops.delete_by_group_id(clean_driver, doomed)
    await clean_driver.saga_node_ops.delete_by_group_id(clean_driver, doomed)

    for table in ('Entity', 'Episodic', 'Community', 'Saga', 'RelatesTo'):
        assert await _count(clean_driver, table, 'group_id = @g', g=doomed) == 0, table
        assert await _count(clean_driver, table, 'group_id = @g', g=kept) > 0, table


async def test_entity_delete_by_group_id_inside_transaction(clean_driver: SpannerDriver):
    group = _group()
    ops = clean_driver.entity_node_ops
    await ops.save_bulk(clean_driver, [_entity(group, f'q{i}') for i in range(5)])

    async with clean_driver.transaction() as tx:
        await ops.delete_by_group_id(clean_driver, group, tx)
        assert await _count(clean_driver, 'Entity', 'group_id = @g', g=group) == 5

    assert await _count(clean_driver, 'Entity', 'group_id = @g', g=group) == 0


async def test_episode_delete_removes_incoming_has_episode_edge(clean_driver: SpannerDriver):
    group = _group()
    saga = _saga(group)
    first, second = _episode(group, 0), _episode(group, 1)
    await clean_driver.saga_node_ops.save(clean_driver, saga)
    await clean_driver.episode_node_ops.save_bulk(clean_driver, [first, second])
    links = [_has_episode(group, saga, ep) for ep in (first, second)]
    await _seed(
        clean_driver,
        'HasEpisode',
        [has_episode_edge_to_row(link) for link in links],
        HAS_EPISODE_EDGE_COLUMNS,
    )

    await clean_driver.episode_node_ops.delete(clean_driver, first)
    assert await _count(clean_driver, 'HasEpisode') == 1

    await clean_driver.saga_node_ops.delete(clean_driver, saga)
    assert await _count(clean_driver, 'HasEpisode') == 0


# --- episode and saga queries ---------------------------------------------------------------------


async def test_retrieve_episodes_ordering_and_filters(clean_driver: SpannerDriver):
    ops = clean_driver.episode_node_ops
    group = _group()
    episodes = [_episode(group, i) for i in range(4)]
    await ops.save_bulk(clean_driver, episodes)
    saga = _saga(group, 'mysaga')
    await clean_driver.saga_node_ops.save(clean_driver, saga)
    links = [_has_episode(group, saga, ep) for ep in (episodes[1], episodes[3])]
    await _seed(
        clean_driver,
        'HasEpisode',
        [has_episode_edge_to_row(link) for link in links],
        HAS_EPISODE_EDGE_COLUMNS,
    )
    later = T0 + timedelta(minutes=10)

    newest = await ops.retrieve_episodes(clean_driver, later, 3, [group])
    assert [e.uuid for e in newest] == [episodes[3].uuid, episodes[2].uuid, episodes[1].uuid]

    as_of = await ops.retrieve_episodes(clean_driver, T0 + timedelta(minutes=1), 10, [group])
    assert [e.uuid for e in as_of] == [episodes[1].uuid, episodes[0].uuid]

    in_saga = await ops.retrieve_episodes(clean_driver, later, 10, [group], saga='mysaga')
    assert [e.uuid for e in in_saga] == [episodes[3].uuid, episodes[1].uuid]
    assert await ops.retrieve_episodes(clean_driver, later, 10, [group], saga='nosuch') == []

    assert await ops.retrieve_episodes(clean_driver, later, 10, [_group()]) == []
    assert len(await ops.retrieve_episodes(clean_driver, later, 10, [group], source='message')) == 4
    assert await ops.retrieve_episodes(clean_driver, later, 10, None, source='text') == []
    assert len(await ops.retrieve_episodes(clean_driver, later, 2)) == 2


async def test_mentions_lookup_and_count(clean_driver: SpannerDriver):
    group = _group()
    mentioned, ignored = _entity(group, 'mentioned'), _entity(group, 'ignored')
    episode = _episode(group)
    await clean_driver.entity_node_ops.save_bulk(clean_driver, [mentioned, ignored])
    await clean_driver.episode_node_ops.save(clean_driver, episode)
    mention = EpisodicEdge(
        group_id=group,
        source_node_uuid=episode.uuid,
        target_node_uuid=mentioned.uuid,
        created_at=T0,
    )
    await _seed(clean_driver, 'Mentions', [episodic_edge_to_row(mention)], EPISODIC_EDGE_COLUMNS)
    ops = clean_driver.episode_node_ops

    found = await ops.get_by_entity_node_uuid(clean_driver, mentioned.uuid)

    assert [e.uuid for e in found] == [episode.uuid]
    assert await ops.get_by_entity_node_uuid(clean_driver, ignored.uuid) == []
    assert await ops.count_mentions_of_entity(clean_driver, mentioned.uuid) == 1
    assert await ops.count_mentions_of_entity(clean_driver, ignored.uuid) == 0


async def test_saga_get_by_name(clean_driver: SpannerDriver):
    ops = clean_driver.saga_node_ops
    group = _group()
    saga = _saga(group, 'mysaga')
    await ops.save(clean_driver, saga)

    hit = await ops.get_by_name(clean_driver, 'mysaga', group)

    assert hit is not None
    assert hit.uuid == saga.uuid
    assert await ops.get_by_name(clean_driver, 'mysaga', _group()) is None
    assert await ops.get_by_name(clean_driver, 'other-name', group) is None


# --- behavioural scenarios from the design spec --------------------------------------------------


async def test_delete_hub_entity_with_thousands_of_incoming_edges(clean_driver: SpannerDriver):
    group = _group()
    hub = _entity(group, 'hub')
    sources = [_entity(group, f'src-{i}') for i in range(3000)]
    await clean_driver.entity_node_ops.save(clean_driver, hub)
    await _seed(
        clean_driver, 'Entity', [entity_node_to_row(s) for s in sources], ENTITY_NODE_COLUMNS
    )
    await _seed(
        clean_driver,
        'RelatesTo',
        [entity_edge_to_row(_relates_to(group, s, hub)) for s in sources],
        ENTITY_EDGE_COLUMNS,
    )
    assert await _count(clean_driver, 'RelatesTo', 'target_node_uuid = @u', u=hub.uuid) == 3000

    await clean_driver.entity_node_ops.delete(clean_driver, hub)

    assert await _count(clean_driver, 'RelatesTo', 'target_node_uuid = @u', u=hub.uuid) == 0
    assert await _count(clean_driver, 'Entity', 'uuid = @u', u=hub.uuid) == 0
    assert await _count(clean_driver, 'Entity', 'group_id = @g', g=group) == 3000


async def test_save_bulk_thousand_entities_with_embeddings_and_attributes(
    clean_driver: SpannerDriver,
):
    group = _group()
    nodes = [
        _entity(
            group,
            f'node-{i}',
            labels=['Person'],
            attributes={'index': i, 'tags': ['a', 'b'], 'nested': {'k': 'v'}},
            name_embedding=_embedding(i),
        )
        for i in range(1000)
    ]

    await clean_driver.entity_node_ops.save_bulk(clean_driver, nodes)

    loaded = await clean_driver.entity_node_ops.get_by_group_ids(clean_driver, [group])
    assert len(loaded) == 1000
    sample = next(n for n in loaded if n.uuid == nodes[500].uuid)
    assert sample.attributes == {'index': 500, 'tags': ['a', 'b'], 'nested': {'k': 'v'}}
    assert sample.name_embedding == pytest.approx(_embedding(500), abs=1e-6)
