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

# The Spanner edge operations (entity, episodic, community, has-episode, next-episode edges)
# against the Spanner emulator (or a real instance); see conftest.py.
#
#   SPANNER_EMULATOR_HOST=localhost:9010 pytest tests/driver/spanner/test_edge_ops_int.py -q

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import pytest

from graphiti_core.driver.spanner import SpannerDriver
from graphiti_core.driver.spanner.errors import SpannerMissingEndpointError
from graphiti_core.driver.spanner.mutations import upsert
from graphiti_core.driver.spanner.rows import (
    community_node_to_row,
    entity_node_to_row,
    episodic_node_to_row,
    saga_node_to_row,
)
from graphiti_core.driver.spanner.schema import (
    COMMUNITY_TABLE,
    ENTITY_TABLE,
    EPISODIC_TABLE,
    HAS_EPISODE_TABLE,
    HAS_MEMBER_TABLE,
    MENTIONS_TABLE,
    NEXT_EPISODE_TABLE,
    RELATES_TO_TABLE,
    SAGA_TABLE,
    Table,
)
from graphiti_core.edges import (
    CommunityEdge,
    Edge,
    EntityEdge,
    EpisodicEdge,
    HasEpisodeEdge,
    NextEpisodeEdge,
)
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode, SagaNode
from tests.driver.spanner.conftest import TEST_EMBEDDING_DIM

# Emulator-backed, like the repository's database-backed suites; skipped without the
# emulator by the fixtures, so no `integration` marker (which would exclude them from CI).
pytestmark = pytest.mark.asyncio(loop_scope='session')

T0 = datetime(2026, 10, 28, 12, 30, 15, 123456, tzinfo=timezone.utc)
T1 = T0 + timedelta(hours=1)
T2 = T0 + timedelta(hours=2)
T3 = T0 + timedelta(hours=3)
# 0.5 and 0.25 are exact in FLOAT32, so embeddings compare equal after a round trip.
EMBEDDING = [0.5] * (TEST_EMBEDDING_DIM - 1) + [0.25]
OTHER_EMBEDDING = [1.0] * (TEST_EMBEDDING_DIM - 1) + [2.0]


def _group() -> str:
    return f'group_{uuid4().hex[:8]}'


async def _put(driver: SpannerDriver, table: Table, row: dict[str, Any]) -> None:
    await driver.apply_mutations([upsert(table.name, [row], table.writable_columns)])


async def _entity(driver: SpannerDriver, group_id: str, name: str = 'node') -> EntityNode:
    node = EntityNode(name=name, group_id=group_id, created_at=T0)
    await _put(driver, ENTITY_TABLE, entity_node_to_row(node))
    return node


async def _entities(driver: SpannerDriver, group_id: str, count: int) -> list[EntityNode]:
    return [await _entity(driver, group_id, f'node{i}') for i in range(count)]


async def _episode(driver: SpannerDriver, group_id: str, name: str = 'episode') -> EpisodicNode:
    node = EpisodicNode(
        name=name,
        group_id=group_id,
        created_at=T0,
        source=EpisodeType.text,
        source_description='test',
        content='content',
        valid_at=T0,
    )
    await _put(driver, EPISODIC_TABLE, episodic_node_to_row(node))
    return node


async def _community(driver: SpannerDriver, group_id: str) -> CommunityNode:
    node = CommunityNode(name='community', group_id=group_id, created_at=T0)
    await _put(driver, COMMUNITY_TABLE, community_node_to_row(node))
    return node


async def _saga(driver: SpannerDriver, group_id: str) -> SagaNode:
    node = SagaNode(name='saga', group_id=group_id, created_at=T0)
    await _put(driver, SAGA_TABLE, saga_node_to_row(node))
    return node


def _edge(group_id: str, source: str, target: str, **kwargs: Any) -> EntityEdge:
    return EntityEdge(
        group_id=group_id,
        source_node_uuid=source,
        target_node_uuid=target,
        created_at=T0,
        name=kwargs.pop('name', 'KNOWS'),
        fact=kwargs.pop('fact', 'a fact'),
        episodes=kwargs.pop('episodes', []),
        **kwargs,
    )


def _edges(group_id: str, source: str, target: str, count: int, **kwargs: Any) -> list[EntityEdge]:
    return [_edge(group_id, source, target, fact=f'fact {i}', **kwargs) for i in range(count)]


# --- the four edge kinds that carry no fields beyond the endpoints ---


@dataclass
class Kind:
    name: str
    ops: Callable[[SpannerDriver], Any]
    make: Callable[[str, str, str], Edge]
    # Seeds a source and a target node, returning their uuids.
    seed: Callable[[SpannerDriver, str], Awaitable[tuple[str, str]]]
    table: str
    # The community edge operations (like the abstract interface) have no save_bulk.
    has_bulk: bool = True


async def _seed_episodic(driver: SpannerDriver, group_id: str) -> tuple[str, str]:
    return (await _episode(driver, group_id)).uuid, (await _entity(driver, group_id)).uuid


async def _seed_community(driver: SpannerDriver, group_id: str) -> tuple[str, str]:
    return (await _community(driver, group_id)).uuid, (await _entity(driver, group_id)).uuid


async def _seed_has_episode(driver: SpannerDriver, group_id: str) -> tuple[str, str]:
    return (await _saga(driver, group_id)).uuid, (await _episode(driver, group_id)).uuid


async def _seed_next_episode(driver: SpannerDriver, group_id: str) -> tuple[str, str]:
    first = await _episode(driver, group_id, 'first')
    second = await _episode(driver, group_id, 'second')
    return first.uuid, second.uuid


SIMPLE_KINDS = [
    Kind(
        'episodic',
        lambda d: d.episodic_edge_ops,
        lambda g, s, t: EpisodicEdge(
            group_id=g, source_node_uuid=s, target_node_uuid=t, created_at=T0
        ),
        _seed_episodic,
        MENTIONS_TABLE.name,
    ),
    Kind(
        'community',
        lambda d: d.community_edge_ops,
        lambda g, s, t: CommunityEdge(
            group_id=g, source_node_uuid=s, target_node_uuid=t, created_at=T0
        ),
        _seed_community,
        HAS_MEMBER_TABLE.name,
        has_bulk=False,
    ),
    Kind(
        'has_episode',
        lambda d: d.has_episode_edge_ops,
        lambda g, s, t: HasEpisodeEdge(
            group_id=g, source_node_uuid=s, target_node_uuid=t, created_at=T0
        ),
        _seed_has_episode,
        HAS_EPISODE_TABLE.name,
    ),
    Kind(
        'next_episode',
        lambda d: d.next_episode_edge_ops,
        lambda g, s, t: NextEpisodeEdge(
            group_id=g, source_node_uuid=s, target_node_uuid=t, created_at=T0
        ),
        _seed_next_episode,
        NEXT_EPISODE_TABLE.name,
    ),
]


@pytest.fixture(params=SIMPLE_KINDS, ids=lambda kind: kind.name)
def kind(request: pytest.FixtureRequest) -> Kind:
    return request.param


# --- EntityEdge round trip and reads ---


async def test_entity_edge_round_trips_every_field(clean_driver: SpannerDriver):
    group = _group()
    source, target = await _entities(clean_driver, group, 2)
    edge = _edge(
        group,
        source.uuid,
        target.uuid,
        name='WORKS_WITH',
        fact='node0 works with node1',
        fact_embedding=EMBEDDING,
        episodes=['ep-1', 'ep-2'],
        attributes={'weight': 2, 'tags': ['a', 'b']},
        valid_at=T1,
        invalid_at=T2,
        expired_at=T3,
        reference_time=T1,
    )

    await clean_driver.entity_edge_ops.save(clean_driver, edge)
    got = await clean_driver.entity_edge_ops.get_by_uuid(clean_driver, edge.uuid)

    assert got.uuid == edge.uuid
    assert got.group_id == group
    assert got.source_node_uuid == source.uuid
    assert got.target_node_uuid == target.uuid
    assert got.name == 'WORKS_WITH'
    assert got.fact == 'node0 works with node1'
    assert got.fact_embedding == EMBEDDING
    assert got.episodes == ['ep-1', 'ep-2']
    assert got.attributes == {'weight': 2, 'tags': ['a', 'b']}
    assert got.created_at == T0
    assert got.valid_at == T1
    assert got.invalid_at == T2
    assert got.expired_at == T3
    assert got.reference_time == T1


async def test_save_is_an_upsert(clean_driver: SpannerDriver):
    group = _group()
    source, target = await _entities(clean_driver, group, 2)
    edge = _edge(group, source.uuid, target.uuid, fact='before')
    await clean_driver.entity_edge_ops.save(clean_driver, edge)

    edge.fact = 'after'
    await clean_driver.entity_edge_ops.save(clean_driver, edge)

    found = await clean_driver.entity_edge_ops.get_by_group_ids(clean_driver, [group])
    assert [e.fact for e in found] == ['after']


async def test_get_between_nodes_returns_parallel_edges(clean_driver: SpannerDriver):
    group = _group()
    a, b = await _entities(clean_driver, group, 2)
    knows = _edge(group, a.uuid, b.uuid, name='KNOWS')
    likes = _edge(group, a.uuid, b.uuid, name='LIKES')
    for edge in (knows, likes):
        await clean_driver.entity_edge_ops.save(clean_driver, edge)

    ops = clean_driver.entity_edge_ops
    forward = await ops.get_between_nodes(clean_driver, a.uuid, b.uuid)
    backward = await ops.get_between_nodes(clean_driver, b.uuid, a.uuid)

    assert sorted(e.uuid for e in forward) == sorted([knows.uuid, likes.uuid])
    assert backward == []


async def test_get_by_node_uuid_covers_source_target_and_self_loop(clean_driver: SpannerDriver):
    group = _group()
    a, b, c = await _entities(clean_driver, group, 3)
    incoming = _edge(group, a.uuid, b.uuid)
    outgoing = _edge(group, b.uuid, c.uuid)
    loop = _edge(group, b.uuid, b.uuid)
    unrelated = _edge(group, a.uuid, c.uuid)
    ops = clean_driver.entity_edge_ops
    await ops.save_bulk(clean_driver, [incoming, outgoing, loop, unrelated])

    found = await ops.get_by_node_uuid(clean_driver, b.uuid)

    # The self-loop matches both the source and the target select, but is returned once.
    assert sorted(e.uuid for e in found) == sorted([incoming.uuid, outgoing.uuid, loop.uuid])
    assert await ops.get_by_node_uuid(clean_driver, 'no-such-node') == []


# --- the other four kinds ---


async def test_simple_edge_round_trip_and_delete(clean_driver: SpannerDriver, kind: Kind):
    group = _group()
    source, target = await kind.seed(clean_driver, group)
    ops = kind.ops(clean_driver)
    edge = kind.make(group, source, target)

    await ops.save(clean_driver, edge)
    got = await ops.get_by_uuid(clean_driver, edge.uuid)

    assert type(got) is type(edge)
    assert got.uuid == edge.uuid
    assert got.group_id == group
    assert got.source_node_uuid == source
    assert got.target_node_uuid == target
    assert got.created_at == T0

    await ops.delete(clean_driver, edge)

    assert await ops.get_by_uuids(clean_driver, [edge.uuid]) == []
    with pytest.raises(EdgeNotFoundError):
        await ops.get_by_uuid(clean_driver, edge.uuid)


async def test_simple_edge_bulk_group_cursor_and_delete_by_uuids(
    clean_driver: SpannerDriver, kind: Kind
):
    group = _group()
    source, target = await kind.seed(clean_driver, group)
    ops = kind.ops(clean_driver)
    edges = [kind.make(group, source, target) for _ in range(5)]

    if kind.has_bulk:
        await ops.save_bulk(clean_driver, edges, batch_size=2)
    else:
        for edge in edges:
            await ops.save(clean_driver, edge)

    assert len(await ops.get_by_uuids(clean_driver, [e.uuid for e in edges])) == 5
    page = await ops.get_by_group_ids(clean_driver, [group], limit=2)
    rest = await ops.get_by_group_ids(clean_driver, [group], uuid_cursor=page[-1].uuid)
    assert len(page) == 2
    assert len(rest) == 3
    assert {e.uuid for e in page + rest} == {e.uuid for e in edges}

    await ops.delete_by_uuids(clean_driver, [e.uuid for e in edges[:3]] + ['missing'])

    left = await ops.get_by_group_ids(clean_driver, [group])
    assert {e.uuid for e in left} == {e.uuid for e in edges[3:]}


async def test_simple_edge_delete_in_transaction_is_deferred(
    clean_driver: SpannerDriver, kind: Kind
):
    group = _group()
    source, target = await kind.seed(clean_driver, group)
    ops = kind.ops(clean_driver)
    edge = kind.make(group, source, target)
    await ops.save(clean_driver, edge)

    async with clean_driver.transaction() as tx:
        await ops.delete(clean_driver, edge, tx)
        assert len(await ops.get_by_uuids(clean_driver, [edge.uuid])) == 1

    assert await ops.get_by_uuids(clean_driver, [edge.uuid]) == []


async def test_simple_edge_missing_source_raises(clean_driver: SpannerDriver, kind: Kind):
    group = _group()
    _, target = await kind.seed(clean_driver, group)
    ops = kind.ops(clean_driver)
    orphan = kind.make(group, f'ghost-{uuid4()}', target)

    with pytest.raises(SpannerMissingEndpointError) as single:
        await ops.save(clean_driver, orphan)

    assert single.value.edge_uuid == orphan.uuid
    assert single.value.node_uuid == orphan.source_node_uuid
    assert single.value.table == kind.table
    assert single.value.__cause__ is not None
    if kind.has_bulk:
        with pytest.raises(SpannerMissingEndpointError) as bulk:
            await ops.save_bulk(clean_driver, [orphan])
        assert bulk.value.table == kind.table
        assert bulk.value.__cause__ is not None
    assert await ops.get_by_uuids(clean_driver, [orphan.uuid]) == []


# --- bulk saves and chunked reads ---


async def test_save_bulk_in_batches_of_100(clean_driver: SpannerDriver):
    group = _group()
    a, b = await _entities(clean_driver, group, 2)
    edges = _edges(group, a.uuid, b.uuid, 250, fact_embedding=EMBEDDING)
    ops = clean_driver.entity_edge_ops

    await ops.save_bulk(clean_driver, edges, batch_size=100)

    found = await ops.get_by_uuids(clean_driver, [e.uuid for e in edges])
    assert len(found) == 250
    assert {e.uuid for e in found} == {e.uuid for e in edges}


async def test_save_bulk_default_batch_crosses_the_read_chunk(clean_driver: SpannerDriver):
    group = _group()
    a, b = await _entities(clean_driver, group, 2)
    edges = _edges(group, a.uuid, b.uuid, 620)
    ops = clean_driver.entity_edge_ops

    await ops.save_bulk(clean_driver, edges)

    # 620 uuids exceed one 500-uuid read chunk.
    found = await ops.get_by_uuids(clean_driver, [e.uuid for e in edges])
    assert len(found) == 620
    assert len(await ops.get_by_group_ids(clean_driver, [group])) == 620

    await ops.delete_by_uuids(clean_driver, [e.uuid for e in edges])

    assert await ops.get_by_group_ids(clean_driver, [group]) == []


async def test_empty_inputs_are_no_ops(clean_driver: SpannerDriver):
    ops = clean_driver.entity_edge_ops

    await ops.save_bulk(clean_driver, [])
    await ops.delete_by_uuids(clean_driver, [])

    assert await ops.get_by_uuids(clean_driver, []) == []
    assert await ops.get_by_group_ids(clean_driver, []) == []


# --- group reads and pagination ---


async def test_get_by_group_ids_filters_by_group(clean_driver: SpannerDriver):
    group_a, group_b, group_c = _group(), _group(), _group()
    a1, a2 = await _entities(clean_driver, group_a, 2)
    b1, b2 = await _entities(clean_driver, group_b, 2)
    ops = clean_driver.entity_edge_ops
    await ops.save_bulk(
        clean_driver, _edges(group_a, a1.uuid, a2.uuid, 3) + _edges(group_b, b1.uuid, b2.uuid, 2)
    )

    assert len(await ops.get_by_group_ids(clean_driver, [group_a])) == 3
    assert len(await ops.get_by_group_ids(clean_driver, [group_a, group_b])) == 5
    assert await ops.get_by_group_ids(clean_driver, [group_c]) == []


async def test_get_by_group_ids_pages_descending_with_uuid_cursor(clean_driver: SpannerDriver):
    group = _group()
    a, b = await _entities(clean_driver, group, 2)
    edges = _edges(group, a.uuid, b.uuid, 25)
    ops = clean_driver.entity_edge_ops
    await ops.save_bulk(clean_driver, edges)

    first = await ops.get_by_group_ids(clean_driver, [group], limit=10)
    second = await ops.get_by_group_ids(clean_driver, [group], limit=10, uuid_cursor=first[-1].uuid)
    third = await ops.get_by_group_ids(clean_driver, [group], limit=10, uuid_cursor=second[-1].uuid)

    first_ids, second_ids = [e.uuid for e in first], [e.uuid for e in second]
    assert (len(first), len(second), len(third)) == (10, 10, 5)
    assert first_ids == sorted(first_ids, reverse=True)
    assert second_ids == sorted(second_ids, reverse=True)
    assert max(second_ids) < min(first_ids)
    assert {e.uuid for e in first + second + third} == {e.uuid for e in edges}


# --- embeddings ---


async def test_load_embeddings_fills_the_edge(clean_driver: SpannerDriver):
    group = _group()
    a, b = await _entities(clean_driver, group, 2)
    saved = _edge(group, a.uuid, b.uuid, fact_embedding=EMBEDDING)
    ops = clean_driver.entity_edge_ops
    await ops.save(clean_driver, saved)
    probe = _edge(group, a.uuid, b.uuid, uuid=saved.uuid)
    assert probe.fact_embedding is None

    await ops.load_embeddings(clean_driver, probe)

    assert probe.fact_embedding == EMBEDDING


async def test_load_embeddings_raises_for_a_missing_edge(clean_driver: SpannerDriver):
    group = _group()
    a, b = await _entities(clean_driver, group, 2)

    with pytest.raises(EdgeNotFoundError):
        await clean_driver.entity_edge_ops.load_embeddings(
            clean_driver, _edge(group, a.uuid, b.uuid)
        )


async def test_load_embeddings_bulk_skips_missing_edges(clean_driver: SpannerDriver):
    group = _group()
    a, b = await _entities(clean_driver, group, 2)
    ops = clean_driver.entity_edge_ops
    saved = [
        _edge(group, a.uuid, b.uuid, fact_embedding=EMBEDDING if i % 2 else OTHER_EMBEDDING)
        for i in range(7)
    ]
    await ops.save_bulk(clean_driver, saved)
    probes = [_edge(group, a.uuid, b.uuid, uuid=e.uuid) for e in saved]
    missing = _edge(group, a.uuid, b.uuid)

    # batch_size 3 splits the 8 probes over three reads.
    await ops.load_embeddings_bulk(clean_driver, [*probes, missing], batch_size=3)

    assert [p.fact_embedding for p in probes] == [e.fact_embedding for e in saved]
    assert missing.fact_embedding is None


# --- deletes ---


async def test_delete_and_delete_by_uuids(clean_driver: SpannerDriver):
    group = _group()
    a, b = await _entities(clean_driver, group, 2)
    edges = _edges(group, a.uuid, b.uuid, 6)
    ops = clean_driver.entity_edge_ops
    await ops.save_bulk(clean_driver, edges)

    await ops.delete(clean_driver, edges[0])
    await ops.delete_by_uuids(clean_driver, [e.uuid for e in edges[1:3]] + ['missing'])

    left = await ops.get_by_group_ids(clean_driver, [group])
    assert {e.uuid for e in left} == {e.uuid for e in edges[3:]}


async def test_delete_in_a_transaction_applies_on_exit(clean_driver: SpannerDriver):
    group = _group()
    a, b = await _entities(clean_driver, group, 2)
    first, second = _edges(group, a.uuid, b.uuid, 2)
    ops = clean_driver.entity_edge_ops
    await ops.save_bulk(clean_driver, [first, second])
    uuids = [first.uuid, second.uuid]

    async with clean_driver.transaction() as tx:
        await ops.delete(clean_driver, first, tx)
        await ops.delete_by_uuids(clean_driver, [second.uuid], tx)
        assert len(await ops.get_by_uuids(clean_driver, uuids)) == 2

    assert await ops.get_by_uuids(clean_driver, uuids) == []


async def test_node_and_edge_saved_in_one_transaction(clean_driver: SpannerDriver):
    group = _group()
    existing = await _entity(clean_driver, group)
    fresh = EntityNode(name='fresh', group_id=group, created_at=T0)
    edge = _edge(group, fresh.uuid, existing.uuid)
    ops = clean_driver.entity_edge_ops

    async with clean_driver.transaction() as tx:
        await clean_driver.apply_mutations(
            [upsert(ENTITY_TABLE.name, [entity_node_to_row(fresh)], ENTITY_TABLE.writable_columns)],
            tx,
        )
        await ops.save(clean_driver, edge, tx)
        await ops.save_bulk(clean_driver, [], tx)

    got = await ops.get_by_uuid(clean_driver, edge.uuid)
    assert got.source_node_uuid == fresh.uuid


# --- missing rows and endpoints ---


async def test_get_by_uuid_raises_for_a_missing_edge(clean_driver: SpannerDriver):
    with pytest.raises(EdgeNotFoundError):
        await clean_driver.entity_edge_ops.get_by_uuid(clean_driver, 'missing')


async def test_entity_edge_with_missing_source_raises(clean_driver: SpannerDriver):
    group = _group()
    target = await _entity(clean_driver, group)
    ops = clean_driver.entity_edge_ops
    orphan = _edge(group, 'no-such-node', target.uuid)

    with pytest.raises(SpannerMissingEndpointError) as single:
        await ops.save(clean_driver, orphan)
    with pytest.raises(SpannerMissingEndpointError) as bulk:
        await ops.save_bulk(clean_driver, [orphan])

    assert single.value.edge_uuid == orphan.uuid
    assert single.value.node_uuid == 'no-such-node'
    assert single.value.table == RELATES_TO_TABLE.name
    assert single.value.__cause__ is not None
    assert bulk.value.table == RELATES_TO_TABLE.name
    assert bulk.value.__cause__ is not None
    assert await ops.get_by_uuids(clean_driver, [orphan.uuid]) == []


async def test_entity_edge_with_missing_target_is_stored(clean_driver: SpannerDriver):
    # Documented behaviour: only the source is a parent row (interleaving); the target is not
    # checked.
    group = _group()
    source = await _entity(clean_driver, group)
    dangling = _edge(group, source.uuid, 'no-such-node')
    ops = clean_driver.entity_edge_ops

    await ops.save(clean_driver, dangling)

    got = await ops.get_by_uuid(clean_driver, dangling.uuid)
    assert got.target_node_uuid == 'no-such-node'
    by_target = await ops.get_by_node_uuid(clean_driver, 'no-such-node')
    assert [e.uuid for e in by_target] == [dangling.uuid]
