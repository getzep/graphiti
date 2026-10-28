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

# SpannerGraphMaintenanceOperations against the Spanner emulator (or a real instance); see
# conftest.py.
#
#   SPANNER_EMULATOR_HOST=localhost:9010 pytest tests/driver/spanner/test_graph_ops_int.py -q

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import pytest_asyncio

from graphiti_core.driver.spanner import SpannerDriver, mutations
from graphiti_core.driver.spanner.rows import (
    COMMUNITY_EDGE_COLUMNS,
    COMMUNITY_NODE_COLUMNS,
    ENTITY_EDGE_COLUMNS,
    ENTITY_NODE_COLUMNS,
    EPISODIC_EDGE_COLUMNS,
    EPISODIC_NODE_COLUMNS,
    HAS_EPISODE_EDGE_COLUMNS,
    NEXT_EPISODE_EDGE_COLUMNS,
    SAGA_NODE_COLUMNS,
    community_edge_to_row,
    community_node_to_row,
    entity_edge_to_row,
    entity_node_to_row,
    episodic_edge_to_row,
    episodic_node_to_row,
    has_episode_edge_to_row,
    next_episode_edge_to_row,
    saga_node_to_row,
)
from graphiti_core.driver.spanner.schema import TABLES
from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    HasEpisodeEdge,
    NextEpisodeEdge,
)
from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    EpisodeType,
    EpisodicNode,
    SagaNode,
)

pytestmark = pytest.mark.asyncio(loop_scope='session')

EMBEDDING_DIM = 384
NOW = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)
ENTITY_EMBEDDING = [0.25] * EMBEDDING_DIM
COMMUNITY_EMBEDDING = [0.5] * EMBEDDING_DIM


def ent(key: str, group: str) -> EntityNode:
    return EntityNode(
        uuid=f'ent-{key}',
        name=key,
        group_id=group,
        labels=['Entity', 'Person'],
        created_at=NOW,
        summary=f'summary of {key}',
        name_embedding=ENTITY_EMBEDDING,
        attributes={'k': key},
    )


def rel(a: EntityNode, b: EntityNode, group: str | None = None) -> EntityEdge:
    return EntityEdge(
        source_node_uuid=a.uuid,
        target_node_uuid=b.uuid,
        group_id=group or a.group_id,
        name='KNOWS',
        fact=f'{a.name} knows {b.name}',
        created_at=NOW,
    )


def episode(key: str, group: str, offset: int) -> EpisodicNode:
    return EpisodicNode(
        uuid=f'ep-{key}',
        name=key,
        group_id=group,
        source=EpisodeType.message,
        source_description='chat',
        content=f'content {key}',
        valid_at=NOW + timedelta(minutes=offset),
        created_at=NOW + timedelta(minutes=offset),
    )


def community(key: str, group: str) -> CommunityNode:
    return CommunityNode(
        uuid=f'comm-{key}',
        name=key,
        group_id=group,
        labels=['Community'],
        created_at=NOW,
        summary=f'community {key}',
        name_embedding=COMMUNITY_EMBEDDING,
    )


def member(c: CommunityNode, e: EntityNode) -> CommunityEdge:
    return CommunityEdge(
        source_node_uuid=c.uuid, target_node_uuid=e.uuid, group_id=c.group_id, created_at=NOW
    )


def mention(ep: EpisodicNode, e: EntityNode) -> EpisodicEdge:
    return EpisodicEdge(
        source_node_uuid=ep.uuid, target_node_uuid=e.uuid, group_id=ep.group_id, created_at=NOW
    )


# Group A: cluster C1 = K4 (a1..a4, plus a parallel a2->a1 edge), cluster C2 = triangle
# b1..b3 with b4 hanging off b3, isolated entity iso. a1 also has a RELATES_TO edge to a
# group-B entity, which the same-group projection must ignore.
a1, a2, a3, a4 = (ent(k, 'A') for k in ('a1', 'a2', 'a3', 'a4'))
b1, b2, b3, b4 = (ent(k, 'A') for k in ('b1', 'b2', 'b3', 'b4'))
iso = ent('iso', 'A')
# The parallel-edge pair that used to hang label_propagation: p1 and p2 are joined by three
# RELATES_TO edges and nothing else.
p1, p2 = ent('p1', 'A'), ent('p2', 'A')
# Group B: x1-x2 pair (single edge), x3 isolated.
x1, x2, x3 = (ent(k, 'B') for k in ('x1', 'x2', 'x3'))

C1 = frozenset({a1.uuid, a2.uuid, a3.uuid, a4.uuid})
C2 = frozenset({b1.uuid, b2.uuid, b3.uuid, b4.uuid})
ISO = frozenset({iso.uuid})
PAIR = frozenset({p1.uuid, p2.uuid})
XB = frozenset({x1.uuid, x2.uuid})
X3 = frozenset({x3.uuid})
CLUSTERS_A = {C1, C2, ISO, PAIR}
CLUSTERS_B = {XB, X3}

ENTITIES = [a1, a2, a3, a4, b1, b2, b3, b4, iso, p1, p2, x1, x2, x3]
REL_EDGES = [
    rel(a1, a2),
    rel(a1, a3),
    rel(a1, a4),
    rel(a2, a3),
    rel(a2, a4),
    rel(a3, a4),
    rel(a2, a1),
    rel(b1, b2),
    rel(b2, b3),
    rel(b3, b1),
    rel(b3, b4),
    rel(p1, p2),
    rel(p1, p2),
    rel(p2, p1),
    rel(a1, x1, 'A'),  # cross-group
    rel(x1, x2),
]

e1, e2 = episode('e1', 'A', 0), episode('e2', 'A', 1)
eB = episode('eB', 'B', 2)
MENTIONS = [
    mention(e1, a1),
    mention(e1, a2),
    mention(e1, iso),
    mention(e2, a2),
    mention(e2, b1),
    mention(eB, x1),
]

comm1, comm2 = community('comm1', 'A'), community('comm2', 'A')
commB = community('commB', 'B')
MEMBERS = [
    member(comm1, a1),
    member(comm1, a2),
    member(comm1, a3),
    member(comm2, a3),
    member(commB, x1),
]

sagaA = SagaNode(uuid='saga-A', name='sagaA', group_id='A', created_at=NOW)
sagaB = SagaNode(uuid='saga-B', name='sagaB', group_id='B', created_at=NOW)
HAS_EPISODE = [
    HasEpisodeEdge(
        source_node_uuid=sagaA.uuid, target_node_uuid=e.uuid, group_id='A', created_at=NOW
    )
    for e in (e1, e2)
] + [
    HasEpisodeEdge(
        source_node_uuid=sagaB.uuid, target_node_uuid=eB.uuid, group_id='B', created_at=NOW
    )
]
NEXT_EPISODE = [
    NextEpisodeEdge(
        source_node_uuid=e1.uuid, target_node_uuid=e2.uuid, group_id='A', created_at=NOW
    )
]


async def seed(driver: SpannerDriver) -> None:
    await driver.apply_mutations(
        [
            mutations.upsert(
                'Entity', [entity_node_to_row(n) for n in ENTITIES], ENTITY_NODE_COLUMNS
            ),
            mutations.upsert(
                'Episodic', [episodic_node_to_row(n) for n in (e1, e2, eB)], EPISODIC_NODE_COLUMNS
            ),
            mutations.upsert(
                'Community',
                [community_node_to_row(n) for n in (comm1, comm2, commB)],
                COMMUNITY_NODE_COLUMNS,
            ),
            mutations.upsert(
                'Saga', [saga_node_to_row(n) for n in (sagaA, sagaB)], SAGA_NODE_COLUMNS
            ),
            mutations.upsert(
                'RelatesTo', [entity_edge_to_row(e) for e in REL_EDGES], ENTITY_EDGE_COLUMNS
            ),
            mutations.upsert(
                'Mentions', [episodic_edge_to_row(e) for e in MENTIONS], EPISODIC_EDGE_COLUMNS
            ),
            mutations.upsert(
                'HasMember', [community_edge_to_row(e) for e in MEMBERS], COMMUNITY_EDGE_COLUMNS
            ),
            mutations.upsert(
                'HasEpisode',
                [has_episode_edge_to_row(e) for e in HAS_EPISODE],
                HAS_EPISODE_EDGE_COLUMNS,
            ),
            mutations.upsert(
                'NextEpisode',
                [next_episode_edge_to_row(e) for e in NEXT_EPISODE],
                NEXT_EPISODE_EDGE_COLUMNS,
            ),
        ]
    )


async def counts(driver: SpannerDriver, group: str | None = None) -> dict[str, int]:
    out: dict[str, int] = {}
    for table in TABLES:
        if group is None:
            rows = await driver.read(f'SELECT COUNT(*) AS n FROM {table}')
        else:
            rows = await driver.read(
                f'SELECT COUNT(*) AS n FROM {table} WHERE group_id = @g', {'g': group}
            )
        out[table] = rows[0]['n']
    return out


async def determine(driver: SpannerDriver, entity: EntityNode) -> tuple[Any, bool]:
    # The base class annotates determine_entity_community as returning None; it returns a
    # (community, is_new) tuple.
    ops: Any = driver.graph_ops
    return await ops.determine_entity_community(driver, entity)


def cluster_sets(clusters: list[list[EntityNode]]) -> set[frozenset[str]]:
    return {frozenset(n.uuid for n in cluster) for cluster in clusters}


@pytest_asyncio.fixture(loop_scope='session')
async def seeded(clean_driver: SpannerDriver) -> SpannerDriver:
    await seed(clean_driver)
    return clean_driver


async def test_seed_populates_every_table(seeded: SpannerDriver):
    totals = await counts(seeded)
    assert all(n > 0 for n in totals.values()), totals


# --- get_community_clusters -----------------------------------------------------------------


async def test_clusters_single_group(seeded: SpannerDriver):
    clusters = await seeded.graph_ops.get_community_clusters(seeded, ['A'])

    assert cluster_sets(clusters) == CLUSTERS_A
    assert sum(len(c) for c in clusters) == len(C1 | C2 | ISO | PAIR)  # no duplicates


async def test_clusters_decode_entity_nodes(seeded: SpannerDriver):
    clusters = await seeded.graph_ops.get_community_clusters(seeded, ['A'])

    nodes = [n for c in clusters for n in c]
    assert all(isinstance(n, EntityNode) and n.group_id == 'A' for n in nodes)
    for n in nodes:
        assert n.attributes == {'k': n.name}
        assert 'Person' in n.labels
        assert n.summary == f'summary of {n.name}'
        assert n.name_embedding is not None
        assert n.name_embedding == pytest.approx(ENTITY_EMBEDDING)


async def test_clusters_parallel_edges_form_one_cluster(seeded: SpannerDriver):
    # Used to hang label_propagation; the call must return and the pair must stay together.
    clusters = await seeded.graph_ops.get_community_clusters(seeded, ['A'])

    assert PAIR in cluster_sets(clusters)


async def test_clusters_all_groups_ignore_cross_group_edge(seeded: SpannerDriver):
    clusters = await seeded.graph_ops.get_community_clusters(seeded, None)

    # a1 -> x1 crosses groups; the same-group projection must not merge A and B clusters.
    assert cluster_sets(clusters) == CLUSTERS_A | CLUSTERS_B


async def test_clusters_other_group(seeded: SpannerDriver):
    clusters = await seeded.graph_ops.get_community_clusters(seeded, ['B'])

    assert cluster_sets(clusters) == CLUSTERS_B


async def test_clusters_empty_group_list(seeded: SpannerDriver):
    assert await seeded.graph_ops.get_community_clusters(seeded, []) == []


async def test_clusters_empty_graph(clean_driver: SpannerDriver):
    assert await clean_driver.graph_ops.get_community_clusters(clean_driver, None) == []


# --- determine_entity_community -------------------------------------------------------------


async def test_determine_community_existing_member(seeded: SpannerDriver):
    found, is_new = await determine(seeded, a1)

    assert found is not None
    assert found.uuid == comm1.uuid
    assert is_new is False
    assert found.summary == 'community comm1'
    assert found.name_embedding == pytest.approx(COMMUNITY_EMBEDDING)


async def test_determine_community_non_member_uses_neighbours(seeded: SpannerDriver):
    # a4 is in no community, but its neighbours a1, a2, a3 are mostly in comm1.
    found, is_new = await determine(seeded, a4)

    assert found is not None
    assert found.uuid == comm1.uuid
    assert is_new is True


async def test_determine_community_isolated_entity(seeded: SpannerDriver):
    assert await determine(seeded, iso) == (None, False)


async def test_determine_community_neighbours_without_community(seeded: SpannerDriver):
    assert await determine(seeded, b1) == (None, False)


# --- get_mentioned_nodes / get_communities_by_nodes -----------------------------------------


async def test_mentioned_nodes_one_episode(seeded: SpannerDriver):
    nodes = await seeded.graph_ops.get_mentioned_nodes(seeded, [e1])

    assert sorted(n.uuid for n in nodes) == sorted([a1.uuid, a2.uuid, iso.uuid])


async def test_mentioned_nodes_two_episodes_deduplicated(seeded: SpannerDriver):
    nodes = await seeded.graph_ops.get_mentioned_nodes(seeded, [e1, e2])

    # a2 is mentioned by both episodes but returned once.
    assert sorted(n.uuid for n in nodes) == sorted([a1.uuid, a2.uuid, iso.uuid, b1.uuid])


async def test_mentioned_nodes_empty_input(seeded: SpannerDriver):
    assert await seeded.graph_ops.get_mentioned_nodes(seeded, []) == []


async def test_communities_by_nodes_single(seeded: SpannerDriver):
    found = await seeded.graph_ops.get_communities_by_nodes(seeded, [a1])

    assert [c.uuid for c in found] == [comm1.uuid]


async def test_communities_by_nodes_deduplicated(seeded: SpannerDriver):
    found = await seeded.graph_ops.get_communities_by_nodes(seeded, [a1, a3, a2])

    assert sorted(c.uuid for c in found) == sorted([comm1.uuid, comm2.uuid])


async def test_communities_by_nodes_no_community(seeded: SpannerDriver):
    assert await seeded.graph_ops.get_communities_by_nodes(seeded, [b1]) == []


async def test_communities_by_nodes_empty_input(seeded: SpannerDriver):
    assert await seeded.graph_ops.get_communities_by_nodes(seeded, []) == []


# --- remove_communities ---------------------------------------------------------------------


async def test_remove_communities_one_group(seeded: SpannerDriver):
    before_a, before_b = await counts(seeded, 'A'), await counts(seeded, 'B')

    await seeded.graph_ops.remove_communities(seeded, ['A'])

    after_a, after_b = await counts(seeded, 'A'), await counts(seeded, 'B')
    assert after_a['Community'] == 0
    assert after_a['HasMember'] == 0
    assert after_a['Entity'] == before_a['Entity']
    for table in TABLES:
        if table not in ('Community', 'HasMember'):
            assert after_a[table] == before_a[table], table
    assert after_b == before_b
    assert await determine(seeded, a4) == (None, False)


async def test_remove_communities_all_groups(seeded: SpannerDriver):
    before = await counts(seeded)

    await seeded.graph_ops.remove_communities(seeded, None)

    after = await counts(seeded)
    assert after['Community'] == 0
    assert after['HasMember'] == 0
    for table in TABLES:
        if table not in ('Community', 'HasMember'):
            assert after[table] == before[table], table


async def test_remove_communities_empty_list_removes_all(seeded: SpannerDriver):
    # As in the base class, an empty group list means every group.
    await seeded.graph_ops.remove_communities(seeded, [])

    after = await counts(seeded)
    assert after['Community'] == 0
    assert after['HasMember'] == 0
    assert after['Entity'] == len(ENTITIES)


# --- clear_data -----------------------------------------------------------------------------


async def test_clear_data_one_group(seeded: SpannerDriver):
    before_b = await counts(seeded, 'B')

    await seeded.graph_ops.clear_data(seeded, ['A'])

    after_a, after_b = await counts(seeded, 'A'), await counts(seeded, 'B')
    assert all(n == 0 for n in after_a.values()), after_a
    assert after_b == before_b
    assert await counts(seeded) == after_b
    clusters = await seeded.graph_ops.get_community_clusters(seeded, None)
    assert cluster_sets(clusters) == CLUSTERS_B


async def test_clear_data_all_groups(seeded: SpannerDriver):
    await seeded.graph_ops.clear_data(seeded, None)

    totals = await counts(seeded)
    assert all(n == 0 for n in totals.values()), totals


async def test_clear_data_empty_list_deletes_nothing(seeded: SpannerDriver):
    before = await counts(seeded)

    await seeded.graph_ops.clear_data(seeded, [])

    assert await counts(seeded) == before
