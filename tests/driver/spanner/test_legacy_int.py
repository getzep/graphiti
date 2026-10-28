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

# The SQL-backed methods of the legacy adapters (graphiti_core/driver/spanner/legacy.py)
# against the Spanner emulator (or a real instance); see conftest.py.
#
#   SPANNER_EMULATOR_HOST=localhost:9010 pytest tests/driver/spanner/test_legacy_int.py -q

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from graphiti_core.driver.spanner import SpannerDriver, mutations, rows
from graphiti_core.driver.spanner.legacy import (
    SpannerLegacyGraphOperations,
    SpannerLegacySearchInterface,
)
from graphiti_core.edges import CommunityEdge, EntityEdge, EpisodicEdge, HasEpisodeEdge
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode, SagaNode

pytestmark = pytest.mark.asyncio(loop_scope='session')

DIM = 384
UTC = timezone.utc
T0 = datetime(2026, 1, 1, tzinfo=UTC)
H = timedelta(hours=1)


def emb(index: int) -> list[float]:
    """A 384-float one-hot vector."""
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


def entity(
    uuid: str,
    name: str,
    group_id: str,
    labels: list[str],
    attributes: dict,
    created_at: datetime,
    embedding: list[float] | None = None,
) -> EntityNode:
    return EntityNode(
        uuid=uuid,
        name=name,
        group_id=group_id,
        labels=labels,
        attributes=attributes,
        created_at=created_at,
        name_embedding=embedding,
    )


def episode(uuid: str, content: str, valid_at: datetime, created_at: datetime) -> EpisodicNode:
    return EpisodicNode(
        uuid=uuid,
        name=uuid,
        group_id='g1',
        source=EpisodeType.message,
        source_description='chat',
        content=content,
        valid_at=valid_at,
        created_at=created_at,
    )


def relates(uuid: str, source: str, target: str, embedding: list[float] | None = None):
    return EntityEdge(
        uuid=uuid,
        group_id='g1',
        source_node_uuid=source,
        target_node_uuid=target,
        created_at=T0,
        name='KNOWS',
        fact=f'{source} knows {target}',
        fact_embedding=embedding,
    )


def link(cls: type, uuid: str, source: str, target: str):
    return cls(
        uuid=uuid, group_id='g1', source_node_uuid=source, target_node_uuid=target, created_at=T0
    )


N1 = entity(
    'n1', 'Alice', 'g1', ['Entity', 'Person'], {'sku': ' A-1 ', 'age': 30, 'vip': True}, T0, emb(0)
)
N2 = entity('n2', 'Bob', 'g1', ['Entity'], {'sku': 'B-2', 'age': 30.0}, T0 + H, emb(1))
N3 = entity('n3', 'Alice', 'g2', ['Entity', 'Person'], {'sku': 'A-1'}, T0 + 2 * H)
N4 = entity('n4', 'Carol', 'g1', ['Entity'], {'age': '30', 'vip': False}, T0 + 3 * H)
E1 = relates('e1', 'n1', 'n2', emb(2))
E2 = relates('e2', 'n4', 'n1')
C1 = CommunityNode(uuid='c1', name='C1', group_id='g1', created_at=T0, name_embedding=emb(3))
C2 = CommunityNode(uuid='c2', name='C2', group_id='g1', created_at=T0)
SAGA = SagaNode(uuid='s1', name='S', group_id='g1', created_at=T0)
EP1 = episode('ep1', 'one', T0, T0)
EP2 = episode('ep2', 'two (backfilled)', T0 - 24 * H, T0 + H)  # older valid_at, later ingestion
EP3 = episode('ep3', 'three', T0 + 2 * H, T0 + 2 * H)
EP4 = episode('ep4', '', T0 + 3 * H, T0 + 3 * H)  # empty content is skipped by contents


@pytest_asyncio.fixture(loop_scope='session')
async def seeded(clean_driver: SpannerDriver) -> SpannerDriver:
    members = [
        link(CommunityEdge, 'hm1', 'c1', 'n2'),
        link(CommunityEdge, 'hm2', 'c2', 'n2'),
        link(CommunityEdge, 'hm3', 'c2', 'n4'),
    ]
    has_episode = [link(HasEpisodeEdge, f'he{i}', 's1', f'ep{i}') for i in range(1, 5)]
    mentions = [
        link(EpisodicEdge, 'm1', 'ep1', 'n1'),
        link(EpisodicEdge, 'm2', 'ep2', 'n1'),
        link(EpisodicEdge, 'm3', 'ep3', 'n2'),
    ]
    await clean_driver.apply_mutations(
        [
            mutations.upsert(
                'Entity',
                [rows.entity_node_to_row(n) for n in (N1, N2, N3, N4)],
                rows.ENTITY_NODE_COLUMNS,
            ),
            mutations.upsert(
                'Episodic',
                [rows.episodic_node_to_row(e) for e in (EP1, EP2, EP3, EP4)],
                rows.EPISODIC_NODE_COLUMNS,
            ),
            mutations.upsert(
                'Community',
                [rows.community_node_to_row(c) for c in (C1, C2)],
                rows.COMMUNITY_NODE_COLUMNS,
            ),
            mutations.upsert('Saga', [rows.saga_node_to_row(SAGA)], rows.SAGA_NODE_COLUMNS),
            mutations.upsert(
                'RelatesTo',
                [rows.entity_edge_to_row(e) for e in (E1, E2)],
                rows.ENTITY_EDGE_COLUMNS,
            ),
            mutations.upsert(
                'HasMember',
                [rows.community_edge_to_row(e) for e in members],
                rows.COMMUNITY_EDGE_COLUMNS,
            ),
            mutations.upsert(
                'HasEpisode',
                [rows.has_episode_edge_to_row(e) for e in has_episode],
                rows.HAS_EPISODE_EDGE_COLUMNS,
            ),
            mutations.upsert(
                'Mentions',
                [rows.episodic_edge_to_row(e) for e in mentions],
                rows.EPISODIC_EDGE_COLUMNS,
            ),
        ]
    )
    return clean_driver


# --- node_get_by_names -----------------------------------------------------------------------


async def test_node_get_by_names_filters_group_and_names(seeded: SpannerDriver) -> None:
    g = SpannerLegacyGraphOperations()
    found = await g.node_get_by_names(EntityNode, seeded, ['Alice', 'Bob', 'Nobody'], ['g1'])
    assert sorted(n.uuid for n in found) == ['n1', 'n2']
    alice = next(n for n in found if n.uuid == 'n1')
    assert alice.attributes == {'sku': ' A-1 ', 'age': 30, 'vip': True}
    assert alice.name_embedding == emb(0)
    assert alice.labels == ['Entity', 'Person']


async def test_node_get_by_names_spans_several_groups(seeded: SpannerDriver) -> None:
    g = SpannerLegacyGraphOperations()
    found = await g.node_get_by_names(EntityNode, seeded, ['Alice'], ['g1', 'g2'])
    assert sorted(n.uuid for n in found) == ['n1', 'n3']
    assert await g.node_get_by_names(EntityNode, seeded, ['Alice'], ['g3']) == []


# --- node_get_by_attribute_values ------------------------------------------------------------


async def by_attr(
    driver: SpannerDriver,
    attributes: dict,
    labels: list[str] | None = None,
    group_id: str = 'g1',
    limit: int = 100,
) -> list[str]:
    nodes = await SpannerLegacyGraphOperations().node_get_by_attribute_values(
        EntityNode, driver, group_id, attributes, labels, limit
    )
    return [n.uuid for n in nodes]


@pytest.mark.parametrize(
    'attributes, labels, group_id, expected',
    [
        pytest.param({'sku': 'A-1'}, ['Person'], 'g1', ['n1'], id='string-trimmed-with-label'),
        pytest.param({'sku': 'A-1'}, None, 'g2', ['n3'], id='string-other-group'),
        pytest.param({'age': 30}, None, 'g1', ['n1', 'n2', 'n4'], id='int-matches-float-and-text'),
        pytest.param({'age': 30.0}, ['Person'], 'g1', ['n1'], id='float-with-label'),
        pytest.param({'vip': True}, None, 'g1', ['n1'], id='bool-true'),
        pytest.param({'vip': False}, None, 'g1', ['n4'], id='bool-false'),
        pytest.param({'sku': 'B-2', 'age': 30}, None, 'g1', ['n2'], id='two-keys-anded'),
        pytest.param({'sku': 'zzz'}, None, 'g1', [], id='no-match'),
        pytest.param({'vip': None}, None, 'g1', ['n2'], id='none-means-missing'),
    ],
)
async def test_node_get_by_attribute_values(
    seeded: SpannerDriver,
    attributes: dict,
    labels: list[str] | None,
    group_id: str,
    expected: list[str],
) -> None:
    assert await by_attr(seeded, attributes, labels, group_id) == expected


async def test_node_get_by_attribute_values_limit_and_empty(seeded: SpannerDriver) -> None:
    assert await by_attr(seeded, {'age': 30}, limit=1) == ['n1']
    assert await by_attr(seeded, {}) == []


async def test_node_get_by_attribute_values_rejects_bad_keys_and_values(
    seeded: SpannerDriver,
) -> None:
    with pytest.raises(ValueError):
        await by_attr(seeded, {"x') OR TRUE --": 1})
    with pytest.raises(ValueError):
        await by_attr(seeded, {'has space': 1})
    with pytest.raises(TypeError):
        await by_attr(seeded, {'k': [1]})


# --- saga helpers ----------------------------------------------------------------------------


async def test_saga_get_previous_episode_uuid(seeded: SpannerDriver) -> None:
    g = SpannerLegacyGraphOperations()
    # Latest valid_at among the other episodes, excluding the current one.
    assert await g.saga_get_previous_episode_uuid(seeded, 's1', 'ep4') == 'ep3'
    assert await g.saga_get_previous_episode_uuid(seeded, 's1', 'ep3') == 'ep4'
    assert await g.saga_get_previous_episode_uuid(seeded, 'nope', 'ep1') is None


async def test_saga_get_episode_contents_orders_by_created_at(seeded: SpannerDriver) -> None:
    g = SpannerLegacyGraphOperations()
    contents = await g.saga_get_episode_contents(seeded, 's1')
    # Ingestion order (created_at), not valid_at order; the empty episode is skipped.
    assert contents == [
        ('one', T0, T0),
        ('two (backfilled)', T0 - 24 * H, T0 + H),
        ('three', T0 + 2 * H, T0 + 2 * H),
    ]


async def test_saga_get_episode_contents_since_watermark_and_limit(seeded: SpannerDriver) -> None:
    g = SpannerLegacyGraphOperations()
    since = await g.saga_get_episode_contents(seeded, 's1', since=T0)
    assert [c[0] for c in since] == ['two (backfilled)', 'three']
    limited = await g.saga_get_episode_contents(seeded, 's1', since=T0, limit=1)
    assert [c[0] for c in limited] == ['two (backfilled)']
    assert [c[0] for c in await g.saga_get_episode_contents(seeded, 's1', limit=2)] == [
        'one',
        'two (backfilled)',
    ]
    assert await g.saga_get_episode_contents(seeded, 'nope') == []


# --- embeddings ------------------------------------------------------------------------------


async def test_load_embeddings_bulk_returns_uuid_to_embedding(seeded: SpannerDriver) -> None:
    g = SpannerLegacyGraphOperations()
    nodes = await g.node_load_embeddings_bulk(seeded, [N1, N2, N3], batch_size=2)
    assert nodes == {'n1': emb(0), 'n2': emb(1)}  # n3 has no embedding
    edges = await g.edge_load_embeddings_bulk(seeded, [E1, E2])
    assert edges == {'e1': emb(2)}
    assert await g.node_load_embeddings_bulk(seeded, []) == {}


async def test_get_embeddings_for_communities(seeded: SpannerDriver) -> None:
    s = SpannerLegacySearchInterface()
    assert await s.get_embeddings_for_communities(seeded, [C1, C2]) == {'c1': emb(3)}


# --- determine_entity_community --------------------------------------------------------------


async def test_determine_entity_community_tuple_shape(seeded: SpannerDriver) -> None:
    g = SpannerLegacyGraphOperations()
    # n2 is a member of c1 and c2: it is already in a community, so is_new is False.
    community, is_new = await g.determine_entity_community(seeded, N2)
    assert isinstance(community, CommunityNode)
    assert community.uuid in {'c1', 'c2'} and is_new is False
    # n1 is in none, but its neighbours (n2, n4) are: the most common one wins, flagged new.
    community, is_new = await g.determine_entity_community(seeded, N1)
    assert community is not None and community.uuid == 'c2' and is_new is True
    # n3 has no edges and no membership.
    assert await g.determine_entity_community(seeded, N3) == (None, False)


# --- rerankers -------------------------------------------------------------------------------


async def test_node_distance_reranker(seeded: SpannerDriver) -> None:
    s = SpannerLegacySearchInterface()
    uuids, scores = await s.node_distance_reranker(seeded, ['n3', 'n2', 'n1', 'n4'], 'n1')
    # Centre first with score 10, direct neighbours 1, everything else 0.
    assert uuids == ['n1', 'n2', 'n4', 'n3']
    assert scores == [10.0, 1.0, 1.0, 0.0]


async def test_node_distance_reranker_min_score(seeded: SpannerDriver) -> None:
    s = SpannerLegacySearchInterface()
    uuids, scores = await s.node_distance_reranker(
        seeded, ['n3', 'n2', 'n1', 'n4'], 'n1', min_score=0.5
    )
    assert uuids == ['n1', 'n2', 'n4'] and scores == [10.0, 1.0, 1.0]


async def test_episode_mentions_reranker_orders_by_mention_count(seeded: SpannerDriver) -> None:
    s = SpannerLegacySearchInterface()
    # n1 is mentioned twice, n2 once, n4 never.
    uuids, scores = await s.episode_mentions_reranker(seeded, [['n2', 'n1', 'n4']], min_score=1)
    assert uuids == ['n1', 'n2'] and scores == [2, 1]


async def test_episode_mentions_reranker_scores_unmentioned_zero(seeded: SpannerDriver) -> None:
    s = SpannerLegacySearchInterface()
    uuids, scores = await s.episode_mentions_reranker(seeded, [['n4', 'n2', 'n1']])
    assert uuids == ['n1', 'n2', 'n4'] and scores == [2, 1, 0]


async def test_episode_mentions_reranker_fuses_result_lists_with_rrf(
    seeded: SpannerDriver,
) -> None:
    s = SpannerLegacySearchInterface()
    # n3 and n4 are never mentioned; n4 ranks higher than n3 in the fused list, so the
    # stable sort keeps it ahead of n3 at score 0.
    uuids, scores = await s.episode_mentions_reranker(
        seeded, [['n3', 'n4', 'n2'], ['n4', 'n2', 'n1']]
    )
    assert uuids == ['n1', 'n2', 'n4', 'n3'] and scores == [2, 1, 0, 0]
    assert await s.episode_mentions_reranker(seeded, []) == ([], [])
