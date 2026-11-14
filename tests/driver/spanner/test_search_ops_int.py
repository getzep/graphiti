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

# SpannerSearchOperations against the Spanner emulator (or a real instance); see conftest.py.
#
#   SPANNER_EMULATOR_HOST=localhost:9010 pytest tests/driver/spanner/test_search_ops_int.py -q
#
# One deterministic dataset is seeded once per module (the tests only read it):
#
#   groups gA / gB / gC; entity chain a1 -> a2 -> a3 -> a4 -> b1 -> b2 (RELATES_TO) with a
#   parallel edge a1 -> a2 (r1p) and a two-node cycle k1 <-> k2 in gC; MENTIONS from episodes
#   ep1/ep2/ep3/epB; communities c1/c2 (gA) and c3 (gB) with HAS_MEMBER edges. Embeddings are
#   padded one-hot-like vectors, so cosine ordering is predictable. Similarity scores use
#   Neo4j's (1 + cos) / 2 scale, so orthogonal vectors score 0.5.

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timezone
from typing import Any

import pytest
import pytest_asyncio

from graphiti_core.driver.spanner import SpannerDriver
from graphiti_core.driver.spanner import rows as R
from graphiti_core.driver.spanner.errors import SpannerDriverError
from graphiti_core.driver.spanner.filters import SpannerSearchFilter
from graphiti_core.driver.spanner.mutations import upsert
from graphiti_core.driver.spanner.operations.search_ops import SpannerSearchOperations
from graphiti_core.edges import CommunityEdge, EntityEdge, EpisodicEdge
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters

pytestmark = pytest.mark.asyncio(loop_scope='session')

DIM = 384
GA, GB = 'gA', 'gB'
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
OP = ComparisonOperator
NO_FILTER = SearchFilters()


def ts(year: int, month: int = 1) -> datetime:
    return datetime(year, month, 1, tzinfo=timezone.utc)


def df(op: ComparisonOperator, date: datetime | None = None) -> DateFilter:
    return DateFilter(date=date, comparison_operator=op)


def emb(*leading: float) -> list[float]:
    """A DIM-long vector starting with the given components and zero elsewhere."""
    return [*leading, *([0.0] * (DIM - len(leading)))]


V = emb(1.0)
ZERO = emb()


def ids(items: Sequence[Any]) -> list[str]:
    return [item.uuid for item in items]


def idset(items: Sequence[Any]) -> set[str]:
    return set(ids(items))


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------


def _entity(
    uid: str, name: str, group: str, labels: list[str], summary: str, vec: list[float] | None
) -> EntityNode:
    return EntityNode(
        uuid=uid,
        name=name,
        group_id=group,
        labels=labels,
        summary=summary,
        created_at=T0,
        attributes={'seed': uid, 'tags': ['x', 'y']},
        name_embedding=vec,
    )


ENTITIES = [
    _entity('a1', 'Alice', GA, ['Person'], 'Alice is an engineer', emb(1.0)),
    _entity('a2', 'Bob', GA, ['Person'], 'Bob is a designer', emb(0.9, 0.1)),
    _entity('a3', 'Acme', GA, ['Company'], 'Acme is a company', emb(0.0, 1.0)),
    _entity('a4', 'Carol', GA, ['Person'], 'Carol is a manager', emb(0.0, 0.0, 1.0)),
    _entity('a5', 'Dave', GA, ['Person'], 'Dave has no embedding', None),
    _entity('b1', 'Alice', GB, ['Person'], 'Alice from group B', emb(1.0)),
    _entity('b2', 'Zed', GB, ['Company'], 'Zed is a company', emb(0.0, 1.0)),
    # two-node cycle in its own group
    _entity('k1', 'Kilo', 'gC', ['Thing'], 'Kilo node', None),
    _entity('k2', 'Lima', 'gC', ['Thing'], 'Lima node', None),
]


def _episode(uid: str, group: str, content: str) -> EpisodicNode:
    return EpisodicNode(
        uuid=uid,
        name=uid,
        group_id=group,
        source=EpisodeType.text,
        source_description='search ops test',
        content=content,
        valid_at=T0,
        created_at=T0,
        entity_edges=[],
    )


EPISODES = [
    _episode('ep1', GA, 'Alice met Bob at Acme'),
    _episode('ep2', GA, 'Bob joined Acme'),
    _episode('ep3', GA, 'Carol likes tea'),
    _episode('epB', GB, 'Alice in group B'),
]

MENTIONS = [
    ('ep1', 'a1', GA),
    ('ep1', 'a2', GA),
    ('ep1', 'a3', GA),
    ('ep2', 'a2', GA),
    ('ep2', 'a3', GA),
    ('ep3', 'a4', GA),
    ('epB', 'b1', GB),
]


def _edge(
    uid: str,
    src: str,
    tgt: str,
    group: str,
    name: str,
    fact: str,
    vec: list[float] | None,
    valid: datetime | None = None,
    invalid: datetime | None = None,
    expired: datetime | None = None,
) -> EntityEdge:
    return EntityEdge(
        uuid=uid,
        source_node_uuid=src,
        target_node_uuid=tgt,
        group_id=group,
        name=name,
        fact=fact,
        fact_embedding=vec,
        episodes=['ep1'],
        created_at=T0,
        valid_at=valid,
        invalid_at=invalid,
        expired_at=expired,
        attributes={'edge': uid},
    )


EDGES = [
    _edge('r1', 'a1', 'a2', GA, 'KNOWS', 'Alice knows Bob', emb(1.0), ts(2020)),
    # parallel to r1, undated
    _edge('r1p', 'a1', 'a2', GA, 'WORKS_WITH', 'Alice works with Bob', emb(0.8, 0.2)),
    _edge(
        'r2',
        'a2',
        'a3',
        GA,
        'WORKS_AT',
        'Bob works at Acme',
        emb(0.0, 1.0),
        ts(2021),
        ts(2023),
        ts(2023),
    ),
    _edge('r3', 'a3', 'a4', GA, 'EMPLOYS', 'Acme employs Carol', emb(0.0, 0.0, 1.0), ts(2022)),
    # cross-group hop: edge in gA, target node in gB
    _edge('rx', 'a4', 'b1', GA, 'KNOWS', 'Carol knows Alice', emb(0.0, 0.0, 0.0, 1.0)),
    _edge('rb', 'b1', 'b2', GB, 'WORKS_AT', 'Alice works at Zed', emb(1.0), ts(2020)),
    _edge('rk1', 'k1', 'k2', 'gC', 'LINKS', 'Kilo links Lima', None),
    _edge('rk2', 'k2', 'k1', 'gC', 'LINKS', 'Lima links Kilo', None),
]

COMMUNITIES = [
    CommunityNode(
        uuid='c1',
        name='Acme people',
        group_id=GA,
        summary='Engineers at Acme',
        created_at=T0,
        name_embedding=emb(1.0),
    ),
    CommunityNode(
        uuid='c2',
        name='Tea lovers',
        group_id=GA,
        summary='People who like tea',
        created_at=T0,
        name_embedding=emb(0.0, 0.0, 1.0),
    ),
    CommunityNode(
        uuid='c3',
        name='Zed folks',
        group_id=GB,
        summary='People at Zed',
        created_at=T0,
        name_embedding=emb(1.0),
    ),
]
MEMBERS = [('c1', 'a1', GA), ('c1', 'a2', GA), ('c2', 'a4', GA), ('c3', 'b2', GB)]


async def seed(driver: SpannerDriver) -> None:
    mentions = [
        EpisodicEdge(
            uuid=f'm_{s}_{t}', source_node_uuid=s, target_node_uuid=t, group_id=g, created_at=T0
        )
        for s, t, g in MENTIONS
    ]
    members = [
        CommunityEdge(
            uuid=f'hm_{s}_{t}', source_node_uuid=s, target_node_uuid=t, group_id=g, created_at=T0
        )
        for s, t, g in MEMBERS
    ]
    await driver.apply_mutations(
        [
            upsert('Entity', [R.entity_node_to_row(n) for n in ENTITIES], R.ENTITY_NODE_COLUMNS),
            upsert(
                'Episodic', [R.episodic_node_to_row(n) for n in EPISODES], R.EPISODIC_NODE_COLUMNS
            ),
            upsert(
                'Community',
                [R.community_node_to_row(n) for n in COMMUNITIES],
                R.COMMUNITY_NODE_COLUMNS,
            ),
            upsert('RelatesTo', [R.entity_edge_to_row(e) for e in EDGES], R.ENTITY_EDGE_COLUMNS),
            upsert(
                'Mentions', [R.episodic_edge_to_row(e) for e in mentions], R.EPISODIC_EDGE_COLUMNS
            ),
            upsert(
                'HasMember',
                [R.community_edge_to_row(e) for e in members],
                R.COMMUNITY_EDGE_COLUMNS,
            ),
        ]
    )


@pytest_asyncio.fixture(scope='module', loop_scope='session')
async def driver(spanner_driver: SpannerDriver) -> AsyncIterator[SpannerDriver]:
    """The shared driver with the dataset above seeded once for this module."""
    await spanner_driver.clear_tables()
    await seed(spanner_driver)
    yield spanner_driver


@pytest.fixture(scope='module')
def ops() -> SpannerSearchOperations:
    return SpannerSearchOperations()


# ---------------------------------------------------------------------------
# build_fulltext_query and filter builders (delegation)
# ---------------------------------------------------------------------------


async def test_build_fulltext_query_sanitises_and_or_joins(ops: SpannerSearchOperations):
    assert ops.build_fulltext_query('Alice & Bob!') == 'alice OR bob'
    assert ops.build_fulltext_query('!!! &&') == ''


async def test_build_node_search_filters_delegates(ops: SpannerSearchOperations):
    node_filter = ops.build_node_search_filters(SearchFilters(node_labels=['Person', 'Company']))
    assert isinstance(node_filter, SpannerSearchFilter)
    assert node_filter.sql == ('ARRAY_INCLUDES_ANY(n.labels, @nf_labels)',)
    assert node_filter.gql == node_filter.sql
    assert node_filter.params == {'nf_labels': ['Person', 'Company']}
    assert ops.build_node_search_filters(NO_FILTER).sql == ()


async def test_build_edge_search_filters_delegates(ops: SpannerSearchOperations):
    edge_filter = ops.build_edge_search_filters(
        SearchFilters(
            edge_uuids=['r1'],
            edge_source_node_uuids=['a1'],
            node_labels=['Person'],
            valid_at=[
                [df(OP.greater_than, ts(2020)), df(OP.less_than, ts(2022))],
                [df(OP.is_null)],
            ],
        )
    )
    assert edge_filter.needs_endpoints
    assert 'e.edge_uuid IN UNNEST(@ef_edge_uuids)' in edge_filter.sql
    assert 'e.uuid IN UNNEST(@ef_edge_uuids)' in edge_filter.gql
    assert 'e.uuid IN UNNEST(@ef_source_node_uuids)' in edge_filter.sql
    assert 'e.source_node_uuid IN UNNEST(@ef_source_node_uuids)' in edge_filter.gql
    assert set(edge_filter.params) == set(edge_filter.param_types)


# ---------------------------------------------------------------------------
# Node full-text
# ---------------------------------------------------------------------------


async def test_node_fulltext_group_isolation(driver: SpannerDriver, ops: SpannerSearchOperations):
    assert idset(await ops.node_fulltext_search(driver, 'alice', NO_FILTER, None)) == {'a1', 'b1'}
    assert ids(await ops.node_fulltext_search(driver, 'alice', NO_FILTER, [GA])) == ['a1']
    assert await ops.node_fulltext_search(driver, 'alice', NO_FILTER, []) == []


async def test_node_fulltext_matches_name_and_summary(
    driver: SpannerDriver, ops: SpannerSearchOperations
):
    assert ids(await ops.node_fulltext_search(driver, 'Acme', NO_FILTER, [GA])) == ['a3']
    assert ids(await ops.node_fulltext_search(driver, 'engineer', NO_FILTER, [GA])) == ['a1']


async def test_node_fulltext_label_filter(driver: SpannerDriver, ops: SpannerSearchOperations):
    query = 'alice bob acme'
    company = await ops.node_fulltext_search(
        driver, query, SearchFilters(node_labels=['Company']), [GA]
    )
    person = await ops.node_fulltext_search(
        driver, query, SearchFilters(node_labels=['Person']), [GA]
    )
    assert ids(company) == ['a3']
    assert idset(person) == {'a1', 'a2'}


async def test_node_fulltext_miss_empty_query_and_limit(
    driver: SpannerDriver, ops: SpannerSearchOperations
):
    assert await ops.node_fulltext_search(driver, 'nonexistentterm', NO_FILTER, None) == []
    assert await ops.node_fulltext_search(driver, '!!!', NO_FILTER, None) == []
    limited = await ops.node_fulltext_search(driver, 'alice bob acme carol', NO_FILTER, [GA], 2)
    assert len(limited) == 2


async def test_node_fulltext_decodes_node(driver: SpannerDriver, ops: SpannerSearchOperations):
    (node,) = await ops.node_fulltext_search(driver, 'alice', NO_FILTER, [GA])
    assert node.labels == ['Person', 'Entity']
    assert node.attributes == {'seed': 'a1', 'tags': ['x', 'y']}
    assert node.created_at == T0
    assert node.summary == 'Alice is an engineer'
    assert node.name_embedding is None


# ---------------------------------------------------------------------------
# Node similarity
# ---------------------------------------------------------------------------


async def test_node_similarity_orders_by_score(driver: SpannerDriver, ops: SpannerSearchOperations):
    # Default min_score keeps cos >= 0.6 on the (1 + cos) / 2 scale: a1 and b1 (1.0), a2 (~0.99).
    everything = await ops.node_similarity_search(driver, V, NO_FILTER, None)
    assert idset(everything) == {'a1', 'b1', 'a2'}
    assert ids(everything)[2] == 'a2'
    assert ids(await ops.node_similarity_search(driver, V, NO_FILTER, [GA])) == ['a1', 'a2']


async def test_node_similarity_min_score_cutoff(
    driver: SpannerDriver, ops: SpannerSearchOperations
):
    assert ids(await ops.node_similarity_search(driver, V, NO_FILTER, [GA], min_score=0.999)) == [
        'a1'
    ]
    # Orthogonal vectors score 0.5; a node without an embedding is never returned.
    low = ids(await ops.node_similarity_search(driver, V, NO_FILTER, [GA], min_score=0.4))
    assert low[:2] == ['a1', 'a2']
    assert set(low) == {'a1', 'a2', 'a3', 'a4'}


async def test_node_similarity_label_filter_limit_and_int_vector(
    driver: SpannerDriver, ops: SpannerSearchOperations
):
    company = await ops.node_similarity_search(
        driver, V, SearchFilters(node_labels=['Company']), [GA], min_score=0.4
    )
    assert ids(company) == ['a3']
    assert ids(await ops.node_similarity_search(driver, V, NO_FILTER, [GA], limit=1)) == ['a1']
    int_vector: Any = [0, 1, *([0] * (DIM - 2))]  # ints are accepted at runtime
    assert idset(await ops.node_similarity_search(driver, int_vector, NO_FILTER, None)) == {
        'a3',
        'b2',
    }


async def test_node_similarity_zero_vector_returns_nothing(
    driver: SpannerDriver, ops: SpannerSearchOperations
):
    assert await ops.node_similarity_search(driver, ZERO, NO_FILTER, None) == []


async def test_node_similarity_wrong_dimension_raises(
    driver: SpannerDriver, ops: SpannerSearchOperations
):
    with pytest.raises(SpannerDriverError):
        await ops.node_similarity_search(driver, [1.0, 0.0], NO_FILTER, None)


# ---------------------------------------------------------------------------
# Node BFS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('depth', 'expected'),
    [(1, {'a2'}), (2, {'a2', 'a3'}), (3, {'a2', 'a3', 'a4'}), (4, {'a2', 'a3', 'a4'})],
)
async def test_node_bfs_entity_origin_depth(
    driver: SpannerDriver, ops: SpannerSearchOperations, depth: int, expected: set[str]
):
    # b1 is never reached: the hop a4 -> b1 leaves group gA.
    found = await ops.node_bfs_search(driver, ['a1'], NO_FILTER, depth, None, 50)
    assert idset(found) == expected
    assert len(found) == len(expected)  # distinct


@pytest.mark.parametrize(
    ('origin', 'depth', 'expected'),
    [
        ('ep1', 1, {'a1', 'a2', 'a3'}),
        ('ep1', 2, {'a1', 'a2', 'a3', 'a4'}),
        ('ep3', 2, {'a4'}),
    ],
)
async def test_node_bfs_episodic_origin(
    driver: SpannerDriver,
    ops: SpannerSearchOperations,
    origin: str,
    depth: int,
    expected: set[str],
):
    found = await ops.node_bfs_search(driver, [origin], NO_FILTER, depth, None, 50)
    assert idset(found) == expected
    assert len(found) == len(expected)


async def test_node_bfs_group_label_limit_and_degenerate_inputs(
    driver: SpannerDriver, ops: SpannerSearchOperations
):
    two_origins = await ops.node_bfs_search(driver, ['a1', 'ep3'], NO_FILTER, 1, [GA], 50)
    assert idset(two_origins) == {'a2', 'a4'}
    assert await ops.node_bfs_search(driver, ['a1'], NO_FILTER, 3, [GB], 50) == []
    company = await ops.node_bfs_search(
        driver, ['a1'], SearchFilters(node_labels=['Company']), 3, None, 50
    )
    assert ids(company) == ['a3']
    assert await ops.node_bfs_search(driver, ['a1'], NO_FILTER, 0, None, 50) == []
    assert await ops.node_bfs_search(driver, [], NO_FILTER, 2, None, 50) == []
    assert len(await ops.node_bfs_search(driver, ['a1'], NO_FILTER, 3, None, 2)) == 2


@pytest.mark.parametrize(('depth', 'expected'), [(1, {'k2'}), (2, {'k1', 'k2'}), (3, {'k1', 'k2'})])
async def test_node_bfs_cycle(
    driver: SpannerDriver, ops: SpannerSearchOperations, depth: int, expected: set[str]
):
    # In a two-node cycle the origin is reachable again at depth 2.
    found = await ops.node_bfs_search(driver, ['k1'], NO_FILTER, depth, None, 50)
    assert idset(found) == expected
    assert len(found) == len(expected)


async def test_node_bfs_decodes_node(driver: SpannerDriver, ops: SpannerSearchOperations):
    (node,) = await ops.node_bfs_search(driver, ['a1'], NO_FILTER, 1, None, 50)
    assert node.name == 'Bob'
    assert node.labels == ['Person', 'Entity']
    assert node.attributes == {'seed': 'a2', 'tags': ['x', 'y']}


# ---------------------------------------------------------------------------
# Edge BFS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('depth', 'expected'),
    [
        (1, {'r1', 'r1p'}),
        (2, {'r1', 'r1p', 'r2'}),
        (3, {'r1', 'r1p', 'r2', 'r3'}),
        (4, {'r1', 'r1p', 'r2', 'r3', 'rx'}),
        (5, {'r1', 'r1p', 'r2', 'r3', 'rx', 'rb'}),
    ],
)
async def test_edge_bfs_entity_origin_depth(
    driver: SpannerDriver, ops: SpannerSearchOperations, depth: int, expected: set[str]
):
    found = await ops.edge_bfs_search(driver, ['a1'], depth, NO_FILTER, None, 50)
    assert idset(found) == expected
    assert len(found) == len(expected)  # parallel edge returned once


@pytest.mark.parametrize(
    ('depth', 'expected'), [(1, {'rk1'}), (2, {'rk1', 'rk2'}), (3, {'rk1', 'rk2'})]
)
async def test_edge_bfs_cycle(
    driver: SpannerDriver, ops: SpannerSearchOperations, depth: int, expected: set[str]
):
    found = await ops.edge_bfs_search(driver, ['k1'], depth, NO_FILTER, None, 50)
    assert idset(found) == expected
    assert len(found) == len(expected)


async def test_edge_bfs_group_episodic_origin_and_limit(
    driver: SpannerDriver, ops: SpannerSearchOperations
):
    in_group = await ops.edge_bfs_search(driver, ['a1'], 5, NO_FILTER, [GA], 50)
    assert idset(in_group) == {'r1', 'r1p', 'r2', 'r3', 'rx'}
    # An episode's first hop is its MENTIONS edge, which is not a RELATES_TO edge.
    assert await ops.edge_bfs_search(driver, ['ep1'], 1, NO_FILTER, None, 50) == []
    deeper = await ops.edge_bfs_search(driver, ['ep1'], 2, NO_FILTER, None, 50)
    assert idset(deeper) == {'r1', 'r1p', 'r2', 'r3'}
    assert len(await ops.edge_bfs_search(driver, ['a1'], 3, NO_FILTER, None, 2)) == 2
    assert await ops.edge_bfs_search(driver, ['a1'], 0, NO_FILTER, None, 50) == []


async def test_edge_bfs_edge_types(driver: SpannerDriver, ops: SpannerSearchOperations):
    knows = SearchFilters(edge_types=['KNOWS'])
    assert ids(await ops.edge_bfs_search(driver, ['a1'], 1, knows, None, 50)) == ['r1']
    assert idset(await ops.edge_bfs_search(driver, ['a1'], 4, knows, None, 50)) == {'r1', 'rx'}


async def test_edge_bfs_valid_at_filters(driver: SpannerDriver, ops: SpannerSearchOperations):
    valid_after = SearchFilters(valid_at=[[df(OP.greater_than, ts(2020, 6))]])
    assert ids(await ops.edge_bfs_search(driver, ['a1'], 2, valid_after, None, 50)) == ['r2']
    deeper = await ops.edge_bfs_search(driver, ['a1'], 3, valid_after, None, 50)
    assert idset(deeper) == {'r2', 'r3'}

    undated = SearchFilters(valid_at=[[df(OP.is_null)]])
    assert idset(await ops.edge_bfs_search(driver, ['a1'], 4, undated, None, 50)) == {'r1p', 'rx'}

    either = SearchFilters(
        valid_at=[
            [df(OP.greater_than_equal, ts(2020)), df(OP.less_than, ts(2021, 6))],
            [df(OP.is_null)],
        ]
    )
    assert idset(await ops.edge_bfs_search(driver, ['a1'], 4, either, None, 50)) == {
        'r1',
        'r2',
        'r1p',
        'rx',
    }

    not_2020 = SearchFilters(valid_at=[[df(OP.not_equals, ts(2020))]])
    # NULL never satisfies <>, so the undated r1p is dropped too.
    assert idset(await ops.edge_bfs_search(driver, ['a1'], 3, not_2020, None, 50)) == {'r2', 'r3'}


async def test_edge_bfs_other_date_fields(driver: SpannerDriver, ops: SpannerSearchOperations):
    invalidated = SearchFilters(invalid_at=[[df(OP.is_not_null)]])
    assert ids(await ops.edge_bfs_search(driver, ['a1'], 4, invalidated, None, 50)) == ['r2']
    expired = SearchFilters(expired_at=[[df(OP.greater_than, ts(2022))]])
    assert ids(await ops.edge_bfs_search(driver, ['a1'], 4, expired, None, 50)) == ['r2']
    created = SearchFilters(created_at=[[df(OP.equals, T0)]])
    assert idset(await ops.edge_bfs_search(driver, ['a1'], 3, created, None, 50)) == {
        'r1',
        'r1p',
        'r2',
        'r3',
    }


async def test_edge_bfs_label_and_endpoint_filters(
    driver: SpannerDriver, ops: SpannerSearchOperations
):
    labelled = SearchFilters(node_labels=['Person'])  # both endpoints must be Person
    assert idset(await ops.edge_bfs_search(driver, ['a1'], 4, labelled, None, 50)) == {
        'r1',
        'r1p',
        'rx',
    }
    by_uuid = SearchFilters(edge_uuids=['r2', 'r3', 'rb'])
    assert idset(await ops.edge_bfs_search(driver, ['a1'], 3, by_uuid, None, 50)) == {'r2', 'r3'}
    source = SearchFilters(edge_source_node_uuids=['a2'])
    assert ids(await ops.edge_bfs_search(driver, ['a1'], 3, source, None, 50)) == ['r2']
    target = SearchFilters(edge_target_node_uuids=['a4'])
    assert ids(await ops.edge_bfs_search(driver, ['a1'], 3, target, None, 50)) == ['r3']


async def test_edge_bfs_decodes_edge(driver: SpannerDriver, ops: SpannerSearchOperations):
    found = await ops.edge_bfs_search(driver, ['a1'], 2, NO_FILTER, None, 50)
    r2 = next(edge for edge in found if edge.uuid == 'r2')
    assert (r2.source_node_uuid, r2.target_node_uuid) == ('a2', 'a3')
    assert (r2.valid_at, r2.invalid_at, r2.expired_at) == (ts(2021), ts(2023), ts(2023))
    assert r2.episodes == ['ep1']
    assert r2.attributes == {'edge': 'r2'}
    assert r2.fact_embedding is None


# ---------------------------------------------------------------------------
# Edge full-text
# ---------------------------------------------------------------------------


async def test_edge_fulltext_group_isolation(driver: SpannerDriver, ops: SpannerSearchOperations):
    everywhere = await ops.edge_fulltext_search(driver, 'alice', NO_FILTER, None, 50)
    assert idset(everywhere) == {'r1', 'r1p', 'rx', 'rb'}
    in_group = await ops.edge_fulltext_search(driver, 'alice', NO_FILTER, [GA], 50)
    assert idset(in_group) == {'r1', 'r1p', 'rx'}


async def test_edge_fulltext_hits_misses_and_limit(
    driver: SpannerDriver, ops: SpannerSearchOperations
):
    assert idset(await ops.edge_fulltext_search(driver, 'acme', NO_FILTER, [GA], 50)) == {
        'r2',
        'r3',
    }
    assert ids(await ops.edge_fulltext_search(driver, 'employs', NO_FILTER, None, 50)) == ['r3']
    assert await ops.edge_fulltext_search(driver, 'zzzz', NO_FILTER, None, 50) == []
    assert await ops.edge_fulltext_search(driver, '!!!', NO_FILTER, None, 50) == []
    assert len(await ops.edge_fulltext_search(driver, 'alice', NO_FILTER, None, 2)) == 2


async def test_edge_fulltext_filters(driver: SpannerDriver, ops: SpannerSearchOperations):
    typed = SearchFilters(edge_types=['WORKS_AT'])
    assert idset(await ops.edge_fulltext_search(driver, 'works', typed, None, 50)) == {'r2', 'rb'}

    people = SearchFilters(node_labels=['Person'])
    assert idset(await ops.edge_fulltext_search(driver, 'alice', people, [GA], 50)) == {
        'r1',
        'r1p',
        'rx',
    }
    # r2 and r3 each join a Person to a Company, so a Company-only label filter drops them.
    companies = SearchFilters(node_labels=['Company'])
    assert await ops.edge_fulltext_search(driver, 'acme', companies, [GA], 50) == []

    source = SearchFilters(edge_source_node_uuids=['a1'])
    assert idset(await ops.edge_fulltext_search(driver, 'alice', source, None, 50)) == {
        'r1',
        'r1p',
    }
    target = SearchFilters(edge_target_node_uuids=['a2'])
    assert idset(await ops.edge_fulltext_search(driver, 'bob', target, None, 50)) == {'r1', 'r1p'}

    undated = SearchFilters(valid_at=[[df(OP.is_null)]])
    assert idset(await ops.edge_fulltext_search(driver, 'alice', undated, [GA], 50)) == {
        'r1p',
        'rx',
    }


# ---------------------------------------------------------------------------
# Edge similarity
# ---------------------------------------------------------------------------


async def test_edge_similarity_orders_by_score(driver: SpannerDriver, ops: SpannerSearchOperations):
    found = await ops.edge_similarity_search(driver, V, None, None, NO_FILTER, None, 50)
    assert idset(found) == {'r1', 'rb', 'r1p'}
    assert ids(found)[2] == 'r1p'
    in_group = await ops.edge_similarity_search(driver, V, None, None, NO_FILTER, [GA], 50)
    assert ids(in_group) == ['r1', 'r1p']


async def test_edge_similarity_source_and_target_filters_without_group(
    driver: SpannerDriver, ops: SpannerSearchOperations
):
    by_source = await ops.edge_similarity_search(driver, V, 'a1', None, NO_FILTER, None, 50)
    assert ids(by_source) == ['r1', 'r1p']
    both = await ops.edge_similarity_search(driver, V, 'a1', 'a2', NO_FILTER, None, 50)
    assert ids(both) == ['r1', 'r1p']
    by_target = await ops.edge_similarity_search(driver, V, None, 'b2', NO_FILTER, None, 50)
    assert ids(by_target) == ['rb']
    assert await ops.edge_similarity_search(driver, V, 'a2', 'a1', NO_FILTER, None, 50) == []


async def test_edge_similarity_filters_cutoff_and_limit(
    driver: SpannerDriver, ops: SpannerSearchOperations
):
    typed = SearchFilters(edge_types=['WORKS_WITH'])
    assert ids(await ops.edge_similarity_search(driver, V, None, None, typed, None, 50)) == ['r1p']
    strict = await ops.edge_similarity_search(
        driver, V, None, None, NO_FILTER, None, 50, min_score=0.99
    )
    assert idset(strict) == {'r1', 'rb'}
    labelled = await ops.edge_similarity_search(
        driver, V, None, None, SearchFilters(node_labels=['Person']), [GA], 50, min_score=0.4
    )
    assert ids(labelled)[:2] == ['r1', 'r1p']
    assert idset(labelled) == {'r1', 'r1p', 'rx'}
    limited = await ops.edge_similarity_search(driver, V, None, None, NO_FILTER, None, 1)
    assert len(limited) == 1
    assert limited[0].uuid in {'r1', 'rb'}


async def test_edge_similarity_zero_vector_and_wrong_dimension(
    driver: SpannerDriver, ops: SpannerSearchOperations
):
    assert await ops.edge_similarity_search(driver, ZERO, None, None, NO_FILTER, None, 50) == []
    with pytest.raises(SpannerDriverError):
        await ops.edge_similarity_search(driver, [1.0], None, None, NO_FILTER, None, 50)


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


async def test_episode_fulltext_search(driver: SpannerDriver, ops: SpannerSearchOperations):
    assert idset(await ops.episode_fulltext_search(driver, 'alice', NO_FILTER, None)) == {
        'ep1',
        'epB',
    }
    assert ids(await ops.episode_fulltext_search(driver, 'alice', NO_FILTER, [GA])) == ['ep1']
    assert idset(await ops.episode_fulltext_search(driver, 'acme', NO_FILTER, [GA])) == {
        'ep1',
        'ep2',
    }
    assert await ops.episode_fulltext_search(driver, 'zzzz', NO_FILTER, None) == []
    assert await ops.episode_fulltext_search(driver, '!!!', NO_FILTER, None) == []


async def test_episode_fulltext_decodes_episode(
    driver: SpannerDriver, ops: SpannerSearchOperations
):
    (episode,) = await ops.episode_fulltext_search(driver, 'tea', NO_FILTER, [GA])
    assert episode.content == 'Carol likes tea'
    assert episode.source == EpisodeType.text
    assert episode.valid_at == T0


# ---------------------------------------------------------------------------
# Communities
# ---------------------------------------------------------------------------


async def test_community_fulltext_search(driver: SpannerDriver, ops: SpannerSearchOperations):
    assert ids(await ops.community_fulltext_search(driver, 'acme', None)) == ['c1']
    assert ids(await ops.community_fulltext_search(driver, 'tea', [GA])) == ['c2']
    assert await ops.community_fulltext_search(driver, 'tea', [GB]) == []
    assert ids(await ops.community_fulltext_search(driver, 'zed', None)) == ['c3']
    assert await ops.community_fulltext_search(driver, 'zzzz', None) == []
    assert await ops.community_fulltext_search(driver, '!!!', None) == []


async def test_community_similarity_search(driver: SpannerDriver, ops: SpannerSearchOperations):
    assert idset(await ops.community_similarity_search(driver, V, None)) == {'c1', 'c3'}
    (community,) = await ops.community_similarity_search(driver, V, [GA])
    assert community.uuid == 'c1'
    assert community.name_embedding == V
    low = await ops.community_similarity_search(driver, V, [GA], min_score=0.4)
    assert ids(low) == ['c1', 'c2']
    assert await ops.community_similarity_search(driver, ZERO, None) == []
    with pytest.raises(SpannerDriverError):
        await ops.community_similarity_search(driver, [1.0], None)


# ---------------------------------------------------------------------------
# Rerankers
# ---------------------------------------------------------------------------


async def test_node_distance_reranker(driver: SpannerDriver, ops: SpannerSearchOperations):
    candidates = ['a3', 'a1', 'a4', 'a2']
    # The centre first, then its direct neighbours in input order, then everything else.
    assert ids(await ops.node_distance_reranker(driver, candidates, 'a2')) == [
        'a2',
        'a3',
        'a1',
        'a4',
    ]
    # min_score drops non-neighbours (a4 is two hops from a2).
    assert ids(await ops.node_distance_reranker(driver, candidates, 'a2', 0.5)) == [
        'a2',
        'a3',
        'a1',
    ]
    assert ids(await ops.node_distance_reranker(driver, ['a4', 'a1'], 'a2')) == ['a1', 'a4']
    # Only the centre has score 1 / 0.1 = 10, which passes min_score 5.
    assert ids(await ops.node_distance_reranker(driver, ['a1', 'a2'], 'a2', 5)) == ['a2']
    assert await ops.node_distance_reranker(driver, [], 'a2') == []


async def test_episode_mentions_reranker(driver: SpannerDriver, ops: SpannerSearchOperations):
    candidates = ['a1', 'a2', 'a5', 'a4']
    # Ascending mention count as in Neo4j; entities never mentioned come last.
    assert ids(await ops.episode_mentions_reranker(driver, candidates)) == [
        'a1',
        'a4',
        'a2',
        'a5',
    ]
    assert ids(await ops.episode_mentions_reranker(driver, candidates, 2)) == ['a2', 'a5']
    assert await ops.episode_mentions_reranker(driver, []) == []
