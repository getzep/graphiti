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

# Hermetic tests for the legacy GraphOperationsInterface / SearchInterface adapters
# (graphiti_core/driver/spanner/legacy.py). The driver is a MagicMock whose ops objects are
# mocked, so nothing here needs a Spanner instance.

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface
from graphiti_core.driver.operations.community_edge_ops import CommunityEdgeOperations
from graphiti_core.driver.operations.community_node_ops import CommunityNodeOperations
from graphiti_core.driver.operations.entity_edge_ops import EntityEdgeOperations
from graphiti_core.driver.operations.entity_node_ops import EntityNodeOperations
from graphiti_core.driver.operations.episode_node_ops import EpisodeNodeOperations
from graphiti_core.driver.operations.episodic_edge_ops import EpisodicEdgeOperations
from graphiti_core.driver.operations.graph_ops import GraphMaintenanceOperations
from graphiti_core.driver.operations.has_episode_edge_ops import HasEpisodeEdgeOperations
from graphiti_core.driver.operations.next_episode_edge_ops import NextEpisodeEdgeOperations
from graphiti_core.driver.operations.saga_node_ops import SagaNodeOperations
from graphiti_core.driver.operations.search_ops import SearchOperations
from graphiti_core.driver.search_interface.search_interface import SearchInterface
from graphiti_core.driver.spanner import legacy
from graphiti_core.driver.spanner.driver import SpannerDriver
from graphiti_core.driver.spanner.legacy import (
    SpannerLegacyGraphOperations,
    SpannerLegacySearchInterface,
)
from graphiti_core.driver.spanner.rows import community_node_to_row, entity_node_to_row
from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    HasEpisodeEdge,
    NextEpisodeEdge,
)
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode, SagaNode
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk_tx

pytestmark = pytest.mark.asyncio

UTC = timezone.utc
T0 = datetime(2026, 1, 1, tzinfo=UTC)

OPS_ABCS: dict[str, type] = {
    'entity_node_ops': EntityNodeOperations,
    'episode_node_ops': EpisodeNodeOperations,
    'community_node_ops': CommunityNodeOperations,
    'saga_node_ops': SagaNodeOperations,
    'entity_edge_ops': EntityEdgeOperations,
    'episodic_edge_ops': EpisodicEdgeOperations,
    'community_edge_ops': CommunityEdgeOperations,
    'has_episode_edge_ops': HasEpisodeEdgeOperations,
    'next_episode_edge_ops': NextEpisodeEdgeOperations,
    'search_ops': SearchOperations,
    'graph_ops': GraphMaintenanceOperations,
}

# Placeholder in parametrised argument tuples, replaced by the fake driver.
D: Any = object()


def make_driver(read_results: list[list[dict[str, Any]]] | None = None) -> Any:
    d = MagicMock(spec=SpannerDriver)
    for name, abc in OPS_ABCS.items():
        ops = MagicMock(spec=abc)
        for attr, value in inspect.getmembers(abc, inspect.isfunction):
            if inspect.iscoroutinefunction(value):
                setattr(ops, attr, AsyncMock())
        setattr(d, name, ops)
    d.provider = GraphProvider.SPANNER
    d.read = AsyncMock(side_effect=list(read_results) if read_results is not None else None)
    if read_results is None:
        d.read.return_value = []
    return d


def ops_calls(d: Any) -> dict[str, list[str]]:
    """Every call made on the ops mocks, as {ops_attr: [method, ...]}."""
    seen: dict[str, list[str]] = {}
    for name in OPS_ABCS:
        for call in getattr(d, name).mock_calls:
            seen.setdefault(name, []).append(call[0])
    return seen


def fill(d: Any, items: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(d if x is D else x for x in items)


def entity_node(uuid: str = 'n1', **kw: Any) -> EntityNode:
    return EntityNode(
        uuid=uuid,
        name=kw.pop('name', 'Alice'),
        group_id=kw.pop('group_id', 'g1'),
        labels=kw.pop('labels', ['Entity', 'Person']),
        summary='sum',
        name_embedding=[0.1, 0.2, 0.3, 0.4],
        created_at=T0,
        attributes=kw.pop('attributes', {'sku': 'A-1', 'age': 30, 'born': T0}),
    )


def episode(uuid: str = 'ep1', valid_at: datetime = T0) -> EpisodicNode:
    return EpisodicNode(
        uuid=uuid,
        name='ep',
        group_id='g1',
        source=EpisodeType.message,
        source_description='chat',
        content='user: hi',
        valid_at=valid_at,
        created_at=T0,
        entity_edges=['e1'],
        episode_metadata={'k': 'v'},
    )


def entity_edge(uuid: str = 'e1') -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        group_id='g1',
        source_node_uuid='n1',
        target_node_uuid='n2',
        created_at=T0,
        name='KNOWS',
        fact='Alice knows Bob',
        fact_embedding=[0.5, 0.5, 0.5, 0.5],
        episodes=['ep1'],
        valid_at=T0,
        reference_time=T0,
        attributes={'since': '2020', 'weight': 2},
    )


def simple_edge(cls: type, uuid: str) -> Any:
    return cls(uuid=uuid, group_id='g1', source_node_uuid='s', target_node_uuid='t', created_at=T0)


COMMUNITY = CommunityNode(uuid='c1', name='C', group_id='g1', created_at=T0, summary='cs')
SAGA = SagaNode(uuid='s1', name='S', group_id='g1', created_at=T0)
N = entity_node()
EP = episode()
ED = entity_edge()
EPE = simple_edge(EpisodicEdge, 'm1')
CE = simple_edge(CommunityEdge, 'hm1')
HE = simple_edge(HasEpisodeEdge, 'he1')
NE = simple_edge(NextEpisodeEdge, 'ne1')
FILTERS = SearchFilters()


def interface_methods(cls: type) -> list[str]:
    return [name for name, value in cls.__dict__.items() if inspect.isfunction(value)]


def signature_shape(fn: Any) -> list[tuple[str, Any, Any]]:
    return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]


# --- structure -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    'adapter, base, count',
    [
        (SpannerLegacyGraphOperations, GraphOperationsInterface, 85),
        (SpannerLegacySearchInterface, SearchInterface, 14),
    ],
)
async def test_adapter_overrides_every_interface_method(
    adapter: type, base: type, count: int
) -> None:
    names = interface_methods(base)
    assert len(names) == count
    for name in names:
        mine = adapter.__dict__.get(name)
        assert mine is not None, f'{adapter.__name__}.{name} falls through to the base class'
        assert signature_shape(mine) == signature_shape(getattr(base, name)), name
        assert inspect.iscoroutinefunction(mine) == inspect.iscoroutinefunction(getattr(base, name))


async def test_adapters_are_stateless() -> None:
    assert issubclass(SpannerLegacyGraphOperations, GraphOperationsInterface)
    assert issubclass(SpannerLegacySearchInterface, SearchInterface)
    assert type(SpannerLegacyGraphOperations()).model_fields == {}


# --- plain delegation ------------------------------------------------------------------------

# (interface method, call args, call kwargs, ops attr, ops method, expected args, expected kwargs)
DELEGATIONS: list[tuple[str, tuple, dict, str, str, tuple, dict]] = [
    ('node_save', (N, D), {}, 'entity_node_ops', 'save', (D, N), {}),
    ('node_get_by_uuid', (None, D, 'u'), {}, 'entity_node_ops', 'get_by_uuid', (D, 'u'), {}),
    (
        'node_get_by_group_ids',
        (None, D, ['g1'], 5, 'cur'),
        {},
        'entity_node_ops',
        'get_by_group_ids',
        (D, ['g1'], 5, 'cur'),
        {},
    ),
    ('node_load_embeddings', (N, D), {}, 'entity_node_ops', 'load_embeddings', (D, N), {}),
    ('episodic_node_save', (EP, D), {}, 'episode_node_ops', 'save', (D, EP), {}),
    ('episodic_node_delete', (EP, D), {}, 'episode_node_ops', 'delete', (D, EP), {}),
    (
        'episodic_node_delete_by_group_id',
        (None, D, 'g1', 7),
        {},
        'episode_node_ops',
        'delete_by_group_id',
        (D, 'g1'),
        {'batch_size': 7},
    ),
    (
        'episodic_node_get_by_uuid',
        (None, D, 'u'),
        {},
        'episode_node_ops',
        'get_by_uuid',
        (D, 'u'),
        {},
    ),
    (
        'episodic_node_get_by_uuids',
        (None, D, ['u']),
        {},
        'episode_node_ops',
        'get_by_uuids',
        (D, ['u']),
        {},
    ),
    (
        'episodic_node_get_by_group_ids',
        (None, D, ['g1'], None, None),
        {},
        'episode_node_ops',
        'get_by_group_ids',
        (D, ['g1'], None, None),
        {},
    ),
    (
        'episodic_node_get_by_entity_node_uuid',
        (None, D, 'n1'),
        {},
        'episode_node_ops',
        'get_by_entity_node_uuid',
        (D, 'n1'),
        {},
    ),
    ('community_node_save', (COMMUNITY, D), {}, 'community_node_ops', 'save', (D, COMMUNITY), {}),
    (
        'community_node_delete',
        (COMMUNITY, D),
        {},
        'community_node_ops',
        'delete',
        (D, COMMUNITY),
        {},
    ),
    (
        'community_node_delete_by_group_id',
        (None, D, 'g1'),
        {},
        'community_node_ops',
        'delete_by_group_id',
        (D, 'g1'),
        {'batch_size': 100},
    ),
    (
        'community_node_get_by_uuid',
        (None, D, 'c1'),
        {},
        'community_node_ops',
        'get_by_uuid',
        (D, 'c1'),
        {},
    ),
    (
        'community_node_get_by_uuids',
        (None, D, ['c1']),
        {},
        'community_node_ops',
        'get_by_uuids',
        (D, ['c1']),
        {},
    ),
    (
        'community_node_get_by_group_ids',
        (None, D, ['g1'], 3, 'z'),
        {},
        'community_node_ops',
        'get_by_group_ids',
        (D, ['g1'], 3, 'z'),
        {},
    ),
    (
        'community_node_load_name_embedding',
        (COMMUNITY, D),
        {},
        'community_node_ops',
        'load_name_embedding',
        (D, COMMUNITY),
        {},
    ),
    ('saga_node_save', (SAGA, D), {}, 'saga_node_ops', 'save', (D, SAGA), {}),
    ('saga_node_delete', (SAGA, D), {}, 'saga_node_ops', 'delete', (D, SAGA), {}),
    (
        'saga_node_delete_by_group_id',
        (None, D, 'g1', 9),
        {},
        'saga_node_ops',
        'delete_by_group_id',
        (D, 'g1'),
        {'batch_size': 9},
    ),
    ('saga_node_get_by_uuid', (None, D, 's1'), {}, 'saga_node_ops', 'get_by_uuid', (D, 's1'), {}),
    (
        'saga_node_get_by_uuids',
        (None, D, ['s1']),
        {},
        'saga_node_ops',
        'get_by_uuids',
        (D, ['s1']),
        {},
    ),
    (
        'saga_node_get_by_group_ids',
        (None, D, ['g1'], 1, None),
        {},
        'saga_node_ops',
        'get_by_group_ids',
        (D, ['g1'], 1, None),
        {},
    ),
    ('edge_save', (ED, D), {}, 'entity_edge_ops', 'save', (D, ED), {}),
    ('edge_get_by_uuid', (None, D, 'e1'), {}, 'entity_edge_ops', 'get_by_uuid', (D, 'e1'), {}),
    (
        'edge_get_by_uuids',
        (None, D, ['e1']),
        {},
        'entity_edge_ops',
        'get_by_uuids',
        (D, ['e1']),
        {},
    ),
    (
        'edge_get_by_group_ids',
        (None, D, ['g1'], 2, 'c'),
        {},
        'entity_edge_ops',
        'get_by_group_ids',
        (D, ['g1'], 2, 'c'),
        {},
    ),
    ('edge_load_embeddings', (ED, D), {}, 'entity_edge_ops', 'load_embeddings', (D, ED), {}),
    (
        'edge_get_between_nodes',
        (None, D, 'n1', 'n2'),
        {},
        'entity_edge_ops',
        'get_between_nodes',
        (D, 'n1', 'n2'),
        {},
    ),
    (
        'edge_get_by_node_uuid',
        (None, D, 'n1'),
        {},
        'entity_edge_ops',
        'get_by_node_uuid',
        (D, 'n1'),
        {},
    ),
    ('episodic_edge_save', (EPE, D), {}, 'episodic_edge_ops', 'save', (D, EPE), {}),
    ('episodic_edge_delete', (EPE, D), {}, 'episodic_edge_ops', 'delete', (D, EPE), {}),
    (
        'episodic_edge_get_by_uuid',
        (None, D, 'm1'),
        {},
        'episodic_edge_ops',
        'get_by_uuid',
        (D, 'm1'),
        {},
    ),
    (
        'episodic_edge_get_by_group_ids',
        (None, D, ['g1'], None, None),
        {},
        'episodic_edge_ops',
        'get_by_group_ids',
        (D, ['g1'], None, None),
        {},
    ),
    ('community_edge_save', (CE, D), {}, 'community_edge_ops', 'save', (D, CE), {}),
    ('community_edge_delete', (CE, D), {}, 'community_edge_ops', 'delete', (D, CE), {}),
    (
        'community_edge_get_by_uuid',
        (None, D, 'hm1'),
        {},
        'community_edge_ops',
        'get_by_uuid',
        (D, 'hm1'),
        {},
    ),
    (
        'community_edge_get_by_uuids',
        (None, D, ['hm1']),
        {},
        'community_edge_ops',
        'get_by_uuids',
        (D, ['hm1']),
        {},
    ),
    (
        'community_edge_get_by_group_ids',
        (None, D, ['g1'], 4, None),
        {},
        'community_edge_ops',
        'get_by_group_ids',
        (D, ['g1'], 4, None),
        {},
    ),
    ('has_episode_edge_save', (HE, D), {}, 'has_episode_edge_ops', 'save', (D, HE), {}),
    ('has_episode_edge_delete', (HE, D), {}, 'has_episode_edge_ops', 'delete', (D, HE), {}),
    (
        'has_episode_edge_get_by_uuid',
        (None, D, 'he1'),
        {},
        'has_episode_edge_ops',
        'get_by_uuid',
        (D, 'he1'),
        {},
    ),
    (
        'has_episode_edge_get_by_uuids',
        (None, D, ['he1']),
        {},
        'has_episode_edge_ops',
        'get_by_uuids',
        (D, ['he1']),
        {},
    ),
    (
        'has_episode_edge_get_by_group_ids',
        (None, D, ['g1'], None, 'x'),
        {},
        'has_episode_edge_ops',
        'get_by_group_ids',
        (D, ['g1'], None, 'x'),
        {},
    ),
    ('next_episode_edge_save', (NE, D), {}, 'next_episode_edge_ops', 'save', (D, NE), {}),
    ('next_episode_edge_delete', (NE, D), {}, 'next_episode_edge_ops', 'delete', (D, NE), {}),
    (
        'next_episode_edge_get_by_uuid',
        (None, D, 'ne1'),
        {},
        'next_episode_edge_ops',
        'get_by_uuid',
        (D, 'ne1'),
        {},
    ),
    (
        'next_episode_edge_get_by_uuids',
        (None, D, ['ne1']),
        {},
        'next_episode_edge_ops',
        'get_by_uuids',
        (D, ['ne1']),
        {},
    ),
    (
        'next_episode_edge_get_by_group_ids',
        (None, D, ['g1'], 10, 'q'),
        {},
        'next_episode_edge_ops',
        'get_by_group_ids',
        (D, ['g1'], 10, 'q'),
        {},
    ),
    ('get_mentioned_nodes', (D, [EP]), {}, 'graph_ops', 'get_mentioned_nodes', (D, [EP]), {}),
    (
        'get_communities_by_nodes',
        (D, [N]),
        {},
        'graph_ops',
        'get_communities_by_nodes',
        (D, [N]),
        {},
    ),
    ('clear_data', (D, ['g1']), {}, 'graph_ops', 'clear_data', (D, ['g1']), {}),
    ('clear_data', (D,), {}, 'graph_ops', 'clear_data', (D, None), {}),
    (
        'get_community_clusters',
        (D, None),
        {},
        'graph_ops',
        'get_community_clusters',
        (D, None),
        {},
    ),
    (
        'remove_communities',
        (D,),
        {'group_ids': ['g1']},
        'graph_ops',
        'remove_communities',
        (D,),
        {'group_ids': ['g1']},
    ),
]


@pytest.mark.parametrize(
    'name, args, kwargs, attr, method, exp_args, exp_kwargs',
    DELEGATIONS,
    ids=[f'{c[0]}->{c[3]}.{c[4]}' for c in DELEGATIONS],
)
async def test_graph_operations_delegate(
    name: str,
    args: tuple,
    kwargs: dict,
    attr: str,
    method: str,
    exp_args: tuple,
    exp_kwargs: dict,
) -> None:
    d = make_driver()
    target = getattr(getattr(d, attr), method)
    sentinel = object()
    target.return_value = sentinel
    result = await getattr(SpannerLegacyGraphOperations(), name)(*fill(d, args), **kwargs)
    target.assert_awaited_once_with(*fill(d, exp_args), **exp_kwargs)
    assert ops_calls(d) == {attr: [method]}
    assert d.read.await_count == 0
    if '_get_' in name or name.startswith('get_'):
        assert result is sentinel


# --- dispatching deletes ---------------------------------------------------------------------


@pytest.mark.parametrize(
    'node, attr',
    [
        (N, 'entity_node_ops'),
        (EP, 'episode_node_ops'),
        (COMMUNITY, 'community_node_ops'),
        (SAGA, 'saga_node_ops'),
    ],
)
async def test_node_delete_dispatches_on_type(node: Any, attr: str) -> None:
    d = make_driver()
    await SpannerLegacyGraphOperations().node_delete(node, d)
    getattr(d, attr).delete.assert_awaited_once_with(d, node)
    assert ops_calls(d) == {attr: ['delete']}


@pytest.mark.parametrize(
    'edge, attr',
    [
        (ED, 'entity_edge_ops'),
        (EPE, 'episodic_edge_ops'),
        (CE, 'community_edge_ops'),
        (HE, 'has_episode_edge_ops'),
        (NE, 'next_episode_edge_ops'),
    ],
)
async def test_edge_delete_dispatches_on_type(edge: Any, attr: str) -> None:
    d = make_driver()
    await SpannerLegacyGraphOperations().edge_delete(edge, d)
    getattr(d, attr).delete.assert_awaited_once_with(d, edge)
    assert ops_calls(d) == {attr: ['delete']}


async def test_node_delete_by_uuids_fans_out_only_to_tables_holding_uuids() -> None:
    g = SpannerLegacyGraphOperations()
    d = make_driver([[{'kind': 'Entity', 'uuid': 'a'}, {'kind': 'Saga', 'uuid': 's'}]])
    await g.node_delete_by_uuids(None, d, ['a', 's', 'missing', 'a'], batch_size=50)
    sql, params, _ = d.read.await_args.args
    assert sql.count('UNION ALL') == 3 and 'group_id' not in sql
    assert params == {'uuids': ['a', 's', 'missing']}
    d.entity_node_ops.delete_by_uuids.assert_awaited_once_with(d, ['a'], batch_size=50)
    d.saga_node_ops.delete_by_uuids.assert_awaited_once_with(d, ['s'], batch_size=50)
    assert set(ops_calls(d)) == {'entity_node_ops', 'saga_node_ops'}

    d = make_driver([[{'kind': 'Episodic', 'uuid': 'e'}]])
    await g.node_delete_by_uuids(None, d, ['e'], group_id='g1')
    sql, params, _ = d.read.await_args.args
    assert sql.count('AND group_id = @group_id') == 4 and params['group_id'] == 'g1'
    d.episode_node_ops.delete_by_uuids.assert_awaited_once_with(d, ['e'], batch_size=100)

    d = make_driver()
    await g.node_delete_by_uuids(None, d, [])
    assert d.read.await_count == 0 and not ops_calls(d)


async def test_node_delete_by_group_id_skips_sagas() -> None:
    d = make_driver()
    await SpannerLegacyGraphOperations().node_delete_by_group_id(EntityNode, d, 'g1', 25)
    for attr in ('entity_node_ops', 'episode_node_ops', 'community_node_ops'):
        getattr(d, attr).delete_by_group_id.assert_awaited_once_with(d, 'g1', batch_size=25)
    d.saga_node_ops.delete_by_group_id.assert_not_awaited()


async def test_edge_delete_by_uuids_fans_out_only_to_tables_holding_uuids() -> None:
    d = make_driver([[{'kind': 'RelatesTo', 'uuid': 'e1'}, {'kind': 'Mentions', 'uuid': 'm1'}]])
    await SpannerLegacyGraphOperations().edge_delete_by_uuids(None, d, ['e1', 'm1'])
    sql, _, _ = d.read.await_args.args
    assert sql.count('UNION ALL') == 4 and 'edge_uuid IN UNNEST(@uuids)' in sql
    d.entity_edge_ops.delete_by_uuids.assert_awaited_once_with(d, ['e1'])
    d.episodic_edge_ops.delete_by_uuids.assert_awaited_once_with(d, ['m1'])
    assert set(ops_calls(d)) == {'entity_edge_ops', 'episodic_edge_ops'}


@pytest.mark.parametrize(
    'name, attr, table, is_node',
    [
        ('episodic_node_delete_by_uuids', 'episode_node_ops', 'Episodic', True),
        ('community_node_delete_by_uuids', 'community_node_ops', 'Community', True),
        ('saga_node_delete_by_uuids', 'saga_node_ops', 'Saga', True),
        ('episodic_edge_delete_by_uuids', 'episodic_edge_ops', 'Mentions', False),
        ('community_edge_delete_by_uuids', 'community_edge_ops', 'HasMember', False),
        ('has_episode_edge_delete_by_uuids', 'has_episode_edge_ops', 'HasEpisode', False),
        ('next_episode_edge_delete_by_uuids', 'next_episode_edge_ops', 'NextEpisode', False),
    ],
)
async def test_per_kind_delete_by_uuids(name: str, attr: str, table: str, is_node: bool) -> None:
    g = SpannerLegacyGraphOperations()
    extra = {'batch_size': 100} if is_node else {}
    # No group: straight to the ops, no locate read.
    d = make_driver()
    await getattr(g, name)(None, d, ['x', 'y'])
    getattr(d, attr).delete_by_uuids.assert_awaited_once_with(d, ['x', 'y'], **extra)
    assert d.read.await_count == 0
    # With a group: a group-scoped locate read narrows the uuids.
    d = make_driver([[{'kind': table, 'uuid': 'y'}]])
    await getattr(g, name)(None, d, ['x', 'y'], 'g1')
    sql = d.read.await_args.args[0]
    assert f'FROM {table} ' in sql and 'UNION' not in sql and '@group_id' in sql
    getattr(d, attr).delete_by_uuids.assert_awaited_once_with(d, ['y'], **extra)
    # Nothing in the group: no delete at all.
    d = make_driver([[]])
    await getattr(g, name)(None, d, ['x'], 'g1')
    getattr(d, attr).delete_by_uuids.assert_not_awaited()


# --- bulk savers -----------------------------------------------------------------------------


def assert_same_entity(a: EntityNode, b: EntityNode) -> None:
    assert (a.uuid, a.name, a.group_id, a.summary) == (b.uuid, b.name, b.group_id, b.summary)
    assert sorted(a.labels) == sorted(set(b.labels) | {'Entity'})
    assert a.name_embedding == b.name_embedding and a.created_at == b.created_at
    assert a.attributes == b.attributes


async def test_bulk_utils_transaction_drives_the_bulk_savers() -> None:
    """Run the real bulk_utils transaction function so rows have their production shape."""
    d = make_driver()
    d.graph_operations_interface = SpannerLegacyGraphOperations()
    ep = episode()
    n1 = entity_node('n1')
    n2 = entity_node('n2', name='Bob', labels=['Entity'], attributes={})
    m = simple_edge(EpisodicEdge, 'm1')
    e = entity_edge()
    tx = object()
    await add_nodes_and_edges_bulk_tx(tx, [ep], [m], [n1, n2], [e], MagicMock(), driver=d)  # type: ignore[arg-type]

    call = d.episode_node_ops.save_bulk.await_args
    assert call.args[0] is d and call.kwargs == {'tx': tx, 'batch_size': 100}
    assert call.args[1][0].model_dump() == ep.model_dump()

    call = d.entity_node_ops.save_bulk.await_args
    assert call.args[0] is d and call.kwargs == {'tx': tx, 'batch_size': 100}
    assert_same_entity(call.args[1][0], n1)
    assert_same_entity(call.args[1][1], n2)

    call = d.episodic_edge_ops.save_bulk.await_args
    assert call.kwargs == {'tx': tx, 'batch_size': 100}
    assert call.args[1][0].model_dump() == m.model_dump()

    call = d.entity_edge_ops.save_bulk.await_args
    assert call.kwargs == {'tx': tx, 'batch_size': 100}
    assert call.args[1][0].model_dump() == e.model_dump()


async def test_node_save_bulk_rebuilds_models_from_rows() -> None:
    g = SpannerLegacyGraphOperations()
    d = make_driver()
    row = {
        'uuid': 'n9',
        'name': 'Zed',
        'group_id': 'g1',
        'summary': 's',
        'created_at': T0.isoformat(),
        'name_embedding': [1.0, 2.0],
        'labels': ['Entity', 'Thing'],
        'color': 'red',
        'size': 3,
    }
    await g.node_save_bulk(None, d, 'TX', [row], batch_size=7)
    (node,) = d.entity_node_ops.save_bulk.await_args.args[1]
    assert node.created_at == T0 and node.attributes == {'color': 'red', 'size': 3}
    assert d.entity_node_ops.save_bulk.await_args.kwargs == {'tx': 'TX', 'batch_size': 7}
    # A model_dump()-shaped row carries a nested attributes dict.
    await g.node_save_bulk(None, d, 'TX', [entity_node('n8').model_dump()])
    (node,) = d.entity_node_ops.save_bulk.await_args.args[1]
    assert node.attributes == {'sku': 'A-1', 'age': 30, 'born': T0}


async def test_edge_and_episode_save_bulk_rebuild_models_from_rows() -> None:
    g = SpannerLegacyGraphOperations()
    d = make_driver()
    edge_row = {
        'uuid': 'e9',
        'source_node_uuid': 'a',
        'target_node_uuid': 'b',
        'name': 'R',
        'fact': 'f',
        'group_id': 'g1',
        'episodes': ['ep'],
        'created_at': T0.isoformat(),
        'expired_at': None,
        'valid_at': (T0 + timedelta(days=1)).isoformat(),
        'invalid_at': None,
        'reference_time': T0,
        'fact_embedding': None,
        'confidence': 0.9,
    }
    await g.edge_save_bulk(None, d, 'TX', [edge_row])
    (edge,) = d.entity_edge_ops.save_bulk.await_args.args[1]
    assert edge.created_at == T0 and edge.valid_at == T0 + timedelta(days=1)
    assert edge.attributes == {'confidence': 0.9} and edge.expired_at is None
    assert d.entity_edge_ops.save_bulk.await_args.kwargs['tx'] == 'TX'

    ep_row = dict(episode('ep9'))
    ep_row['source'] = 'json'
    ep_row['valid_at'] = T0.isoformat()
    ep_row.pop('labels')
    await g.episodic_node_save_bulk(None, d, 'TX', [ep_row])
    (saved,) = d.episode_node_ops.save_bulk.await_args.args[1]
    assert saved.source == EpisodeType.json and saved.valid_at == T0
    assert d.episode_node_ops.save_bulk.await_args.kwargs['tx'] == 'TX'

    m = simple_edge(EpisodicEdge, 'm1')
    await g.episodic_edge_save_bulk(
        None, d, 'TX', [{**m.model_dump(), 'created_at': T0.isoformat()}]
    )
    assert d.episodic_edge_ops.save_bulk.await_args.args[1][0].created_at == T0
    assert d.episodic_edge_ops.save_bulk.await_args.kwargs == {'tx': 'TX', 'batch_size': 100}


@pytest.mark.parametrize(
    'name, attr, model',
    [
        ('community_node_save_bulk', 'community_node_ops', COMMUNITY),
        ('saga_node_save_bulk', 'saga_node_ops', SAGA),
        ('has_episode_edge_save_bulk', 'has_episode_edge_ops', simple_edge(HasEpisodeEdge, 'h')),
        ('next_episode_edge_save_bulk', 'next_episode_edge_ops', simple_edge(NextEpisodeEdge, 'n')),
    ],
)
async def test_other_bulk_savers_accept_models_or_dicts(name: str, attr: str, model: Any) -> None:
    d = make_driver()
    await getattr(SpannerLegacyGraphOperations(), name)(
        None, d, 'TX', [model, model.model_dump()], 5
    )
    call = getattr(d, attr).save_bulk.await_args
    assert call.kwargs == {'tx': 'TX', 'batch_size': 5}
    first, second = call.args[1]
    assert first is model and type(second) is type(model)
    assert second.model_dump() == model.model_dump()


# --- SQL-backed adapter methods (mock driver.read) -------------------------------------------


async def test_retrieve_episodes_passes_source_value_and_returns_oldest_first() -> None:
    d = make_driver()
    new, old = episode('new', valid_at=T0 + timedelta(hours=1)), episode('old')
    d.episode_node_ops.retrieve_episodes.return_value = [new, old]
    result = await SpannerLegacyGraphOperations().retrieve_episodes(
        d, T0, 2, ['g1'], EpisodeType.message, None
    )
    d.episode_node_ops.retrieve_episodes.assert_awaited_once_with(d, T0, 2, ['g1'], 'message', None)
    assert [e.uuid for e in result] == ['old', 'new']


async def test_node_get_by_uuids_filters_group() -> None:
    d = make_driver()
    g = SpannerLegacyGraphOperations()
    d.entity_node_ops.get_by_uuids.return_value = [N, entity_node('n2', group_id='g2')]
    result = await g.node_get_by_uuids(EntityNode, d, ['n1', 'n2'], 'g1')
    d.entity_node_ops.get_by_uuids.assert_awaited_once_with(d, ['n1', 'n2'])
    assert [n.uuid for n in result] == ['n1']
    assert len(await g.node_get_by_uuids(EntityNode, d, ['n1', 'n2'])) == 2


async def test_episodic_edge_get_by_uuids_raises_when_nothing_matches() -> None:
    g = SpannerLegacyGraphOperations()
    d = make_driver()
    d.episodic_edge_ops.get_by_uuids.return_value = []
    with pytest.raises(EdgeNotFoundError):
        await g.episodic_edge_get_by_uuids(EpisodicEdge, d, ['m1'])
    assert await g.episodic_edge_get_by_uuids(EpisodicEdge, d, []) == []
    d.episodic_edge_ops.get_by_uuids.return_value = ['x']
    assert await g.episodic_edge_get_by_uuids(EpisodicEdge, d, ['m1']) == ['x']


async def test_node_get_by_names_builds_group_scoped_query() -> None:
    g = SpannerLegacyGraphOperations()
    d = make_driver([[entity_node_to_row(N)]])
    result = await g.node_get_by_names(EntityNode, d, ['Alice', '', 'Alice'], ['g1', 'g1'])
    sql, params, _ = d.read.await_args.args
    assert 'name IN UNNEST(@names)' in sql and 'group_id IN UNNEST(@group_ids)' in sql
    assert params == {'group_ids': ['g1'], 'names': ['Alice']}
    assert result[0].uuid == 'n1' and result[0].attributes['sku'] == 'A-1'
    d = make_driver()
    assert await g.node_get_by_names(EntityNode, d, [''], ['g1']) == []
    assert d.read.await_count == 0


async def test_node_get_by_attribute_values_query_shape_and_validation() -> None:
    g = SpannerLegacyGraphOperations()
    d = make_driver([[entity_node_to_row(N)]])
    result = await g.node_get_by_attribute_values(
        EntityNode, d, 'g1', {'sku': ' A-1 ', 'age': 30, 'vip': True, 'gone': None}, ['Person'], 5
    )
    sql, params = d.read.await_args.args
    assert "TRIM(JSON_VALUE(attributes, '$.sku')) = @v0" in sql
    assert "SAFE_CAST(JSON_VALUE(attributes, '$.age') AS FLOAT64) = @v1" in sql
    assert "JSON_VALUE(attributes, '$.vip') = @v2" in sql
    assert "JSON_VALUE(attributes, '$.gone') IS NULL" in sql
    assert '@label0 IN UNNEST(labels)' in sql and sql.endswith('LIMIT @limit')
    assert params == {
        'group_id': 'g1',
        'v0': 'A-1',
        'v1': 30.0,
        'v2': 'true',
        'label0': 'Person',
        'limit': 5,
    }
    assert [n.uuid for n in result] == ['n1']
    with pytest.raises(ValueError):
        await g.node_get_by_attribute_values(EntityNode, make_driver(), 'g1', {"x') OR TRUE --": 1})
    with pytest.raises(TypeError):
        await g.node_get_by_attribute_values(EntityNode, make_driver(), 'g1', {'k': [1]})
    assert await g.node_get_by_attribute_values(EntityNode, make_driver(), 'g1', {}) == []


async def test_load_embeddings_bulk_batches_and_skips_missing() -> None:
    g = SpannerLegacyGraphOperations()
    d = make_driver([[{'uuid': 'n1', 'embedding': [1, 2]}, {'uuid': 'n2', 'embedding': None}], []])
    emb = await g.node_load_embeddings_bulk(d, [N, entity_node('n2'), N], batch_size=1)
    assert emb == {'n1': [1.0, 2.0]}
    assert d.read.await_count == 2 and 'FROM Entity' in d.read.await_args_list[0].args[0]
    assert not ops_calls(d)
    d = make_driver([[{'uuid': 'e1', 'embedding': [0.5]}]])
    emb = await g.edge_load_embeddings_bulk(d, [ED])
    assert (
        'edge_uuid AS uuid, fact_embedding AS embedding FROM RelatesTo'
        in (d.read.await_args.args[0])
    )
    assert emb == {'e1': [0.5]}


async def test_saga_helpers_query_shape() -> None:
    g = SpannerLegacyGraphOperations()
    d = make_driver([[{'uuid': 'ep0'}], []])
    assert await g.saga_get_previous_episode_uuid(d, 's1', 'ep1') == 'ep0'
    sql, params = d.read.await_args.args
    assert 'h.uuid = @saga_uuid' in sql and 'e.uuid != @current_episode_uuid' in sql
    assert 'ORDER BY e.valid_at DESC, e.created_at DESC' in sql
    assert params == {'saga_uuid': 's1', 'current_episode_uuid': 'ep1'}
    assert await g.saga_get_previous_episode_uuid(d, 's1', 'ep1') is None

    rows = [
        {'content': 'a', 'valid_at': T0, 'created_at': T0},
        {'content': '', 'valid_at': T0, 'created_at': T0},
    ]
    d = make_driver([rows, rows])
    assert await g.saga_get_episode_contents(d, 's1', since=T0, limit=3) == [('a', T0, T0)]
    sql, params = d.read.await_args.args
    assert 'e.created_at > @since' in sql and sql.endswith('LIMIT @limit')
    assert 'ORDER BY e.created_at ASC, e.valid_at ASC' in sql
    assert params == {'saga_uuid': 's1', 'since': T0, 'limit': 3}
    await g.saga_get_episode_contents(d, 's1')
    sql, params = d.read.await_args.args
    assert '@since' not in sql and params == {'saga_uuid': 's1', 'limit': 200}


async def test_determine_entity_community_tuple_shapes() -> None:
    g = SpannerLegacyGraphOperations()
    c_row = community_node_to_row(COMMUNITY)
    d = make_driver([[c_row]])
    comm, is_new = await g.determine_entity_community(d, N)
    assert comm is not None and comm.uuid == 'c1' and is_new is False and d.read.await_count == 1
    d = make_driver([[], [{**c_row, 'member_count': 3}]])
    comm, is_new = await g.determine_entity_community(d, N)
    assert comm is not None and comm.uuid == 'c1' and is_new is True
    assert 'GROUP BY m.uuid' in d.read.await_args.args[0]
    d = make_driver([[], []])
    assert await g.determine_entity_community(d, N) == (None, False)


# --- search interface ------------------------------------------------------------------------

SEARCH_DELEGATIONS: list[tuple[str, tuple]] = [
    ('edge_fulltext_search', (D, 'q', FILTERS, ['g'], 5)),
    ('edge_similarity_search', (D, [0.1], 'a', 'b', FILTERS, ['g'], 5, 0.5)),
    ('node_fulltext_search', (D, 'q', FILTERS, None, 7)),
    ('node_similarity_search', (D, [0.1], FILTERS, ['g'], 3, 0.2)),
    ('episode_fulltext_search', (D, 'q', FILTERS, ['g'], 4)),
    ('edge_bfs_search', (D, ['o'], 2, FILTERS, ['g'], 9)),
    ('node_bfs_search', (D, ['o'], FILTERS, 3, None, 8)),
    ('community_fulltext_search', (D, 'q', ['g'], 2)),
    ('community_similarity_search', (D, [0.1], ['g'], 2, 0.3)),
]


@pytest.mark.parametrize('name, args', SEARCH_DELEGATIONS, ids=[c[0] for c in SEARCH_DELEGATIONS])
async def test_search_interface_delegates_to_search_ops(name: str, args: tuple) -> None:
    d = make_driver()
    target = getattr(d.search_ops, name)
    sentinel = object()
    target.return_value = sentinel
    assert await getattr(SpannerLegacySearchInterface(), name)(*fill(d, args)) is sentinel
    target.assert_awaited_once_with(*fill(d, args))
    assert ops_calls(d) == {'search_ops': [name]}


async def test_bfs_searches_short_circuit_without_origins_or_depth() -> None:
    s = SpannerLegacySearchInterface()
    d = make_driver()
    assert await s.edge_bfs_search(d, None, 2, FILTERS) == []
    assert await s.node_bfs_search(d, [], FILTERS, 2) == []
    assert await s.node_bfs_search(d, ['o'], FILTERS, 0) == []
    assert not ops_calls(d)


async def test_get_embeddings_for_communities() -> None:
    d = make_driver([[{'uuid': 'c1', 'embedding': [1, 0]}]])
    result = await SpannerLegacySearchInterface().get_embeddings_for_communities(d, [COMMUNITY])
    assert result == {'c1': [1.0, 0.0]}
    assert 'FROM Community' in d.read.await_args.args[0]


async def test_node_distance_reranker_shape() -> None:
    s = SpannerLegacySearchInterface()
    d = make_driver([[{'uuid': 'b'}]])
    uuids, scores = await s.node_distance_reranker(d, ['a', 'center', 'b'], 'center')
    assert uuids == ['center', 'b', 'a'] and scores == [10.0, 1.0, 0.0]
    sql, params, _ = d.read.await_args.args
    assert 'UNION DISTINCT' in sql and params == {'center_uuid': 'center', 'uuids': ['a', 'b']}
    d = make_driver([[{'uuid': 'b'}]])
    assert await s.node_distance_reranker(d, ['a', 'b'], 'center', min_score=0.5) == (['b'], [1.0])
    d = make_driver()
    assert await s.node_distance_reranker(d, ['center'], 'center') == (['center'], [10.0])
    assert d.read.await_count == 0


async def test_episode_mentions_reranker_shape() -> None:
    d = make_driver([[{'uuid': 'b', 'score': 3}, {'uuid': 'a', 'score': 1}]])
    uuids, scores = await SpannerLegacySearchInterface().episode_mentions_reranker(
        d, [['a', 'b', 'c'], ['b']], min_score=1
    )
    assert uuids == ['b', 'a'] and scores == [3, 1]
    assert 'GROUP BY target_node_uuid' in d.read.await_args.args[0]


async def test_filter_builders_delegate_to_spanner_search_ops() -> None:
    s = SpannerLegacySearchInterface()
    with patch.object(legacy, 'SpannerSearchOperations') as ops_cls:
        ops_cls.return_value.build_node_search_filters.return_value = 'NF'
        ops_cls.return_value.build_edge_search_filters.return_value = 'EF'
        assert s.build_node_search_filters(FILTERS) == 'NF'
        assert s.build_edge_search_filters(FILTERS) == 'EF'
        ops_cls.return_value.build_node_search_filters.assert_called_once_with(FILTERS)
        ops_cls.return_value.build_edge_search_filters.assert_called_once_with(FILTERS)
