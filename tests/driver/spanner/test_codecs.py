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

# Hermetic tests for value codecs, row encoders/decoders and mutation helpers.

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from enum import Enum
from types import SimpleNamespace
from typing import Any

import pytest
from google.api_core.datetime_helpers import DatetimeWithNanoseconds
from google.cloud.spanner_v1 import JsonObject, param_types
from pydantic import BaseModel

from graphiti_core.driver.spanner import rows
from graphiti_core.driver.spanner.codecs import (
    coerce_embedding,
    encode_params,
    from_spanner_value,
    infer_param_types,
    rows_to_dicts,
    to_spanner_value,
)
from graphiti_core.driver.spanner.mutations import Mutation, chunk_mutations, delete, upsert
from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    HasEpisodeEdge,
    NextEpisodeEdge,
)
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode, SagaNode

A = param_types.Array
T0 = datetime(2026, 10, 28, 12, 30, 15, 123456, tzinfo=timezone.utc)
T1 = T0 + timedelta(days=1)
T2 = T0 + timedelta(days=2)
T3 = T0 + timedelta(days=3)


class Color(Enum):
    red = 'red'


# ---------------------------------------------------------------------------
# infer_param_types / encode_params
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ('x', param_types.STRING),
        (True, param_types.BOOL),
        (3, param_types.INT64),
        (0.5, param_types.FLOAT64),
        (T0, param_types.TIMESTAMP),
        ({'a': 1}, param_types.JSON),
        (JsonObject({'a': 1}), param_types.JSON),
        (Color.red, param_types.STRING),
        (['a', 'b'], A(param_types.STRING)),
        ([0.1, 0.2], A(param_types.FLOAT32)),
        ([1, 0.5], A(param_types.FLOAT32)),
        ([1, 2], A(param_types.INT64)),
        ([True, False], A(param_types.BOOL)),
        ([T0], A(param_types.TIMESTAMP)),
        ([], A(param_types.STRING)),
    ],
)
def test_infer_param_types(value: Any, expected: Any):
    assert infer_param_types({'p': value}) == {'p': expected}


def test_infer_param_types_none_requires_explicit_type():
    with pytest.raises(ValueError, match="'cursor'"):
        infer_param_types({'cursor': None, 'uuid': 'x'})
    types = infer_param_types({'cursor': None}, {'cursor': param_types.STRING})
    assert types == {'cursor': param_types.STRING}


def test_infer_param_types_explicit_wins_and_extra_explicit_ignored():
    types = infer_param_types(
        {'v': [1, 2]}, {'v': A(param_types.FLOAT32), 'unused': param_types.STRING}
    )
    assert types == {'v': A(param_types.FLOAT32)}


def test_infer_param_types_rejects_mixed_lists():
    with pytest.raises(ValueError, match="'mixed'"):
        infer_param_types({'mixed': ['a', 1]})


def test_encode_params_matches_types():
    naive = datetime(2026, 1, 1, 0, 0)
    params = {'vec': [1, 0, 0.5], 'score': 1, 'meta': {'k': T0}, 'at': naive, 'ids': ['a']}
    types = infer_param_types(params, {'score': param_types.FLOAT64})
    encoded = encode_params(params, types)
    assert encoded['vec'] == [1.0, 0.0, 0.5]
    assert all(type(item) is float for item in encoded['vec'])
    assert type(encoded['score']) is float
    assert isinstance(encoded['meta'], JsonObject)
    assert from_spanner_value(encoded['meta']) == {'k': T0.isoformat()}
    assert encoded['at'] == naive.replace(tzinfo=timezone.utc)
    assert encoded['ids'] == ['a']


# ---------------------------------------------------------------------------
# to_spanner_value / from_spanner_value / coerce_embedding / rows_to_dicts
# ---------------------------------------------------------------------------


def test_to_spanner_value():
    plus_two = timezone(timedelta(hours=2))
    local = datetime(2026, 1, 1, 14, 0, tzinfo=plus_two)
    converted = to_spanner_value(local)
    assert converted == local
    assert converted.tzinfo == timezone.utc
    assert to_spanner_value(datetime(2026, 1, 1)).tzinfo == timezone.utc
    assert to_spanner_value([1, 2]) == [1.0, 2.0]
    assert all(type(item) is float for item in to_spanner_value([1, 2]))
    assert to_spanner_value(['a', 'b']) == ['a', 'b']
    assert to_spanner_value(Color.red) == 'red'
    assert to_spanner_value(None) is None
    json_value = to_spanner_value({'when': T0, 'tags': ('a', 'b'), 'color': Color.red})
    assert isinstance(json_value, JsonObject)
    assert from_spanner_value(json_value) == {
        'when': T0.isoformat(),
        'tags': ['a', 'b'],
        'color': 'red',
    }


def test_from_spanner_value_json():
    value = from_spanner_value(JsonObject({'age': 30, 'nested': {'a': [1, 2]}}))
    assert type(value) is dict
    assert value == {'age': 30, 'nested': {'a': [1, 2]}}
    assert type(value['nested']) is dict
    assert from_spanner_value(JsonObject(None)) is None
    assert from_spanner_value(JsonObject([1, 'a'])) == [1, 'a']
    assert from_spanner_value(JsonObject({})) == {}


def test_from_spanner_value_datetime():
    value = DatetimeWithNanoseconds(
        2026, 10, 28, 12, 30, 15, nanosecond=123456789, tzinfo=timezone.utc
    )
    converted = from_spanner_value(value)
    assert type(converted) is datetime
    assert converted == datetime(2026, 10, 28, 12, 30, 15, 123456, tzinfo=timezone.utc)
    assert converted.tzinfo is not None
    as_list = from_spanner_value([value, None])
    assert type(as_list[0]) is datetime
    assert as_list[1] is None
    assert from_spanner_value(['a']) == ['a']
    assert from_spanner_value(1.5) == 1.5


def test_coerce_embedding():
    result = coerce_embedding([1, 2, 0.5])
    assert result is not None
    assert result == [1.0, 2.0, 0.5]
    assert all(type(item) is float for item in result)
    assert coerce_embedding(None) is None
    assert coerce_embedding([]) == []


class _FakeResultSet:
    """Mimics StreamedResultSet: ``fields`` is only populated once iteration starts."""

    def __init__(self, names: list[str], data: list[list[Any]]):
        self._names = names
        self._data = data
        self.fields: list[Any] | None = None

    def __iter__(self):
        self.fields = [SimpleNamespace(name=name) for name in self._names]
        return iter(self._data)


def test_rows_to_dicts_reads_fields_after_iteration():
    stamp = DatetimeWithNanoseconds(2026, 1, 1, tzinfo=timezone.utc)
    result = _FakeResultSet(
        ['uuid', 'attributes', 'created_at'], [['a', JsonObject({'k': 1}), stamp]]
    )
    records = rows_to_dicts(result)
    assert records == [{'uuid': 'a', 'attributes': {'k': 1}, 'created_at': stamp}]
    assert type(records[0]['attributes']) is dict
    assert type(records[0]['created_at']) is datetime
    assert rows_to_dicts(_FakeResultSet(['uuid'], [])) == []


# ---------------------------------------------------------------------------
# Row encoders and decoders
# ---------------------------------------------------------------------------


def _entity_node() -> EntityNode:
    return EntityNode(
        uuid='entity-1',
        name='Alice',
        group_id='group-1',
        labels=['Person'],
        name_embedding=[0.5, 0.25, 1.0],
        summary='Alice is an engineer',
        created_at=T0,
        attributes={'age': 30},
    )


def _episodic_node() -> EpisodicNode:
    return EpisodicNode(
        uuid='episode-1',
        name='episode',
        group_id='group-1',
        source=EpisodeType.message,
        source_description='chat',
        content='user: hello',
        valid_at=T1,
        created_at=T0,
        entity_edges=['edge-1', 'edge-2'],
        episode_metadata={'channel': 'web', 'turn': 3},
    )


def _community_node() -> CommunityNode:
    return CommunityNode(
        uuid='community-1',
        name='engineers',
        group_id='group-1',
        name_embedding=[0.1, 0.2],
        summary='people who build',
        created_at=T0,
    )


def _saga_node() -> SagaNode:
    return SagaNode(
        uuid='saga-1',
        name='onboarding',
        group_id='group-1',
        summary='the onboarding saga',
        first_episode_uuid='episode-1',
        last_episode_uuid='episode-9',
        last_summarized_at=T2,
        last_summarized_episode_valid_at=T1,
        created_at=T0,
    )


def _entity_edge() -> EntityEdge:
    return EntityEdge(
        uuid='edge-1',
        source_node_uuid='entity-1',
        target_node_uuid='entity-2',
        group_id='group-1',
        name='WORKS_WITH',
        fact='Alice works with Bob',
        fact_embedding=[0.5, 0.5],
        episodes=['episode-1'],
        created_at=T0,
        valid_at=T1,
        invalid_at=T2,
        expired_at=T3,
        reference_time=T1,
        attributes={'since': 2020, 'team': 'core'},
    )


def _simple_edge(cls: type[BaseModel]) -> Any:
    return cls(
        uuid=f'{cls.__name__}-1',
        source_node_uuid='source-1',
        target_node_uuid='target-1',
        group_id='group-1',
        created_at=T0,
    )


ROUND_TRIPS: list[tuple[str, Callable[[], Any], Any, Any, tuple[str, ...], bool]] = [
    (
        'entity_node',
        _entity_node,
        rows.entity_node_to_row,
        rows.entity_node_from_row,
        rows.ENTITY_NODE_COLUMNS,
        False,
    ),
    (
        'episodic_node',
        _episodic_node,
        rows.episodic_node_to_row,
        rows.episodic_node_from_row,
        rows.EPISODIC_NODE_COLUMNS,
        False,
    ),
    (
        'community_node',
        _community_node,
        rows.community_node_to_row,
        rows.community_node_from_row,
        rows.COMMUNITY_NODE_COLUMNS,
        False,
    ),
    (
        'saga_node',
        _saga_node,
        rows.saga_node_to_row,
        rows.saga_node_from_row,
        rows.SAGA_NODE_COLUMNS,
        False,
    ),
    (
        'entity_edge',
        _entity_edge,
        rows.entity_edge_to_row,
        rows.entity_edge_from_row,
        rows.ENTITY_EDGE_COLUMNS,
        True,
    ),
    (
        'episodic_edge',
        lambda: _simple_edge(EpisodicEdge),
        rows.episodic_edge_to_row,
        rows.episodic_edge_from_row,
        rows.EPISODIC_EDGE_COLUMNS,
        True,
    ),
    (
        'community_edge',
        lambda: _simple_edge(CommunityEdge),
        rows.community_edge_to_row,
        rows.community_edge_from_row,
        rows.COMMUNITY_EDGE_COLUMNS,
        True,
    ),
    (
        'has_episode_edge',
        lambda: _simple_edge(HasEpisodeEdge),
        rows.has_episode_edge_to_row,
        rows.has_episode_edge_from_row,
        rows.HAS_EPISODE_EDGE_COLUMNS,
        True,
    ),
    (
        'next_episode_edge',
        lambda: _simple_edge(NextEpisodeEdge),
        rows.next_episode_edge_to_row,
        rows.next_episode_edge_from_row,
        rows.NEXT_EPISODE_EDGE_COLUMNS,
        True,
    ),
]


def _expected(model: Any) -> dict[str, Any]:
    expected = model.model_dump()
    if isinstance(model, EntityNode):
        # The stored label set always contains Entity, as Neo4j's labels(n) does.
        expected['labels'] = model.labels + ['Entity']
    return expected


def _as_selected(row: dict[str, Any], is_edge: bool) -> dict[str, Any]:
    """What SQL with edge_select_columns (or GQL over the property graph) returns."""
    if not is_edge:
        return dict(row)
    selected = {k: v for k, v in row.items() if k not in ('uuid', 'edge_uuid')}
    selected['uuid'] = row['edge_uuid']
    selected['source_node_uuid'] = row['uuid']
    return selected


@pytest.mark.parametrize(
    ('name', 'factory', 'to_row', 'from_row', 'columns', 'is_edge'),
    ROUND_TRIPS,
    ids=[case[0] for case in ROUND_TRIPS],
)
def test_row_round_trip(
    name: str,
    factory: Callable[[], Any],
    to_row: Any,
    from_row: Any,
    columns: tuple[str, ...],
    is_edge: bool,
):
    original = factory()
    row = to_row(original)
    assert tuple(row) == columns
    if is_edge:
        assert row['uuid'] == original.source_node_uuid
        assert row['edge_uuid'] == original.uuid
    decoded_selected = from_row(_as_selected(row, is_edge))
    decoded_raw = from_row(row)
    assert decoded_selected.model_dump() == _expected(original)
    assert decoded_raw.model_dump() == _expected(original)
    for field, value in decoded_selected.model_dump().items():
        if isinstance(value, datetime):
            assert type(value) is datetime, field
            assert value.tzinfo is not None, field


def test_entity_node_with_entity_label_round_trips_exactly():
    node = _entity_node()
    node.labels = ['Person', 'Entity']
    decoded = rows.entity_node_from_row(rows.entity_node_to_row(node))
    assert decoded.model_dump() == node.model_dump()
    assert rows.entity_node_to_row(node)['labels'] == ['Person', 'Entity']


def test_entity_row_values_are_spanner_ready():
    row = rows.entity_node_to_row(_entity_node())
    assert isinstance(row['attributes'], JsonObject)
    assert from_spanner_value(row['attributes']) == {'age': 30}
    assert row['created_at'].tzinfo == timezone.utc
    assert all(type(item) is float for item in row['name_embedding'])


def test_entity_attributes_drop_core_field_collisions():
    node = _entity_node()
    node.attributes = {'age': 30, 'name': 'shadow', 'labels': ['X'], 'created_at': 'never'}
    row = rows.entity_node_to_row(node)
    assert from_spanner_value(row['attributes']) == {'age': 30}
    stored = dict(row)
    stored['attributes'] = JsonObject({'age': 30, 'summary': 'shadow'})
    assert rows.entity_node_from_row(stored).attributes == {'age': 30}


def test_entity_edge_attributes_drop_core_field_collisions():
    edge = _entity_edge()
    edge.attributes = {'since': 2020, 'fact': 'shadow', 'episodes': []}
    row = rows.entity_edge_to_row(edge)
    assert from_spanner_value(row['attributes']) == {'since': 2020}


def test_entity_node_from_row_strips_group_label():
    row = rows.entity_node_to_row(_entity_node())
    row['labels'] = ['Person', 'Entity', 'Entity_group1']
    assert rows.entity_node_from_row(row).labels == ['Person', 'Entity']


def test_entity_node_from_row_defaults():
    row = {
        'uuid': 'n1',
        'name': 'n',
        'group_id': 'g',
        'labels': ['Entity'],
        'summary': None,
        'attributes': None,
        'created_at': DatetimeWithNanoseconds(2026, 1, 1, tzinfo=timezone.utc),
    }
    node = rows.entity_node_from_row(row)
    assert node.summary == ''
    assert node.attributes == {}
    assert node.name_embedding is None
    assert type(node.created_at) is datetime


def test_from_row_requires_created_at():
    row = rows.community_node_to_row(_community_node())
    row['created_at'] = None
    with pytest.raises(ValueError, match='created_at'):
        rows.community_node_from_row(row)


def test_episode_without_metadata():
    node = _episodic_node()
    node.episode_metadata = None
    row = rows.episodic_node_to_row(node)
    assert row['episode_metadata'] is None
    assert row['source'] == 'message'
    assert rows.episodic_node_from_row(row).episode_metadata is None


def test_edge_record_from_table_row_leaves_selected_rows_alone():
    selected = {'uuid': 'edge', 'source_node_uuid': 'src', 'edge_uuid': 'ignored'}
    assert rows.edge_record_from_table_row(selected) == selected
    raw = {'uuid': 'src', 'edge_uuid': 'edge', 'target_node_uuid': 'dst'}
    assert rows.edge_record_from_table_row(raw) == {
        'uuid': 'edge',
        'source_node_uuid': 'src',
        'target_node_uuid': 'dst',
    }


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


def test_upsert_orders_values_by_columns():
    mutation = upsert('T', [{'b': 2, 'a': 1}, {'a': 3, 'b': 4}], ('a', 'b'))
    assert mutation == Mutation(
        kind='insert_or_update', table='T', columns=('a', 'b'), values=((1, 2), (3, 4))
    )
    assert mutation.row_count == 2


def test_upsert_rejects_missing_columns():
    with pytest.raises(ValueError, match="'b'"):
        upsert('T', [{'a': 1}], ('a', 'b'))


def test_delete_wraps_scalar_keys():
    mutation = delete('Entity', ['a', ('b',), ['c']])
    assert mutation.keys == (('a',), ('b',), ('c',))
    assert mutation.kind == 'delete'
    assert delete('RelatesTo', [('src', 'edge')]).keys == (('src', 'edge'),)


def test_chunk_mutations_splits_by_rows_in_order():
    first = upsert('Entity', [{'uuid': str(i)} for i in range(5)], ('uuid',))
    second = delete('Entity', ['x', 'y', 'z'])
    empty = upsert('Saga', [], ('uuid',))
    batches = chunk_mutations([first, empty, second], max_rows=3)
    assert [[m.row_count for m in batch] for batch in batches] == [[3], [2, 1], [2]]
    assert [m.kind for batch in batches for m in batch] == [
        'insert_or_update',
        'insert_or_update',
        'delete',
        'delete',
    ]
    flattened = [row for batch in batches for m in batch for row in m.rows]
    assert flattened == [*first.values, *second.keys]
    assert chunk_mutations([], 10) == []
    with pytest.raises(ValueError):
        chunk_mutations([first], 0)
