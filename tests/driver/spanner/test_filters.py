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

# Hermetic tests of graphiti_core/driver/spanner/filters.py: SearchFilters -> SQL/GQL predicates.

from datetime import datetime, timezone

import pytest
from google.cloud.spanner_v1 import param_types

from graphiti_core.driver.spanner.codecs import STRING_ARRAY
from graphiti_core.driver.spanner.filters import build_edge_filter, build_node_filter
from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters

OP = ComparisonOperator
D1 = datetime(2020, 1, 1, tzinfo=timezone.utc)
D2 = datetime(2022, 1, 1, tzinfo=timezone.utc)


def df(op: ComparisonOperator, date: datetime | None = None) -> DateFilter:
    return DateFilter(date=date, comparison_operator=op)


# --- node filter -------------------------------------------------------------------------


def test_node_filter_without_labels_is_empty():
    compiled = build_node_filter(SearchFilters())
    assert compiled.sql == ()
    assert compiled.gql == ()
    assert compiled.params == {}
    assert compiled.param_types == {}
    assert not compiled.needs_endpoints


def test_node_filter_labels_bound_as_array_parameter():
    compiled = build_node_filter(SearchFilters(node_labels=['Person', 'Company']))
    assert compiled.sql == ('ARRAY_INCLUDES_ANY(n.labels, @nf_labels)',)
    assert compiled.gql == compiled.sql
    assert compiled.params == {'nf_labels': ['Person', 'Company']}
    assert compiled.param_types == {'nf_labels': STRING_ARRAY}
    assert not compiled.needs_endpoints


def test_node_filter_custom_prefix():
    compiled = build_node_filter(SearchFilters(node_labels=['Person']), prefix='x_')
    assert compiled.sql == ('ARRAY_INCLUDES_ANY(n.labels, @x_labels)',)
    assert set(compiled.params) == {'x_labels'}


def test_node_filter_ignores_edge_only_fields():
    compiled = build_node_filter(
        SearchFilters(edge_types=['KNOWS'], valid_at=[[df(OP.is_null)]], edge_uuids=['r1'])
    )
    assert compiled.sql == ()


def test_node_filter_rejects_unsafe_labels():
    with pytest.raises(Exception):  # noqa: B017 - validate_node_labels raises its own error type
        build_node_filter(SearchFilters(node_labels=['Person) DETACH DELETE (n']))


# --- edge filter: non-date fields --------------------------------------------------------


def test_edge_filter_without_fields_is_empty():
    compiled = build_edge_filter(SearchFilters())
    assert compiled.sql == ()
    assert compiled.gql == ()
    assert compiled.params == {}
    assert not compiled.needs_endpoints


def test_edge_filter_edge_types():
    compiled = build_edge_filter(SearchFilters(edge_types=['KNOWS', 'WORKS_AT']))
    assert compiled.sql == ('e.name IN UNNEST(@ef_edge_types)',)
    assert compiled.gql == compiled.sql
    assert compiled.params == {'ef_edge_types': ['KNOWS', 'WORKS_AT']}
    assert compiled.param_types == {'ef_edge_types': STRING_ARRAY}


def test_edge_filter_sql_and_gql_use_different_uuid_columns():
    compiled = build_edge_filter(
        SearchFilters(
            edge_uuids=['r1'], edge_source_node_uuids=['a1'], edge_target_node_uuids=['a2']
        )
    )
    # In SQL the base table stores the source node uuid in ``uuid`` and the edge uuid in
    # ``edge_uuid``; the property graph exposes them as ``source_node_uuid`` and ``uuid``.
    assert compiled.sql == (
        'e.edge_uuid IN UNNEST(@ef_edge_uuids)',
        'e.uuid IN UNNEST(@ef_source_node_uuids)',
        'e.target_node_uuid IN UNNEST(@ef_target_node_uuids)',
    )
    assert compiled.gql == (
        'e.uuid IN UNNEST(@ef_edge_uuids)',
        'e.source_node_uuid IN UNNEST(@ef_source_node_uuids)',
        'e.target_node_uuid IN UNNEST(@ef_target_node_uuids)',
    )
    assert not compiled.needs_endpoints


def test_edge_filter_node_labels_require_both_endpoints():
    compiled = build_edge_filter(SearchFilters(node_labels=['Person']))
    expected = (
        'ARRAY_INCLUDES_ANY(n.labels, @ef_labels) AND ARRAY_INCLUDES_ANY(m.labels, @ef_labels)'
    )
    assert compiled.sql == (expected,)
    assert compiled.gql == (expected,)
    assert compiled.needs_endpoints


def test_edge_filter_empty_lists_still_constrain():
    # An empty (but not None) uuid list is a real constraint that matches nothing.
    compiled = build_edge_filter(SearchFilters(edge_uuids=[]))
    assert compiled.sql == ('e.edge_uuid IN UNNEST(@ef_edge_uuids)',)
    assert compiled.params == {'ef_edge_uuids': []}


def test_edge_filter_custom_prefix():
    compiled = build_edge_filter(SearchFilters(edge_types=['KNOWS']), prefix='zz_')
    assert compiled.sql == ('e.name IN UNNEST(@zz_edge_types)',)
    assert set(compiled.params) == {'zz_edge_types'}


# --- edge filter: dates ------------------------------------------------------------------


@pytest.mark.parametrize(
    ('operator', 'text'),
    [
        (OP.equals, '='),
        (OP.not_equals, '<>'),
        (OP.greater_than, '>'),
        (OP.less_than, '<'),
        (OP.greater_than_equal, '>='),
        (OP.less_than_equal, '<='),
    ],
)
def test_edge_filter_date_comparison_operators(operator: ComparisonOperator, text: str):
    compiled = build_edge_filter(SearchFilters(valid_at=[[df(operator, D1)]]))
    assert compiled.sql == (f'(((e.valid_at {text} @ef_valid_at_0_0)))',)
    assert compiled.gql == compiled.sql
    assert compiled.params == {'ef_valid_at_0_0': D1}
    assert compiled.param_types == {'ef_valid_at_0_0': param_types.TIMESTAMP}


@pytest.mark.parametrize(
    ('operator', 'text'), [(OP.is_null, 'IS NULL'), (OP.is_not_null, 'IS NOT NULL')]
)
def test_edge_filter_null_operators_bind_no_parameter(operator: ComparisonOperator, text: str):
    compiled = build_edge_filter(SearchFilters(expired_at=[[df(operator)]]))
    assert compiled.sql == (f'(((e.expired_at {text})))',)
    assert compiled.params == {}
    assert compiled.param_types == {}


def test_edge_filter_or_of_and_groups_with_unique_param_names():
    compiled = build_edge_filter(
        SearchFilters(
            valid_at=[
                [df(OP.greater_than, D1), df(OP.less_than, D2)],
                [df(OP.equals, D2)],
                [df(OP.is_null)],
            ]
        )
    )
    assert compiled.sql == (
        '(((e.valid_at > @ef_valid_at_0_0) AND (e.valid_at < @ef_valid_at_0_1)) OR '
        '((e.valid_at = @ef_valid_at_1_0)) OR ((e.valid_at IS NULL)))',
    )
    assert compiled.params == {
        'ef_valid_at_0_0': D1,
        'ef_valid_at_0_1': D2,
        'ef_valid_at_1_0': D2,
    }
    assert set(compiled.param_types) == set(compiled.params)


def test_edge_filter_same_date_in_different_groups_gets_distinct_names():
    compiled = build_edge_filter(SearchFilters(valid_at=[[df(OP.equals, D1)], [df(OP.equals, D1)]]))
    assert len(compiled.params) == 2
    assert len(set(compiled.params)) == 2


def test_edge_filter_multiple_date_fields_do_not_collide():
    compiled = build_edge_filter(
        SearchFilters(
            valid_at=[[df(OP.equals, D1)]],
            invalid_at=[[df(OP.equals, D1)]],
            created_at=[[df(OP.equals, D1)]],
            expired_at=[[df(OP.equals, D1)]],
        )
    )
    assert set(compiled.params) == {
        'ef_valid_at_0_0',
        'ef_invalid_at_0_0',
        'ef_created_at_0_0',
        'ef_expired_at_0_0',
    }
    assert len(compiled.sql) == 4  # one AND-ed predicate per field


@pytest.mark.parametrize(
    'field_value',
    [[], [[]], [[df(OP.equals, D1)], []], [[], [df(OP.is_null)]]],
    ids=['empty-or', 'empty-and', 'empty-and-after-group', 'empty-and-before-group'],
)
def test_edge_filter_empty_date_groups_add_no_constraint(field_value: list[list[DateFilter]]):
    # An empty OR-list or any empty AND-group is vacuously true: no predicate. (Parameters bound
    # by earlier groups before the empty one is seen are left over and unused; harmless.)
    compiled = build_edge_filter(SearchFilters(invalid_at=field_value))
    assert compiled.sql == ()
    assert compiled.gql == ()
    assert set(compiled.params) == set(compiled.param_types)


def test_edge_filter_unset_date_fields_add_nothing():
    compiled = build_edge_filter(SearchFilters(valid_at=None, invalid_at=None))
    assert compiled.sql == ()


def test_edge_filter_combines_fields_in_a_stable_order():
    compiled = build_edge_filter(
        SearchFilters(edge_types=['KNOWS'], node_labels=['Person'], valid_at=[[df(OP.is_not_null)]])
    )
    assert len(compiled.sql) == 3
    assert compiled.sql[0].startswith('e.name IN')
    assert 'n.labels' in compiled.sql[1]
    assert 'e.valid_at' in compiled.sql[2]
    assert compiled.needs_endpoints
    assert set(compiled.params) == set(compiled.param_types)
