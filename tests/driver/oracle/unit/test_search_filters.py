# Copyright 2026, Zep Software, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import UTC, datetime

import pytest

from graphiti_core.driver.oracle.errors import InvalidSearchFilter, InvalidSearchQuery
from graphiti_core.driver.oracle.search.filters import build_edge_where, build_node_where
from graphiti_core.driver.oracle.search.text import build_oracle_text_query
from graphiti_core.search.search_filters import (
    ComparisonOperator,
    DateFilter,
    PropertyFilter,
    SearchFilters,
)


def test_property_filter_rejects_untrusted_json_path() -> None:
    filters = SearchFilters(
        property_filters=[
            PropertyFilter(
                property_name="name') OR 1=1 --",
                property_value='x',
                comparison_operator=ComparisonOperator.equals,
            )
        ]
    )

    with pytest.raises(InvalidSearchFilter, match='property_name'):
        build_edge_where(filters, group_ids=['group-a'])


def test_empty_group_list_matches_nothing() -> None:
    where = build_node_where(SearchFilters(), group_ids=[])

    assert where.sql == '1 = 0'
    assert dict(where.binds) == {}


def test_values_are_bound_and_date_groups_keep_or_of_and_shape() -> None:
    start = datetime(2026, 9, 21, 8, tzinfo=UTC)
    end = datetime(2026, 9, 21, 18, tzinfo=UTC)
    untrusted_value = "critical') OR 1=1 --"
    filters = SearchFilters(
        edge_types=['ASSIGNED_TO'],
        edge_uuids=['edge-a'],
        valid_at=[
            [
                DateFilter(date=start, comparison_operator=ComparisonOperator.greater_than_equal),
                DateFilter(date=end, comparison_operator=ComparisonOperator.less_than),
            ],
            [DateFilter(comparison_operator=ComparisonOperator.is_null)],
        ],
        property_filters=[
            PropertyFilter(
                property_name='priority',
                property_value=untrusted_value,
                comparison_operator=ComparisonOperator.equals,
            )
        ],
    )

    where = build_edge_where(filters, group_ids=['group-a'])

    assert (
        '((e.valid_at >= :valid_at_0_0 AND e.valid_at < :valid_at_0_1) '
        'OR (e.valid_at IS NULL))' in where.sql
    )
    assert untrusted_value not in where.sql
    assert where.binds['property_0'] == untrusted_value
    assert where.binds['valid_at_0_0'] == start
    assert where.binds['valid_at_0_1'] == end
    assert where.binds['group_ids_json'] == '["group-a"]'
    assert where.binds['edge_types_json'] == '["ASSIGNED_TO"]'
    assert where.binds['edge_uuids_json'] == '["edge-a"]'


def test_node_labels_are_bound_for_json_membership() -> None:
    where = build_node_where(
        SearchFilters(node_labels=['Asset', 'Site']),
        group_ids=None,
    )

    assert 'JSON_TABLE' in where.sql
    assert 'Asset' not in where.sql
    assert where.binds['node_labels_json'] == '["Asset","Site"]'


@pytest.mark.parametrize(('value', 'expected'), [(True, 'true'), (False, 'false')])
def test_boolean_property_filter_uses_oracle_json_text_representation(
    value: bool, expected: str
) -> None:
    where = build_node_where(
        SearchFilters(
            property_filters=[
                PropertyFilter(
                    property_name='active',
                    property_value=value,
                    comparison_operator=ComparisonOperator.equals,
                )
            ]
        ),
        group_ids=['group-a'],
    )

    assert 'RETURNING VARCHAR2' in where.sql
    assert where.binds['property_0'] == expected


def test_edge_endpoint_filters_are_bound_and_directional() -> None:
    where = build_edge_where(
        SearchFilters(
            edge_source_node_uuids=['source-a', 'source-b'],
            edge_target_node_uuids=['target-a'],
        ),
        group_ids=['group-a'],
    )

    assert 'src.uuid' in where.sql
    assert 'dst.uuid' in where.sql
    assert 'source-a' not in where.sql
    assert 'target-a' not in where.sql
    assert where.binds['edge_source_node_uuids_json'] == '["source-a","source-b"]'
    assert where.binds['edge_target_node_uuids_json'] == '["target-a"]'
    assert where.binds['group_ids_json'] == '["group-a"]'


@pytest.mark.parametrize(
    'filters',
    [
        SearchFilters(edge_source_node_uuids=[]),
        SearchFilters(edge_target_node_uuids=[]),
    ],
)
def test_empty_edge_endpoint_filters_match_nothing(filters: SearchFilters) -> None:
    where = build_edge_where(filters, group_ids=['group-a'])

    assert where.sql == '1 = 0'
    assert dict(where.binds) == {}


def test_oracle_text_expression_is_bounded_and_rejects_empty_input() -> None:
    assert build_oracle_text_query('Alpha / beta!') == '{Alpha} ACCUM {beta}'
    assert build_oracle_text_query('O que conecta Alpha e Beta?') == (
        '{O} ACCUM {que} ACCUM {conecta} ACCUM {Alpha} ACCUM {e} ACCUM {Beta}'
    )
    assert len(build_oracle_text_query('alpha ' * 4000, max_query_length=128)) <= 128

    with pytest.raises(InvalidSearchQuery, match='search query'):
        build_oracle_text_query(' / -- ')
