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

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.helpers import validate_node_labels


class ComparisonOperator(Enum):
    equals = '='
    not_equals = '<>'
    greater_than = '>'
    less_than = '<'
    greater_than_equal = '>='
    less_than_equal = '<='
    is_null = 'IS NULL'
    is_not_null = 'IS NOT NULL'


class DateFilter(BaseModel):
    date: datetime | None = Field(default=None, description='A datetime to filter on')
    comparison_operator: ComparisonOperator = Field(
        description='Comparison operator for date filter'
    )


class PropertyFilter(BaseModel):
    property_name: str = Field(description='Property name')
    property_value: str | int | float | None = Field(
        default=None, description='Value you want to match on for the property'
    )
    comparison_operator: ComparisonOperator = Field(
        description='Comparison operator for the property'
    )


class SearchFilters(BaseModel):
    node_labels: list[str] | None = Field(
        default=None, description='List of node labels to filter on'
    )
    edge_types: list[str] | None = Field(
        default=None, description='List of edge types to filter on'
    )
    valid_at: list[list[DateFilter]] | None = Field(default=None)
    invalid_at: list[list[DateFilter]] | None = Field(default=None)
    created_at: list[list[DateFilter]] | None = Field(default=None)
    expired_at: list[list[DateFilter]] | None = Field(default=None)
    edge_uuids: list[str] | None = Field(default=None)
    edge_source_node_uuids: list[str] | None = Field(
        default=None, description='Restrict edge results to these source node UUIDs'
    )
    edge_target_node_uuids: list[str] | None = Field(
        default=None, description='Restrict edge results to these target node UUIDs'
    )
    property_filters: list[PropertyFilter] | None = Field(default=None)

    @field_validator('node_labels')
    @classmethod
    def validate_node_label_filters(cls, value: list[str] | None) -> list[str] | None:
        validate_node_labels(value)
        return value


def cypher_to_opensearch_operator(op: ComparisonOperator) -> str:
    mapping = {
        ComparisonOperator.greater_than: 'gt',
        ComparisonOperator.less_than: 'lt',
        ComparisonOperator.greater_than_equal: 'gte',
        ComparisonOperator.less_than_equal: 'lte',
    }
    return mapping.get(op, op.value)


def node_search_filter_query_constructor(
    filters: SearchFilters,
    provider: GraphProvider,
) -> tuple[list[str], dict[str, Any]]:
    filter_queries: list[str] = []
    filter_params: dict[str, Any] = {}

    if filters.node_labels is not None:
        # Defense-in-depth for model_construct()/other validation bypasses.
        validate_node_labels(filters.node_labels)
        node_labels = '|'.join(filters.node_labels)
        node_label_filter = 'n:' + node_labels
        filter_queries.append(node_label_filter)

    return filter_queries, filter_params


def date_filter_query_constructor(
    value_name: str, param_name: str, operator: ComparisonOperator
) -> str:
    query = '(' + value_name + ' '

    if operator == ComparisonOperator.is_null or operator == ComparisonOperator.is_not_null:
        query += operator.value + ')'
    else:
        query += operator.value + ' ' + param_name + ')'

    return query


def edge_search_filter_query_constructor(
    filters: SearchFilters,
    provider: GraphProvider,
) -> tuple[list[str], dict[str, Any]]:
    filter_queries: list[str] = []
    filter_params: dict[str, Any] = {}

    if filters.edge_types is not None:
        edge_types = filters.edge_types
        filter_queries.append('e.name in $edge_types')
        filter_params['edge_types'] = edge_types

    if filters.edge_uuids is not None:
        filter_queries.append('e.uuid in $edge_uuids')
        filter_params['edge_uuids'] = filters.edge_uuids

    # Edge endpoint filters: every edge query binds n as the source and m as
    # the target endpoint. Backends that can push these down pre-rank an
    # endpoint-scoped candidate set instead of scanning the graph.
    if filters.edge_source_node_uuids is not None:
        filter_queries.append('n.uuid IN $edge_source_node_uuids')
        filter_params['edge_source_node_uuids'] = filters.edge_source_node_uuids

    if filters.edge_target_node_uuids is not None:
        filter_queries.append('m.uuid IN $edge_target_node_uuids')
        filter_params['edge_target_node_uuids'] = filters.edge_target_node_uuids

    if filters.node_labels is not None:
        # Defense-in-depth for model_construct()/other validation bypasses.
        validate_node_labels(filters.node_labels)
        node_labels = '|'.join(filters.node_labels)
        node_label_filter = 'n:' + node_labels + ' AND m:' + node_labels
        filter_queries.append(node_label_filter)

    for field_name, date_filters in (
        ('valid_at', filters.valid_at),
        ('invalid_at', filters.invalid_at),
        ('created_at', filters.created_at),
        ('expired_at', filters.expired_at),
    ):
        if date_filters is None:
            continue

        non_empty_groups = [or_list for or_list in date_filters if or_list]
        if not non_empty_groups:
            continue

        group_queries: list[str] = []
        for or_list in non_empty_groups:
            for j, date_filter in enumerate(or_list):
                if date_filter.comparison_operator not in [
                    ComparisonOperator.is_null,
                    ComparisonOperator.is_not_null,
                ]:
                    filter_params[f'{field_name}_{j}'] = date_filter.date

            and_filters = [
                date_filter_query_constructor(
                    f'e.{field_name}', f'${field_name}_{j}', date_filter.comparison_operator
                )
                for j, date_filter in enumerate(or_list)
            ]
            group_queries.append(' AND '.join(and_filters))

        filter_queries.append('(' + ' OR '.join(group_queries) + ')')

    return filter_queries, filter_params
