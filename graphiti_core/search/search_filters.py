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

from pydantic import BaseModel, Field

from graphiti_core.driver.driver import GraphProvider


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
    date: datetime | None = Field(description='A datetime to filter on')
    comparison_operator: ComparisonOperator = Field(
        description='Comparison operator for date filter'
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
    user_id: str | None = Field(
        default=None,
        description=(
            'User ID for data isolation. '
            'None=no filter (all results), '
            'specific value=shared + that user data, '
            '"__shared__"=shared-only (user_id IS NULL)'
        ),
    )


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
        if provider == GraphProvider.KUZU:
            node_label_filter = 'list_has_all(n.labels, $labels)'
            filter_params['labels'] = filters.node_labels
        else:
            node_labels = '|'.join(filters.node_labels)
            node_label_filter = 'n:' + node_labels
        filter_queries.append(node_label_filter)

    # user_id filtering: entity must have at least one accessible episode
    # (shared episode OR episode matching the requested user_id)
    if filters.user_id is not None:
        if provider == GraphProvider.FALKORDB:
            # FalkorDB does not support EXISTS { MATCH ... } subqueries.
            # Use a post-hoc filter approach: the filter is applied after
            # search results are collected. We return a sentinel that
            # the caller must interpret.  For now, we skip the Cypher-level
            # filter and rely on post-fetch filtering in the service layer.
            #
            # The actual filtering is handled in search_utils.py
            # via _apply_user_id_post_filter for FalkorDB.
            pass
        elif filters.user_id == '__shared__':
            filter_queries.append(
                'EXISTS { MATCH (n)<-[:MENTIONS]-(e:Episodic) WHERE e.user_id IS NULL }'
            )
        else:
            filter_queries.append(
                'EXISTS { MATCH (n)<-[:MENTIONS]-(e:Episodic) '
                'WHERE e.user_id IS NULL OR e.user_id = $user_id }'
            )
            filter_params['user_id'] = filters.user_id

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

    if filters.node_labels is not None:
        if provider == GraphProvider.KUZU:
            node_label_filter = (
                'list_has_all(n.labels, $labels) AND list_has_all(m.labels, $labels)'
            )
            filter_params['labels'] = filters.node_labels
        else:
            node_labels = '|'.join(filters.node_labels)
            node_label_filter = 'n:' + node_labels + ' AND m:' + node_labels
        filter_queries.append(node_label_filter)

    if filters.valid_at is not None:
        valid_at_filter = '('
        for i, or_list in enumerate(filters.valid_at):
            for j, date_filter in enumerate(or_list):
                if date_filter.comparison_operator not in [
                    ComparisonOperator.is_null,
                    ComparisonOperator.is_not_null,
                ]:
                    filter_params['valid_at_' + str(j)] = date_filter.date

            and_filters = [
                date_filter_query_constructor(
                    'e.valid_at', f'$valid_at_{j}', date_filter.comparison_operator
                )
                for j, date_filter in enumerate(or_list)
            ]
            and_filter_query = ''
            for j, and_filter in enumerate(and_filters):
                and_filter_query += and_filter
                if j != len(and_filters) - 1:
                    and_filter_query += ' AND '

            valid_at_filter += and_filter_query

            if i == len(filters.valid_at) - 1:
                valid_at_filter += ')'
            else:
                valid_at_filter += ' OR '

        filter_queries.append(valid_at_filter)

    if filters.invalid_at is not None:
        invalid_at_filter = '('
        for i, or_list in enumerate(filters.invalid_at):
            for j, date_filter in enumerate(or_list):
                if date_filter.comparison_operator not in [
                    ComparisonOperator.is_null,
                    ComparisonOperator.is_not_null,
                ]:
                    filter_params['invalid_at_' + str(j)] = date_filter.date

            and_filters = [
                date_filter_query_constructor(
                    'e.invalid_at', f'$invalid_at_{j}', date_filter.comparison_operator
                )
                for j, date_filter in enumerate(or_list)
            ]
            and_filter_query = ''
            for j, and_filter in enumerate(and_filters):
                and_filter_query += and_filter
                if j != len(and_filters) - 1:
                    and_filter_query += ' AND '

            invalid_at_filter += and_filter_query

            if i == len(filters.invalid_at) - 1:
                invalid_at_filter += ')'
            else:
                invalid_at_filter += ' OR '

        filter_queries.append(invalid_at_filter)

    if filters.created_at is not None:
        created_at_filter = '('
        for i, or_list in enumerate(filters.created_at):
            for j, date_filter in enumerate(or_list):
                if date_filter.comparison_operator not in [
                    ComparisonOperator.is_null,
                    ComparisonOperator.is_not_null,
                ]:
                    filter_params['created_at_' + str(j)] = date_filter.date

            and_filters = [
                date_filter_query_constructor(
                    'e.created_at', f'$created_at_{j}', date_filter.comparison_operator
                )
                for j, date_filter in enumerate(or_list)
            ]
            and_filter_query = ''
            for j, and_filter in enumerate(and_filters):
                and_filter_query += and_filter
                if j != len(and_filters) - 1:
                    and_filter_query += ' AND '

            created_at_filter += and_filter_query

            if i == len(filters.created_at) - 1:
                created_at_filter += ')'
            else:
                created_at_filter += ' OR '

        filter_queries.append(created_at_filter)

    if filters.expired_at is not None:
        expired_at_filter = '('
        for i, or_list in enumerate(filters.expired_at):
            for j, date_filter in enumerate(or_list):
                if date_filter.comparison_operator not in [
                    ComparisonOperator.is_null,
                    ComparisonOperator.is_not_null,
                ]:
                    filter_params['expired_at_' + str(j)] = date_filter.date

            and_filters = [
                date_filter_query_constructor(
                    'e.expired_at', f'$expired_at_{j}', date_filter.comparison_operator
                )
                for j, date_filter in enumerate(or_list)
            ]
            and_filter_query = ''
            for j, and_filter in enumerate(and_filters):
                and_filter_query += and_filter
                if j != len(and_filters) - 1:
                    and_filter_query += ' AND '

            expired_at_filter += and_filter_query

            if i == len(filters.expired_at) - 1:
                expired_at_filter += ')'
            else:
                expired_at_filter += ' OR '

        filter_queries.append(expired_at_filter)

    # user_id filtering: edge must have at least one accessible episode
    if filters.user_id is not None:
        if provider == GraphProvider.FALKORDB:
            # FalkorDB does not support EXISTS { MATCH ... } subqueries.
            # Filtering is handled via post-fetch in search_utils.py.
            pass
        elif filters.user_id == '__shared__':
            filter_queries.append(
                'EXISTS { MATCH (ep:Episodic) WHERE ep.uuid IN e.episodes AND ep.user_id IS NULL }'
            )
        else:
            filter_queries.append(
                'EXISTS { MATCH (ep:Episodic) WHERE ep.uuid IN e.episodes '
                'AND (ep.user_id IS NULL OR ep.user_id = $user_id) }'
            )
            if 'user_id' not in filter_params:
                filter_params['user_id'] = filters.user_id

    return filter_queries, filter_params
