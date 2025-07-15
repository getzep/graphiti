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


class ComparisonOperator(Enum):
    equals = '='
    not_equals = '<>'
    greater_than = '>'
    less_than = '<'
    greater_than_equal = '>='
    less_than_equal = '<='


class DateFilter(BaseModel):
    date: datetime = Field(description='A datetime to filter on')
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


def node_search_filter_query_constructor(
    filters: SearchFilters,
) -> tuple[str, dict[str, Any]]:
    filter_query: str = ''
    filter_params: dict[str, Any] = {}

    if filters.node_labels is not None:
        node_labels = '|'.join(filters.node_labels)
        node_label_filter = ' AND n:' + node_labels
        filter_query += node_label_filter

    return filter_query, filter_params


def edge_search_filter_query_constructor(
    filters: SearchFilters,
) -> tuple[str, dict[str, Any]]:
    filter_query: str = ''
    filter_params: dict[str, Any] = {}

    if filters.edge_types is not None:
        edge_types = filters.edge_types
        edge_types_filter = '\nAND r.name in $edge_types'
        filter_query += edge_types_filter
        filter_params['edge_types'] = edge_types

    if filters.node_labels is not None:
        node_labels = '|'.join(filters.node_labels)
        node_label_filter = '\nAND n:' + node_labels + ' AND m:' + node_labels
        filter_query += node_label_filter

    if filters.valid_at is not None:
        valid_at_filter = '\nAND ('
        for i, or_list in enumerate(filters.valid_at):
            for j, date_filter in enumerate(or_list):
                filter_params['valid_at_' + str(j)] = date_filter.date

            and_filters = [
                '(r.valid_at ' + date_filter.comparison_operator.value + f' $valid_at_{j})'
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

        filter_query += valid_at_filter

    if filters.invalid_at is not None:
        invalid_at_filter = ' AND ('
        for i, or_list in enumerate(filters.invalid_at):
            for j, date_filter in enumerate(or_list):
                filter_params['invalid_at_' + str(j)] = date_filter.date

            and_filters = [
                '(r.invalid_at ' + date_filter.comparison_operator.value + f' $invalid_at_{j})'
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

        filter_query += invalid_at_filter

    if filters.created_at is not None:
        created_at_filter = ' AND ('
        for i, or_list in enumerate(filters.created_at):
            for j, date_filter in enumerate(or_list):
                filter_params['created_at_' + str(j)] = date_filter.date

            and_filters = [
                '(r.created_at ' + date_filter.comparison_operator.value + f' $created_at_{j})'
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

        filter_query += created_at_filter

    if filters.expired_at is not None:
        expired_at_filter = ' AND ('
        for i, or_list in enumerate(filters.expired_at):
            for j, date_filter in enumerate(or_list):
                filter_params['expired_at_' + str(j)] = date_filter.date

            and_filters = [
                '(r.expired_at ' + date_filter.comparison_operator.value + f' $expired_at_{j})'
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

        filter_query += expired_at_filter

    return filter_query, filter_params
