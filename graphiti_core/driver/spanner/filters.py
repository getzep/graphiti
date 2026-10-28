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

# ``SearchFilters`` -> WHERE predicates for Spanner search queries.
#
# Every predicate is produced twice: once for SQL over the base tables and once for GQL over
# the property graph, because the edge tables store the source node uuid in ``uuid`` and the
# edge uuid in ``edge_uuid`` while the graph exposes them as ``source_node_uuid`` and ``uuid``.
# Variable names are fixed, as in Graphiti's Cypher: a node filter targets ``n``; an edge
# filter targets ``e`` with source node ``n`` and target node ``m``.
#
# Values are always bound as parameters with explicit types. Parameter names carry a prefix
# (``nf_`` for node filters, ``ef_`` for edge filters) so the fragments can be combined with
# each other and with the search query's own parameters.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from google.cloud.spanner_v1 import param_types

from graphiti_core.driver.spanner.codecs import STRING_ARRAY
from graphiti_core.helpers import validate_node_labels
from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters

NODE_FILTER_PREFIX = 'nf_'
EDGE_FILTER_PREFIX = 'ef_'

# Edge date fields filtered by SearchFilters; the column has the same name in SQL and GQL.
EDGE_DATE_FIELDS: tuple[str, ...] = ('valid_at', 'invalid_at', 'created_at', 'expired_at')

_OPERATORS: dict[ComparisonOperator, str] = {
    ComparisonOperator.equals: '=',
    ComparisonOperator.not_equals: '<>',
    ComparisonOperator.greater_than: '>',
    ComparisonOperator.less_than: '<',
    ComparisonOperator.greater_than_equal: '>=',
    ComparisonOperator.less_than_equal: '<=',
    ComparisonOperator.is_null: 'IS NULL',
    ComparisonOperator.is_not_null: 'IS NOT NULL',
}
_NULL_OPERATORS = frozenset({ComparisonOperator.is_null, ComparisonOperator.is_not_null})


@dataclass(frozen=True)
class SpannerSearchFilter:
    """``SearchFilters`` compiled into Spanner WHERE predicates.

    ``sql`` predicates reference base-table columns (``e.edge_uuid``, ``e.uuid`` = source node
    uuid); ``gql`` predicates reference property-graph properties (``e.uuid`` = edge uuid,
    ``e.source_node_uuid``). Both lists have the same meaning and are AND-ed by the caller.
    ``needs_endpoints`` is set when an edge predicate references the endpoint nodes ``n`` and
    ``m``: SQL over ``RelatesTo AS e`` must then join ``Entity AS n`` (source) and
    ``Entity AS m`` (target). ``params``/``param_types`` hold every bound value.
    """

    sql: tuple[str, ...] = ()
    gql: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    param_types: dict[str, Any] = field(default_factory=dict)
    needs_endpoints: bool = False


class _FilterBuilder:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.sql: list[str] = []
        self.gql: list[str] = []
        self.params: dict[str, Any] = {}
        self.types: dict[str, Any] = {}
        self.needs_endpoints = False

    def param(self, name: str, value: Any, param_type: Any) -> str:
        key = f'{self.prefix}{name}'
        self.params[key] = value
        self.types[key] = param_type
        return f'@{key}'

    def add(self, sql: str, gql: str | None = None, *, needs_endpoints: bool = False) -> None:
        self.sql.append(sql)
        self.gql.append(sql if gql is None else gql)
        self.needs_endpoints = self.needs_endpoints or needs_endpoints

    def build(self) -> SpannerSearchFilter:
        return SpannerSearchFilter(
            sql=tuple(self.sql),
            gql=tuple(self.gql),
            params=dict(self.params),
            param_types=dict(self.types),
            needs_endpoints=self.needs_endpoints,
        )


def _date_predicate(
    builder: _FilterBuilder,
    column: str,
    name: str,
    groups: Sequence[Sequence[DateFilter]],
) -> str | None:
    """OR of AND-groups of date comparisons, or ``None`` when it constrains nothing.

    A NULL column value never satisfies a comparison (three-valued logic, as in Cypher), so
    only ``IS NULL`` matches edges without the date. An empty OR-list, or any empty AND-group
    (vacuously true), adds no constraint; Neo4j renders invalid Cypher for both.
    """
    or_parts: list[str] = []
    for i, group in enumerate(groups):
        if not group:
            return None
        and_parts: list[str] = []
        for j, date_filter in enumerate(group):
            operator = date_filter.comparison_operator
            sql_operator = _OPERATORS[operator]
            if operator in _NULL_OPERATORS:
                and_parts.append(f'({column} {sql_operator})')
            else:
                value = builder.param(f'{name}_{i}_{j}', date_filter.date, param_types.TIMESTAMP)
                and_parts.append(f'({column} {sql_operator} {value})')
        or_parts.append('(' + ' AND '.join(and_parts) + ')')
    if not or_parts:
        return None
    return '(' + ' OR '.join(or_parts) + ')'


def build_node_filter(
    search_filters: SearchFilters, prefix: str = NODE_FILTER_PREFIX
) -> SpannerSearchFilter:
    """Node predicates on ``n``: ``node_labels`` matches nodes carrying ANY of the labels
    (Neo4j's ``n:A|B``). Edge-only fields and ``property_filters`` are ignored, as in Neo4j."""
    builder = _FilterBuilder(prefix)
    if search_filters.node_labels is not None:
        # Labels are bound as parameters; validation keeps parity with the Cypher drivers.
        validate_node_labels(search_filters.node_labels)
        labels = builder.param('labels', list(search_filters.node_labels), STRING_ARRAY)
        builder.add(f'ARRAY_INCLUDES_ANY(n.labels, {labels})')
    return builder.build()


def build_edge_filter(
    search_filters: SearchFilters, prefix: str = EDGE_FILTER_PREFIX
) -> SpannerSearchFilter:
    """Edge predicates on ``e`` (source ``n``, target ``m``).

    ``edge_types`` matches ``e.name``; ``edge_uuids``, ``edge_source_node_uuids`` and
    ``edge_target_node_uuids`` match the edge and its true source/target (directional, as
    ``startNode``/``endNode`` in Cypher); ``node_labels`` requires BOTH endpoints to carry any
    of the labels; the four date fields are OR-of-AND comparisons. ``property_filters`` is
    ignored, as in Neo4j.
    """
    builder = _FilterBuilder(prefix)

    if search_filters.edge_types is not None:
        edge_types = builder.param('edge_types', list(search_filters.edge_types), STRING_ARRAY)
        builder.add(f'e.name IN UNNEST({edge_types})')

    if search_filters.edge_uuids is not None:
        edge_uuids = builder.param('edge_uuids', list(search_filters.edge_uuids), STRING_ARRAY)
        builder.add(f'e.edge_uuid IN UNNEST({edge_uuids})', f'e.uuid IN UNNEST({edge_uuids})')

    if search_filters.edge_source_node_uuids is not None:
        sources = builder.param(
            'source_node_uuids', list(search_filters.edge_source_node_uuids), STRING_ARRAY
        )
        builder.add(f'e.uuid IN UNNEST({sources})', f'e.source_node_uuid IN UNNEST({sources})')

    if search_filters.edge_target_node_uuids is not None:
        targets = builder.param(
            'target_node_uuids', list(search_filters.edge_target_node_uuids), STRING_ARRAY
        )
        builder.add(f'e.target_node_uuid IN UNNEST({targets})')

    if search_filters.node_labels is not None:
        validate_node_labels(search_filters.node_labels)
        labels = builder.param('labels', list(search_filters.node_labels), STRING_ARRAY)
        builder.add(
            f'ARRAY_INCLUDES_ANY(n.labels, {labels}) AND ARRAY_INCLUDES_ANY(m.labels, {labels})',
            needs_endpoints=True,
        )

    for name in EDGE_DATE_FIELDS:
        groups: list[list[DateFilter]] | None = getattr(search_filters, name)
        if groups is None:
            continue
        predicate = _date_predicate(builder, f'e.{name}', name, groups)
        if predicate is not None:
            builder.add(predicate)

    return builder.build()


__all__ = [
    'EDGE_DATE_FIELDS',
    'EDGE_FILTER_PREFIX',
    'NODE_FILTER_PREFIX',
    'SpannerSearchFilter',
    'build_edge_filter',
    'build_node_filter',
]
