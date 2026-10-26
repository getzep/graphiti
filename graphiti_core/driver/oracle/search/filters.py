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

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from graphiti_core.search.search_filters import (
    ComparisonOperator,
    DateFilter,
    PropertyFilter,
    SearchFilters,
)

from ..codecs import encode_json, ensure_utc_datetime
from ..errors import InvalidSearchFilter

_PROPERTY_NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]{0,63}$')
_NULL_OPERATORS = {ComparisonOperator.is_null, ComparisonOperator.is_not_null}


@dataclass(frozen=True, slots=True)
class OracleWhere:
    """An immutable SQL predicate paired with bind-only user data."""

    sql: str
    binds: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'binds', MappingProxyType(dict(self.binds)))


def _json_list_membership(column: str, bind_name: str, value_name: str) -> str:
    return f"""EXISTS (
  SELECT 1
  FROM JSON_TABLE(
    {column}, '$[*]' COLUMNS ({value_name} VARCHAR2(255) PATH '$')
  ) actual
  JOIN JSON_TABLE(
    :{bind_name}, '$[*]' COLUMNS ({value_name} VARCHAR2(255) PATH '$')
  ) wanted ON wanted.{value_name} = actual.{value_name}
)"""


def _json_bound_membership(column: str, bind_name: str, value_name: str) -> str:
    return f"""{column} IN (
  SELECT {value_name}
  FROM JSON_TABLE(
    :{bind_name}, '$[*]' COLUMNS ({value_name} VARCHAR2(255) PATH '$')
  )
)"""


def _date_groups(
    *,
    alias: str,
    column: str,
    groups: list[list[DateFilter]] | None,
    binds: dict[str, object],
) -> str | None:
    if groups is None:
        return None
    if not groups:
        return '1 = 0'

    or_clauses: list[str] = []
    for outer_index, and_group in enumerate(groups):
        if not and_group:
            or_clauses.append('1 = 1')
            continue
        and_clauses: list[str] = []
        for inner_index, date_filter in enumerate(and_group):
            operator = date_filter.comparison_operator
            if not isinstance(operator, ComparisonOperator):
                raise InvalidSearchFilter(f'invalid {column} comparison_operator')
            if operator in _NULL_OPERATORS:
                and_clauses.append(f'{alias}.{column} {operator.value}')
                continue
            if date_filter.date is None:
                raise InvalidSearchFilter(f'{column} date is required for {operator.value}')
            bind_name = f'{column}_{outer_index}_{inner_index}'
            binds[bind_name] = ensure_utc_datetime(date_filter.date)
            and_clauses.append(f'{alias}.{column} {operator.value} :{bind_name}')
        or_clauses.append(' AND '.join(and_clauses))
    return '(' + ' OR '.join(f'({clause})' for clause in or_clauses) + ')'


def _property_clause(
    *,
    alias: str,
    property_filter: PropertyFilter,
    index: int,
    binds: dict[str, object],
) -> str:
    property_name = property_filter.property_name
    if not _PROPERTY_NAME.fullmatch(property_name):
        raise InvalidSearchFilter('property_name must match ^[A-Za-z_][A-Za-z0-9_]{0,63}$')

    operator = property_filter.comparison_operator
    if not isinstance(operator, ComparisonOperator):
        raise InvalidSearchFilter('invalid property comparison_operator')
    value = property_filter.property_value
    if isinstance(value, bool):
        returning = 'VARCHAR2(5)'
    else:
        returning = 'NUMBER' if isinstance(value, (int, float)) else 'VARCHAR2(4000)'
    expression = (
        f"JSON_VALUE({alias}.properties, '$.{property_name}' RETURNING {returning} NULL ON ERROR)"
    )
    if operator in _NULL_OPERATORS:
        return f'{expression} {operator.value}'
    if value is None:
        raise InvalidSearchFilter(
            'property_value is required unless comparison_operator is IS NULL or IS NOT NULL'
        )
    bind_name = f'property_{index}'
    binds[bind_name] = str(value).lower() if isinstance(value, bool) else value
    return f'{expression} {operator.value} :{bind_name}'


def _property_clauses(
    alias: str,
    filters: list[PropertyFilter] | None,
    binds: dict[str, object],
) -> list[str]:
    return [
        _property_clause(
            alias=alias,
            property_filter=property_filter,
            index=index,
            binds=binds,
        )
        for index, property_filter in enumerate(filters or [])
    ]


def _finish(clauses: list[str], binds: dict[str, object]) -> OracleWhere:
    if '1 = 0' in clauses:
        return OracleWhere('1 = 0', {})
    return OracleWhere(' AND '.join(f'({clause})' for clause in clauses) or '1 = 1', binds)


def build_node_where(
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    *,
    alias: str = 'n',
) -> OracleWhere:
    binds: dict[str, object] = {}
    clauses: list[str] = []

    if group_ids is not None:
        if not group_ids:
            return OracleWhere('1 = 0', {})
        binds['group_ids_json'] = encode_json(group_ids)
        clauses.append(_json_bound_membership(f'{alias}.group_id', 'group_ids_json', 'value'))

    if search_filter.node_labels is not None:
        if not search_filter.node_labels:
            return OracleWhere('1 = 0', {})
        binds['node_labels_json'] = encode_json(search_filter.node_labels)
        clauses.append(_json_list_membership(f'{alias}.labels', 'node_labels_json', 'label'))

    clauses.extend(_property_clauses(alias, search_filter.property_filters, binds))
    return _finish(clauses, binds)


def build_edge_where(
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    *,
    alias: str = 'e',
    source_alias: str = 'src',
    target_alias: str = 'dst',
) -> OracleWhere:
    binds: dict[str, object] = {}
    clauses: list[str] = []

    if group_ids is not None:
        if not group_ids:
            return OracleWhere('1 = 0', {})
        binds['group_ids_json'] = encode_json(group_ids)
        clauses.append(_json_bound_membership(f'{alias}.group_id', 'group_ids_json', 'value'))

    if search_filter.edge_types is not None:
        if not search_filter.edge_types:
            return OracleWhere('1 = 0', {})
        binds['edge_types_json'] = encode_json(search_filter.edge_types)
        clauses.append(_json_bound_membership(f'{alias}.name', 'edge_types_json', 'value'))

    if search_filter.edge_uuids is not None:
        if not search_filter.edge_uuids:
            return OracleWhere('1 = 0', {})
        binds['edge_uuids_json'] = encode_json(search_filter.edge_uuids)
        clauses.append(_json_bound_membership(f'{alias}.uuid', 'edge_uuids_json', 'value'))

    if search_filter.edge_source_node_uuids is not None:
        if not search_filter.edge_source_node_uuids:
            return OracleWhere('1 = 0', {})
        binds['edge_source_node_uuids_json'] = encode_json(search_filter.edge_source_node_uuids)
        clauses.append(
            _json_bound_membership(
                f'{source_alias}.uuid',
                'edge_source_node_uuids_json',
                'value',
            )
        )

    if search_filter.edge_target_node_uuids is not None:
        if not search_filter.edge_target_node_uuids:
            return OracleWhere('1 = 0', {})
        binds['edge_target_node_uuids_json'] = encode_json(search_filter.edge_target_node_uuids)
        clauses.append(
            _json_bound_membership(
                f'{target_alias}.uuid',
                'edge_target_node_uuids_json',
                'value',
            )
        )

    if search_filter.node_labels is not None:
        if not search_filter.node_labels:
            return OracleWhere('1 = 0', {})
        binds['node_labels_json'] = encode_json(search_filter.node_labels)
        clauses.extend(
            (
                _json_list_membership(f'{source_alias}.labels', 'node_labels_json', 'label'),
                _json_list_membership(f'{target_alias}.labels', 'node_labels_json', 'label'),
            )
        )

    for column, groups in (
        ('valid_at', search_filter.valid_at),
        ('invalid_at', search_filter.invalid_at),
        ('created_at', search_filter.created_at),
        ('expired_at', search_filter.expired_at),
    ):
        clause = _date_groups(
            alias=alias,
            column=column,
            groups=groups,
            binds=binds,
        )
        if clause is not None:
            clauses.append(clause)

    clauses.extend(_property_clauses(alias, search_filter.property_filters, binds))
    return _finish(clauses, binds)
