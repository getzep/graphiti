"""
Copyright 2026, Zep Software, Inc.

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

from __future__ import annotations

import operator
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields
from datetime import date, datetime, time, timedelta
from types import MappingProxyType
from typing import Any, TypeAlias

from relate import Duration as _NativeDuration
from relate import Node as _NativeNode
from relate import Path as _NativePath
from relate import Relationship as _NativeRelationship
from relate import Temporal as _NativeTemporal
from relate import Vector as _NativeVector

from ._errors import ResultNormalizationError


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    native_type: str | None


@dataclass(frozen=True, slots=True)
class UpdateStatistics:
    nodes_created: int = 0
    nodes_deleted: int = 0
    relationships_created: int = 0
    relationships_deleted: int = 0
    properties_set: int = 0
    labels_added: int = 0
    labels_removed: int = 0
    indexes_added: int = 0
    indexes_removed: int = 0
    constraints_added: int = 0
    constraints_removed: int = 0


@dataclass(frozen=True, slots=True)
class Temporal:
    text: str


@dataclass(frozen=True, slots=True)
class Vector:
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        try:
            normalized = tuple(float(value) for value in self.values)
        except (TypeError, ValueError, OverflowError) as error:
            raise ResultNormalizationError('vector', type(error).__name__) from error
        object.__setattr__(self, 'values', normalized)


@dataclass(frozen=True, slots=True)
class Node:
    id: str | int
    labels: tuple[str, ...]
    properties: Mapping[str, Value]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'labels', tuple(self.labels))
        object.__setattr__(
            self, 'properties', _normalize_mapping(self.properties, 'node properties')
        )


@dataclass(frozen=True, slots=True)
class Relationship:
    id: str | int
    type: str
    start_node_id: str | int
    end_node_id: str | int
    properties: Mapping[str, Value]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'properties', _normalize_mapping(self.properties, 'relationship properties')
        )


@dataclass(frozen=True, slots=True)
class Path:
    nodes: tuple[Node, ...]
    relationships: tuple[Relationship, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'nodes', tuple(self.nodes))
        object.__setattr__(self, 'relationships', tuple(self.relationships))


Value: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | bytes
    | datetime
    | date
    | time
    | timedelta
    | Temporal
    | Vector
    | Node
    | Relationship
    | Path
    | tuple['Value', ...]
    | Mapping[str, 'Value']
)


@dataclass(frozen=True, slots=True, init=False)
class Record(Mapping[str, Value]):
    _values: Mapping[str, Value] = field(repr=False)

    def __init__(self, values: Mapping[str, Value]) -> None:
        object.__setattr__(self, '_values', _normalize_mapping(values, 'record'))

    def __getitem__(self, key: str) -> Value:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


@dataclass(frozen=True, slots=True)
class QueryResult:
    columns: tuple[Column, ...]
    rows: tuple[Record, ...]
    statistics: UpdateStatistics
    bookmark: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'columns', tuple(self.columns))
        object.__setattr__(self, 'rows', tuple(self.rows))


def _error(
    location: str, value: object, cause: BaseException | None = None
) -> ResultNormalizationError:
    error = ResultNormalizationError(location, type(value))
    if cause is not None:
        error.__cause__ = cause
    return error


def _graph_id(value: object, location: str) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise _error(location, value)
    return value


def _normalize_mapping(values: Mapping[str, Any], location: str) -> Mapping[str, Value]:
    if not isinstance(values, Mapping):
        raise _error(location, values)
    normalized: dict[str, Value] = {}
    for index, (key, value) in enumerate(values.items()):
        if not isinstance(key, str):
            raise _error(f'{location}, key {index}', key)
        normalized[key] = _normalize_value(value, f'{location}, value {index}')
    return MappingProxyType(normalized)


def _normalize_native_graph(value: object, location: str) -> Value | None:
    cls = type(value)
    if isinstance(value, _NativeNode) or cls.__name__ in {'Node', 'FakeNode'}:
        return Node(
            _graph_id(value.id, location),
            tuple(value.labels),
            value.properties,
        )
    if isinstance(value, _NativeRelationship) or cls.__name__ in {
        'Relationship',
        'FakeRelationship',
    }:
        return Relationship(
            _graph_id(value.id, location),
            str(value.type),
            _graph_id(
                value.start_node
                if isinstance(value.start_node, (str, int))
                else value.start_node.id,
                location,
            ),
            _graph_id(
                value.end_node if isinstance(value.end_node, (str, int)) else value.end_node.id,
                location,
            ),
            value.properties,
        )
    if isinstance(value, _NativePath) or cls.__name__ in {'Path', 'FakePath'}:
        nodes = tuple(
            _normalize_native_graph(node, f'{location}, node {index}')
            for index, node in enumerate(value.nodes)
        )
        relationships = tuple(
            _normalize_native_graph(relationship, f'{location}, relationship {index}')
            for index, relationship in enumerate(value.relationships)
        )
        if not all(isinstance(node, Node) for node in nodes) or not all(
            isinstance(edge, Relationship) for edge in relationships
        ):
            raise _error(location, value)
        return Path(nodes, relationships)
    return None


def _normalize_temporal(value: object, location: str) -> Value:
    try:
        if hasattr(value, 'to_native'):
            native = value.to_native()
        elif hasattr(value, 'to_timedelta'):
            native = value.to_timedelta()
        else:
            raise TypeError('native temporal conversion is unavailable')
        if native is None:
            text = getattr(value, 'text', None)
            return Temporal(str(value) if text is None else str(text))
        if isinstance(native, (datetime, date, time, timedelta)):
            return native
    except ResultNormalizationError:
        raise
    except Exception as error:
        raise _error(location, value, error) from error
    raise _error(location, value)


def _normalize_value(value: object, location: str) -> Value:
    if isinstance(value, (Temporal, Vector, Node, Relationship, Path)):
        return value
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, (datetime, date, time, timedelta)):
        return value
    if isinstance(value, _NativeTemporal | _NativeDuration) or type(value).__name__ in {
        'Temporal',
        'Duration',
        'FakeTemporal',
    }:
        return _normalize_temporal(value, location)
    if isinstance(value, _NativeVector) or type(value).__name__ in {'Vector', 'FakeVector'}:
        return Vector(tuple(value.values))

    graph = _normalize_native_graph(value, location)
    if graph is not None:
        return graph
    if isinstance(value, (list, tuple)):
        return tuple(
            _normalize_value(item, f'{location}, item {index}') for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        normalized: dict[str, Value] = {}
        for index, (key, item) in enumerate(value.items()):
            if not isinstance(key, str):
                raise _error(f'{location}, key {index}', key)
            normalized[key] = _normalize_value(item, f'{location}, value {index}')
        return MappingProxyType(normalized)
    raise _error(location, value)


def _statistics(native: Mapping[str, object]) -> UpdateStatistics:
    values: dict[str, int] = {}
    for descriptor in fields(UpdateStatistics):
        name = descriptor.name
        raw = native.get(name, 0)
        try:
            if isinstance(raw, bool):
                raise TypeError
            values[name] = operator.index(raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise ResultNormalizationError(f'statistics.{name}', type(raw)) from error
    return UpdateStatistics(**values)


def materialize_result(cursor: Any, *, bookmark: str | None = None) -> QueryResult:
    """Eagerly detach a native cursor; intended to run only on its owner worker."""

    try:
        description = cursor.description or ()
        columns = tuple(
            Column(str(item[0]), getattr(item[1], 'name', None)) for item in description
        )
        native_rows = cursor.fetchall()
        rows: list[Record] = []
        for row_index, native_row in enumerate(native_rows):
            values = tuple(
                _normalize_value(native_row[column.name], f'row {row_index}, column {column_index}')
                for column_index, column in enumerate(columns)
            )
            rows.append(Record(dict(zip((column.name for column in columns), values, strict=True))))
        statistics = _statistics(cursor.stats)
        # Consumption validates and releases statement metadata. A statement summary's
        # bookmark is deliberately not promoted to a transaction commit bookmark.
        cursor.consume()
        return QueryResult(columns, tuple(rows), statistics, bookmark)
    except ResultNormalizationError:
        raise
    except Exception:
        raise
    finally:
        cursor.close()
