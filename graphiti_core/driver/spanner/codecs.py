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

# Conversions between Python values and Spanner values and parameter types.
#
# Write side: ``to_spanner_value`` for mutation values, ``infer_param_types`` plus
# ``encode_params`` for query parameters. Read side: ``from_spanner_value`` and
# ``rows_to_dicts`` turn ``DatetimeWithNanoseconds`` into plain tz-aware ``datetime`` and
# ``JsonObject`` into plain ``dict``/``list`` so client types never leak into Graphiti models.

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from google.api_core.datetime_helpers import DatetimeWithNanoseconds
from google.cloud.spanner_v1 import JsonObject, TypeCode, param_types

from graphiti_core.utils.datetime_utils import ensure_utc

STRING_ARRAY = param_types.Array(param_types.STRING)
INT64_ARRAY = param_types.Array(param_types.INT64)
FLOAT32_ARRAY = param_types.Array(param_types.FLOAT32)
FLOAT64_ARRAY = param_types.Array(param_types.FLOAT64)
TIMESTAMP_ARRAY = param_types.Array(param_types.TIMESTAMP)
BOOL_ARRAY = param_types.Array(param_types.BOOL)

_FLOAT_CODES = (TypeCode.FLOAT32, TypeCode.FLOAT64)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def to_jsonable(value: Any) -> Any:
    """Make a value safe for ``json.dumps``: datetimes become ISO strings (UTC), enums their
    ``.value``, tuples and sets lists. Nested ``JsonObject`` values are unwrapped."""
    if isinstance(value, JsonObject):
        return from_spanner_value(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [to_jsonable(item) for item in value]
    if isinstance(value, datetime):
        utc_value = ensure_utc(value)
        assert utc_value is not None
        return utc_value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    return value


def to_spanner_value(value: Any) -> Any:
    """Convert a Python value into what the Spanner client accepts for a mutation column.

    datetime -> UTC-aware datetime; list of numbers -> list of float (``ARRAY<FLOAT32>``
    rejects Python ints); other lists element-wise; dict -> ``JsonObject`` (nested datetimes
    become ISO strings); Enum -> ``.value``; ``None`` passes through.
    """
    if value is None:
        return None
    if isinstance(value, Enum):
        return to_spanner_value(value.value)
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, JsonObject):
        return value
    if isinstance(value, Mapping):
        return JsonObject(to_jsonable(value))
    if isinstance(value, list | tuple):
        if value and all(_is_number(item) for item in value):
            return [float(item) for item in value]
        return [to_spanner_value(item) for item in value]
    return value


def from_spanner_value(value: Any) -> Any:
    """Convert a value read from Spanner into plain Python.

    ``DatetimeWithNanoseconds`` -> ``datetime`` (UTC-aware, microsecond precision);
    ``JsonObject`` -> ``dict``/``list``/scalar (JSON null -> ``None``); lists element-wise.
    """
    if value is None:
        return None
    if isinstance(value, JsonObject):
        serialized = value.serialize()
        return None if serialized is None else json.loads(serialized)
    if isinstance(value, DatetimeWithNanoseconds):
        return datetime(
            value.year,
            value.month,
            value.day,
            value.hour,
            value.minute,
            value.second,
            value.microsecond,
            tzinfo=value.tzinfo or timezone.utc,
        )
    if isinstance(value, list):
        return [from_spanner_value(item) for item in value]
    return value


def coerce_embedding(vec: list[float] | None) -> list[float] | None:
    """Return the embedding as a list of Python floats (``ARRAY<FLOAT32>`` rejects ints)."""
    if vec is None:
        return None
    return [float(item) for item in vec]


def _infer_array_type(name: str, values: Iterable[Any]) -> Any:
    items = [item for item in values if item is not None]
    if not items:
        # Empty (or all-NULL) lists carry no element type. They are nearly always uuid or
        # group_id lists, so default to ARRAY<STRING>; pass an explicit type otherwise.
        return STRING_ARRAY
    if all(isinstance(item, str) for item in items):
        return STRING_ARRAY
    if all(isinstance(item, bool) for item in items):
        return BOOL_ARRAY
    if all(isinstance(item, int) and not isinstance(item, bool) for item in items):
        return INT64_ARRAY
    if all(_is_number(item) for item in items):
        return FLOAT32_ARRAY
    if all(isinstance(item, datetime) for item in items):
        return TIMESTAMP_ARRAY
    raise ValueError(
        f'Cannot infer a Spanner array type for parameter {name!r} with element types '
        f'{sorted({type(item).__name__ for item in items})}; pass an explicit param type'
    )


def _infer_type(name: str, value: Any) -> Any:
    if value is None:
        raise ValueError(
            f'Cannot infer the Spanner type of parameter {name!r} because its value is None; '
            f'pass an explicit type, e.g. param_types_={{{name!r}: param_types.STRING}}'
        )
    if isinstance(value, Enum):
        return _infer_type(name, value.value)
    if isinstance(value, bool):
        return param_types.BOOL
    if isinstance(value, int):
        return param_types.INT64
    if isinstance(value, float):
        return param_types.FLOAT64
    if isinstance(value, str):
        return param_types.STRING
    if isinstance(value, datetime):
        return param_types.TIMESTAMP
    if isinstance(value, date):
        return param_types.DATE
    if isinstance(value, bytes):
        return param_types.BYTES
    if isinstance(value, JsonObject | Mapping):
        return param_types.JSON
    if isinstance(value, list | tuple):
        return _infer_array_type(name, value)
    raise ValueError(
        f'Cannot infer the Spanner type of parameter {name!r} from {type(value).__name__}; '
        f'pass an explicit param type'
    )


def infer_param_types(
    params: dict[str, Any], explicit: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Map each parameter to a ``google.cloud.spanner_v1.param_types`` type.

    str -> STRING, bool -> BOOL, int -> INT64, float -> FLOAT64, datetime -> TIMESTAMP,
    dict/JsonObject -> JSON, list[str] -> ARRAY<STRING>, list[int] -> ARRAY<INT64>,
    list[float] (ints allowed) -> ARRAY<FLOAT32>, which is what embedding columns use.
    Pass ``explicit`` for anything else, e.g. a query vector against an ``ARRAY<FLOAT64>``
    expression, an all-int vector meant as FLOAT32, or a parameter whose value is ``None``
    (``None`` without an explicit type raises ``ValueError`` naming the parameter).
    """
    explicit = explicit or {}
    return {
        name: explicit[name] if name in explicit else _infer_type(name, value)
        for name, value in params.items()
    }


def encode_param(value: Any, param_type: Any) -> Any:
    """Convert one parameter value to match its Spanner type (ints -> floats for FLOAT arrays,
    datetimes -> UTC, dicts -> JsonObject, enums -> value)."""
    if value is None:
        return None
    if isinstance(value, Enum):
        value = value.value
    code = param_type.code
    if code == TypeCode.ARRAY:
        element_type = param_type.array_element_type
        return [encode_param(item, element_type) for item in value]
    if code in _FLOAT_CODES and _is_number(value):
        return float(value)
    if code == TypeCode.TIMESTAMP and isinstance(value, datetime):
        return ensure_utc(value)
    if code == TypeCode.JSON and not isinstance(value, JsonObject):
        return JsonObject(to_jsonable(value))
    return value


def encode_params(params: dict[str, Any], types: dict[str, Any]) -> dict[str, Any]:
    """Encode every parameter with :func:`encode_param` using its (inferred) type."""
    return {name: encode_param(value, types[name]) for name, value in params.items()}


def rows_to_dicts(result_set: Any) -> list[dict[str, Any]]:
    """Consume a ``StreamedResultSet`` into dicts keyed by column name.

    The client fills ``result_set.fields`` only once the first response has been read, so the
    rows are materialised before the field names are taken. Values go through
    :func:`from_spanner_value`. Duplicate column names collapse, so alias every column.
    """
    rows = list(result_set)
    fields = result_set.fields or []
    names = [field.name for field in fields]
    return [
        {name: from_spanner_value(value) for name, value in zip(names, row, strict=False)}
        for row in rows
    ]
