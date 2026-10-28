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

# Immutable descriptions of Spanner Mutation API writes, applied by
# ``SpannerDriver.apply_mutations`` or buffered by ``SpannerTransaction``.
#
# Always write every column of a row: a partial-column ``insert_or_update`` keeps the omitted
# columns' old values, which differs from Neo4j's ``SET n = $map`` replace-all semantics.

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

MutationKind = Literal['insert_or_update', 'delete']


@dataclass(frozen=True)
class Mutation:
    """One table write: an ``insert_or_update`` of ``values`` (one tuple per row, in
    ``columns`` order) or a ``delete`` of primary ``keys`` (one tuple per row)."""

    kind: MutationKind
    table: str
    columns: tuple[str, ...] = ()
    values: tuple[tuple[Any, ...], ...] = ()
    keys: tuple[tuple[Any, ...], ...] = ()

    @property
    def row_count(self) -> int:
        return len(self.values) if self.kind == 'insert_or_update' else len(self.keys)

    @property
    def rows(self) -> tuple[tuple[Any, ...], ...]:
        return self.values if self.kind == 'insert_or_update' else self.keys

    def with_rows(self, rows: tuple[tuple[Any, ...], ...]) -> Mutation:
        """Copy of this mutation carrying ``rows`` (values or keys, depending on the kind)."""
        if self.kind == 'insert_or_update':
            return replace(self, values=rows)
        return replace(self, keys=rows)


def upsert(table: str, rows: Sequence[Mapping[str, Any]], columns: tuple[str, ...]) -> Mutation:
    """``insert_or_update`` of ``rows`` (dicts keyed by column name) on ``columns``.

    Every row must carry every column; a missing key raises ``ValueError`` instead of writing
    NULL by accident. Values are written as given, so encode rows with ``rows.*_to_row``.
    """
    values: list[tuple[Any, ...]] = []
    for row in rows:
        missing = [column for column in columns if column not in row]
        if missing:
            raise ValueError(f'Row for {table} is missing columns {missing}')
        values.append(tuple(row[column] for column in columns))
    return Mutation(
        kind='insert_or_update', table=table, columns=tuple(columns), values=tuple(values)
    )


def delete(table: str, keys: Sequence[Any]) -> Mutation:
    """Delete rows by primary key. Node tables use ``(uuid,)``; edge tables use
    ``(source_node_uuid, edge_uuid)``. A bare scalar key is wrapped into a 1-tuple."""
    normalized = tuple(tuple(key) if isinstance(key, tuple | list) else (key,) for key in keys)
    return Mutation(kind='delete', table=table, keys=normalized)


def chunk_mutations(mutations: Iterable[Mutation], max_rows: int) -> list[list[Mutation]]:
    """Split mutations into commit batches of at most ``max_rows`` rows each.

    Order is preserved, so nodes written before their edges stay ahead of them. Mutations
    larger than ``max_rows`` are split across batches; empty mutations are dropped.
    """
    if max_rows < 1:
        raise ValueError('max_rows must be at least 1')
    batches: list[list[Mutation]] = []
    current: list[Mutation] = []
    current_rows = 0
    for mutation in mutations:
        rows = mutation.rows
        start = 0
        while start < len(rows):
            room = max_rows - current_rows
            chunk = rows[start : start + room]
            current.append(mutation.with_rows(chunk))
            current_rows += len(chunk)
            start += len(chunk)
            if current_rows >= max_rows:
                batches.append(current)
                current = []
                current_rows = 0
    if current:
        batches.append(current)
    return batches
