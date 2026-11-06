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

# Shared SQL and mutation helpers for the four Spanner node ops classes.
#
# Writes go through ``SpannerDriver.apply_mutations``: with a transaction they are buffered
# (and nothing is visible, to this or any other read, until the transaction commits); without
# one they are applied immediately in chunks. Table and column names come from schema
# constants; every value is a bound parameter.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from graphiti_core.driver.query_executor import Transaction
from graphiti_core.driver.spanner import mutations
from graphiti_core.driver.spanner.codecs import STRING_ARRAY
from graphiti_core.driver.spanner.operations.base import READ_CHUNK, chunked
from graphiti_core.driver.spanner.schema import EDGE_TABLES, NODE_TABLES, node_select_columns
from graphiti_core.errors import NodeNotFoundError

if TYPE_CHECKING:
    from graphiti_core.driver.spanner.driver import SpannerDriver

# Edge tables whose rows point AT a node of this kind (``target_node_uuid``). They are not
# interleaved under the node, so deleting the node does not remove them.
INCOMING_EDGE_TABLES: dict[str, tuple[str, ...]] = {
    node_table: tuple(
        edge_table.name
        for edge_table in EDGE_TABLES.values()
        if edge_table.target_table == node_table
    )
    for node_table in NODE_TABLES
}

# Every edge table with a node of this kind at either end (for group-wide deletes).
RELATED_EDGE_TABLES: dict[str, tuple[str, ...]] = {
    node_table: tuple(
        edge_table.name
        for edge_table in EDGE_TABLES.values()
        if node_table in (edge_table.source_table, edge_table.target_table)
    )
    for node_table in NODE_TABLES
}


async def save_rows(
    driver: SpannerDriver,
    table: str,
    rows: Sequence[Mapping[str, Any]],
    columns: tuple[str, ...],
    tx: Transaction | None,
    batch_size: int | None = None,
) -> None:
    """Upsert full rows: one mutation per ``batch_size`` rows (one in total when ``None``)."""
    if not rows:
        return
    size = batch_size if batch_size and batch_size > 0 else len(rows)
    muts = [
        mutations.upsert(table, rows[start : start + size], columns)
        for start in range(0, len(rows), size)
    ]
    await driver.apply_mutations(muts, tx)


async def incoming_edge_keys(
    driver: SpannerDriver, edge_table: str, node_uuids: Sequence[str]
) -> list[tuple[str, str]]:
    """Primary keys ``(source uuid, edge uuid)`` of edges pointing at any of ``node_uuids``."""
    keys: list[tuple[str, str]] = []
    for chunk in chunked(list(node_uuids), READ_CHUNK):
        records = await driver.read(
            f'SELECT uuid, edge_uuid FROM {edge_table} WHERE target_node_uuid IN UNNEST(@uuids)',
            {'uuids': list(chunk)},
            {'uuids': STRING_ARRAY},
        )
        keys.extend((r['uuid'], r['edge_uuid']) for r in records)
    return keys


async def delete_nodes(
    driver: SpannerDriver,
    table: str,
    uuids: Sequence[str],
    tx: Transaction | None,
    batch_size: int = 100,
) -> None:
    """Detach-delete nodes: their incoming edges first, then the node rows.

    Outgoing edges are interleaved children of the node row and cascade with it. Without a
    transaction each batch of ``batch_size`` uuids is applied on its own (edges, then nodes).
    With one, everything is buffered; the incoming-edge lookup is a snapshot read, so edges
    created earlier in the same transaction are not seen.
    """
    unique = list(dict.fromkeys(uuids))
    for batch in chunked(unique, batch_size):
        muts: list[mutations.Mutation] = []
        for edge_table in INCOMING_EDGE_TABLES[table]:
            keys = await incoming_edge_keys(driver, edge_table, batch)
            if keys:
                muts.append(mutations.delete(edge_table, keys))
        muts.append(mutations.delete(table, [(uuid,) for uuid in batch]))
        await driver.apply_mutations(muts, tx)


async def delete_group(
    driver: SpannerDriver,
    table: str,
    group_id: str,
    tx: Transaction | None,
    batch_size: int = 100,
) -> None:
    """Delete every node of ``table`` in a group, together with its edges.

    Without a transaction: partitioned DML on the related edge tables, then on the node table
    (not atomic; PDML on the parent alone fails on interleaved children with global indexes).
    Edges are matched by their own ``group_id``, so an edge that crosses groups into this
    group's nodes is left dangling (Graphiti writes an edge in its endpoints' group).
    With a transaction (mutations only): read the group's uuids and use the buffered path.
    """
    if tx is None:
        for edge_table in RELATED_EDGE_TABLES[table]:
            await driver.execute_partitioned_dml(
                f'DELETE FROM {edge_table} WHERE group_id = @g', {'g': group_id}
            )
        await driver.execute_partitioned_dml(
            f'DELETE FROM {table} WHERE group_id = @g', {'g': group_id}
        )
        return
    records = await driver.read(f'SELECT uuid FROM {table} WHERE group_id = @g', {'g': group_id})
    await delete_nodes(driver, table, [r['uuid'] for r in records], tx, batch_size)


async def select_by_uuid(driver: SpannerDriver, table: str, uuid: str) -> dict[str, Any]:
    records = await driver.read(
        f'SELECT {node_select_columns(table)} FROM {table} WHERE uuid = @uuid', {'uuid': uuid}
    )
    if not records:
        raise NodeNotFoundError(uuid)
    return records[0]


async def select_by_uuids(
    driver: SpannerDriver, table: str, uuids: Sequence[str]
) -> list[dict[str, Any]]:
    if not uuids:
        return []
    return await driver.read(
        f'SELECT {node_select_columns(table)} FROM {table} WHERE uuid IN UNNEST(@uuids)',
        {'uuids': list(uuids)},
        {'uuids': STRING_ARRAY},
    )


async def select_by_group_ids(
    driver: SpannerDriver,
    table: str,
    group_ids: Sequence[str],
    limit: int | None,
    uuid_cursor: str | None,
) -> list[dict[str, Any]]:
    """Rows of the groups, highest uuid first; ``uuid_cursor`` keeps only uuids below it."""
    if not group_ids:
        return []
    sql = f'SELECT {node_select_columns(table)} FROM {table} WHERE group_id IN UNNEST(@g)'
    params: dict[str, Any] = {'g': list(group_ids)}
    if uuid_cursor:
        sql += ' AND uuid < @cursor'
        params['cursor'] = uuid_cursor
    sql += ' ORDER BY uuid DESC'
    if limit is not None:
        sql += ' LIMIT @limit'
        params['limit'] = limit
    return await driver.read(sql, params, {'g': STRING_ARRAY})


async def select_embeddings(
    driver: SpannerDriver, table: str, uuids: Sequence[str], batch_size: int = READ_CHUNK
) -> dict[str, list[float] | None]:
    """Map uuid -> ``name_embedding`` for the nodes that exist."""
    found: dict[str, list[float] | None] = {}
    for chunk in chunked(list(dict.fromkeys(uuids)), batch_size):
        records = await driver.read(
            f'SELECT uuid, name_embedding FROM {table} WHERE uuid IN UNNEST(@uuids)',
            {'uuids': list(chunk)},
            {'uuids': STRING_ARRAY},
        )
        for record in records:
            vec = record['name_embedding']
            found[record['uuid']] = [float(x) for x in vec] if vec is not None else None
    return found
