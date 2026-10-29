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

# Shared CRUD for the five Spanner edge tables. Every edge table is interleaved in its source
# node table with PRIMARY KEY (uuid = source node uuid, edge_uuid = Graphiti edge uuid), so an
# edge is addressed by (source_node_uuid, uuid) for writes and by edge_uuid (unique index) for
# reads. Reads alias edge_uuid AS uuid and uuid AS source_node_uuid (schema.edge_select_columns).

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Generic, TypeVar

from google.api_core.exceptions import FailedPrecondition, GoogleAPICallError, NotFound

from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.driver.spanner import mutations
from graphiti_core.driver.spanner.errors import SpannerMissingEndpointError
from graphiti_core.driver.spanner.operations.base import (
    READ_CHUNK,
    chunked,
    spanner_driver,
    spanner_transaction,
)
from graphiti_core.driver.spanner.schema import Table, edge_select_columns
from graphiti_core.edges import Edge
from graphiti_core.errors import EdgeNotFoundError

E = TypeVar('E', bound=Edge)


def is_missing_parent_error(error: BaseException) -> bool:
    """True for the client error raised when an interleaved child's parent row is absent."""
    if not isinstance(error, NotFound | FailedPrecondition):
        return False
    message = str(error).lower()
    return 'parent' in message or 'not found' in message


class EdgeTable(Generic[E]):
    """CRUD for one edge table, parameterised by its row encoder and decoder."""

    def __init__(
        self,
        table: Table,
        to_row: Callable[[Any], dict[str, Any]],
        from_row: Callable[[Mapping[str, Any]], Any],
    ):
        self.table = table
        self.to_row = to_row
        self.from_row = from_row
        self.select = edge_select_columns(table.name)

    # --- writes ---

    async def save(self, executor: QueryExecutor, edge: E, tx: Transaction | None) -> None:
        driver = spanner_driver(executor)
        mutation = mutations.upsert(
            self.table.name, [self.to_row(edge)], self.table.writable_columns
        )
        try:
            await driver.apply_mutations([mutation], spanner_transaction(tx))
        except GoogleAPICallError as error:
            if tx is None and is_missing_parent_error(error):
                raise self.missing_endpoint(edge.uuid, edge.source_node_uuid, error) from error
            raise

    async def save_bulk(
        self, executor: QueryExecutor, edges: Sequence[E], tx: Transaction | None, batch_size: int
    ) -> None:
        if not edges:
            return
        driver = spanner_driver(executor)
        spanner_tx = spanner_transaction(tx)
        for chunk in chunked(edges, batch_size):
            mutation = mutations.upsert(
                self.table.name, [self.to_row(edge) for edge in chunk], self.table.writable_columns
            )
            try:
                await driver.apply_mutations([mutation], spanner_tx)
            except GoogleAPICallError as error:
                if tx is None and is_missing_parent_error(error):
                    raise self.missing_endpoint(None, None, error) from error
                raise

    async def delete(self, executor: QueryExecutor, edge: E, tx: Transaction | None) -> None:
        driver = spanner_driver(executor)
        mutation = mutations.delete(self.table.name, [(edge.source_node_uuid, edge.uuid)])
        await driver.apply_mutations([mutation], spanner_transaction(tx))

    async def delete_by_uuids(
        self, executor: QueryExecutor, uuids: list[str], tx: Transaction | None
    ) -> None:
        if not uuids:
            return
        driver = spanner_driver(executor)
        keys: list[tuple[str, str]] = []
        for chunk in chunked(uuids, READ_CHUNK):
            rows = await driver.read(
                f'SELECT uuid, edge_uuid FROM {self.table.name} WHERE edge_uuid IN UNNEST(@uuids)',
                {'uuids': list(chunk)},
            )
            keys.extend((row['uuid'], row['edge_uuid']) for row in rows)
        if keys:
            await driver.apply_mutations(
                [mutations.delete(self.table.name, keys)], spanner_transaction(tx)
            )

    def missing_endpoint(
        self, edge_uuid: str | None, source_uuid: str | None, error: BaseException
    ) -> SpannerMissingEndpointError:
        detail = f' {edge_uuid}' if edge_uuid else ''
        source = f' (source node {source_uuid} does not exist)' if source_uuid else ''
        return SpannerMissingEndpointError(
            f'Cannot save {self.table.name} edge{detail}{source}: the source node row is '
            f'missing from {self.table.source_table}',
            table=self.table.name,
            edge_uuid=edge_uuid,
            node_uuid=source_uuid,
        )

    # --- reads ---

    async def query(
        self, executor: QueryExecutor, where: str, params: dict[str, Any], tail: str = ''
    ) -> list[E]:
        driver = spanner_driver(executor)
        sql = f'SELECT {self.select} FROM {self.table.name} WHERE {where} {tail}'.strip()
        rows = await driver.read(sql, params)
        return [self.from_row(row) for row in rows]

    async def get_by_uuid(self, executor: QueryExecutor, uuid: str) -> E:
        edges = await self.query(executor, 'edge_uuid = @uuid', {'uuid': uuid})
        if not edges:
            raise EdgeNotFoundError(uuid)
        return edges[0]

    async def get_by_uuids(self, executor: QueryExecutor, uuids: list[str]) -> list[E]:
        edges: list[E] = []
        for chunk in chunked(uuids, READ_CHUNK):
            edges.extend(
                await self.query(executor, 'edge_uuid IN UNNEST(@uuids)', {'uuids': list(chunk)})
            )
        return edges

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None,
        uuid_cursor: str | None,
    ) -> list[E]:
        where = 'group_id IN UNNEST(@group_ids)'
        params: dict[str, Any] = {'group_ids': list(group_ids)}
        if uuid_cursor:
            where += ' AND edge_uuid < @cursor'
            params['cursor'] = uuid_cursor
        tail = 'ORDER BY edge_uuid DESC'
        if limit is not None:
            tail += ' LIMIT @limit'
            params['limit'] = limit
        return await self.query(executor, where, params, tail)


__all__ = ['EdgeTable', 'is_missing_parent_error']
