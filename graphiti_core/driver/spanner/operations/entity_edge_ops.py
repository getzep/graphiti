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

from __future__ import annotations

from typing import Any

from graphiti_core.driver.operations.entity_edge_ops import EntityEdgeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.driver.spanner import rows, schema
from graphiti_core.driver.spanner.operations.base import chunked, spanner_driver
from graphiti_core.driver.spanner.operations.edge_common import EdgeTable
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import EdgeNotFoundError

_TABLE: EdgeTable[EntityEdge] = EdgeTable(
    schema.RELATES_TO_TABLE, rows.entity_edge_to_row, rows.entity_edge_from_row
)
_NAME = schema.RELATES_TO_TABLE.name


class SpannerEntityEdgeOperations(EntityEdgeOperations):
    """Spanner implementation of ``EntityEdgeOperations`` (table ``RelatesTo``)."""

    async def save(
        self,
        executor: QueryExecutor,
        edge: EntityEdge,
        tx: Transaction | None = None,
    ) -> None:
        await _TABLE.save(executor, edge, tx)

    async def save_bulk(
        self,
        executor: QueryExecutor,
        edges: list[EntityEdge],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        await _TABLE.save_bulk(executor, edges, tx, batch_size)

    async def delete(
        self,
        executor: QueryExecutor,
        edge: EntityEdge,
        tx: Transaction | None = None,
    ) -> None:
        await _TABLE.delete(executor, edge, tx)

    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None = None,
    ) -> None:
        await _TABLE.delete_by_uuids(executor, uuids, tx)

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> EntityEdge:
        return await _TABLE.get_by_uuid(executor, uuid)

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[EntityEdge]:
        return await _TABLE.get_by_uuids(executor, uuids)

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EntityEdge]:
        return await _TABLE.get_by_group_ids(executor, group_ids, limit, uuid_cursor)

    async def get_between_nodes(
        self,
        executor: QueryExecutor,
        source_node_uuid: str,
        target_node_uuid: str,
    ) -> list[EntityEdge]:
        # All parallel edges: the leading key column plus target_node_uuid.
        return await _TABLE.query(
            executor,
            'uuid = @source AND target_node_uuid = @target',
            {'source': source_node_uuid, 'target': target_node_uuid},
        )

    async def get_by_node_uuid(
        self,
        executor: QueryExecutor,
        node_uuid: str,
    ) -> list[EntityEdge]:
        # Two index-friendly selects (primary key prefix, ByTarget index) instead of an OR.
        # UNION ALL, not UNION DISTINCT: JSON and array columns are not groupable, so
        # self-loops (matched by both selects) are deduplicated by uuid here.
        columns = schema.edge_select_columns(_NAME)
        sql = (
            f'SELECT {columns} FROM {_NAME} WHERE uuid = @node_uuid '
            f'UNION ALL '
            f'SELECT {columns} FROM {_NAME} WHERE target_node_uuid = @node_uuid'
        )
        found = await spanner_driver(executor).read(sql, {'node_uuid': node_uuid})
        edges: dict[str, EntityEdge] = {}
        for row in found:
            if row['uuid'] not in edges:
                edges[row['uuid']] = rows.entity_edge_from_row(row)
        return list(edges.values())

    async def load_embeddings(
        self,
        executor: QueryExecutor,
        edge: EntityEdge,
    ) -> None:
        found = await self._embeddings(executor, [edge.uuid])
        if edge.uuid not in found:
            raise EdgeNotFoundError(edge.uuid)
        edge.fact_embedding = found[edge.uuid]

    async def load_embeddings_bulk(
        self,
        executor: QueryExecutor,
        edges: list[EntityEdge],
        batch_size: int = 100,
    ) -> None:
        for chunk in chunked(edges, batch_size):
            found = await self._embeddings(executor, [edge.uuid for edge in chunk])
            for edge in chunk:
                if edge.uuid in found:
                    edge.fact_embedding = found[edge.uuid]

    @staticmethod
    async def _embeddings(
        executor: QueryExecutor, uuids: list[str]
    ) -> dict[str, list[float] | None]:
        found = await spanner_driver(executor).read(
            f'SELECT edge_uuid AS uuid, fact_embedding FROM {_NAME} '
            f'WHERE edge_uuid IN UNNEST(@uuids)',
            {'uuids': uuids},
        )
        result: dict[str, Any] = {}
        for row in found:
            embedding = row['fact_embedding']
            result[row['uuid']] = None if embedding is None else [float(x) for x in embedding]
        return result
