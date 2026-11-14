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

from graphiti_core.driver.operations.has_episode_edge_ops import HasEpisodeEdgeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.driver.spanner import rows, schema
from graphiti_core.driver.spanner.operations.edge_common import EdgeTable
from graphiti_core.edges import HasEpisodeEdge

_TABLE: EdgeTable[HasEpisodeEdge] = EdgeTable(
    schema.HAS_EPISODE_TABLE, rows.has_episode_edge_to_row, rows.has_episode_edge_from_row
)


class SpannerHasEpisodeEdgeOperations(HasEpisodeEdgeOperations):
    """Spanner implementation of ``HasEpisodeEdgeOperations`` (table ``HasEpisode``)."""

    async def save(
        self,
        executor: QueryExecutor,
        edge: HasEpisodeEdge,
        tx: Transaction | None = None,
    ) -> None:
        await _TABLE.save(executor, edge, tx)

    async def save_bulk(
        self,
        executor: QueryExecutor,
        edges: list[HasEpisodeEdge],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        await _TABLE.save_bulk(executor, edges, tx, batch_size)

    async def delete(
        self,
        executor: QueryExecutor,
        edge: HasEpisodeEdge,
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
    ) -> HasEpisodeEdge:
        return await _TABLE.get_by_uuid(executor, uuid)

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[HasEpisodeEdge]:
        return await _TABLE.get_by_uuids(executor, uuids)

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[HasEpisodeEdge]:
        return await _TABLE.get_by_group_ids(executor, group_ids, limit, uuid_cursor)
