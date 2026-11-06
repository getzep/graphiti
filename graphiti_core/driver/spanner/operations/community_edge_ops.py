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

from graphiti_core.driver.operations.community_edge_ops import CommunityEdgeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.driver.spanner import rows, schema
from graphiti_core.driver.spanner.operations.edge_common import EdgeTable
from graphiti_core.edges import CommunityEdge

_TABLE: EdgeTable[CommunityEdge] = EdgeTable(
    schema.HAS_MEMBER_TABLE, rows.community_edge_to_row, rows.community_edge_from_row
)


class SpannerCommunityEdgeOperations(CommunityEdgeOperations):
    """Spanner implementation of ``CommunityEdgeOperations`` (table ``HasMember``)."""

    async def save(
        self,
        executor: QueryExecutor,
        edge: CommunityEdge,
        tx: Transaction | None = None,
    ) -> None:
        await _TABLE.save(executor, edge, tx)

    async def delete(
        self,
        executor: QueryExecutor,
        edge: CommunityEdge,
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
    ) -> CommunityEdge:
        return await _TABLE.get_by_uuid(executor, uuid)

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[CommunityEdge]:
        return await _TABLE.get_by_uuids(executor, uuids)

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[CommunityEdge]:
        return await _TABLE.get_by_group_ids(executor, group_ids, limit, uuid_cursor)
