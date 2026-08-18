from __future__ import annotations

import logging
from typing import Any

from graphiti_core.driver.operations.community_edge_ops import CommunityEdgeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.edges import CommunityEdge
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.helpers import parse_db_date

logger = logging.getLogger(__name__)

_SELECT = """
    SELECT uuid, source_node_uuid, target_node_uuid, group_id, created_at
    FROM community_edges
"""


class PGCommunityEdgeOperations(CommunityEdgeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        edge: CommunityEdge,
        tx: Transaction | None = None,
    ) -> None:
        query = """
            INSERT INTO community_edges (uuid, source_node_uuid, target_node_uuid, group_id, created_at)
            VALUES ($uuid, $source_node_uuid, $target_node_uuid, $group_id, $created_at)
            ON CONFLICT (uuid) DO UPDATE SET
                source_node_uuid = EXCLUDED.source_node_uuid,
                target_node_uuid = EXCLUDED.target_node_uuid,
                group_id = EXCLUDED.group_id,
                created_at = EXCLUDED.created_at
        """
        run = tx.run if tx else executor.execute_query
        await run(
            query,
            uuid=edge.uuid,
            source_node_uuid=edge.source_node_uuid,
            target_node_uuid=edge.target_node_uuid,
            group_id=edge.group_id,
            created_at=edge.created_at,
        )

    async def delete(
        self,
        executor: QueryExecutor,
        edge: CommunityEdge,
        tx: Transaction | None = None,
    ) -> None:
        run = tx.run if tx else executor.execute_query
        await run('DELETE FROM community_edges WHERE uuid = $uuid', uuid=edge.uuid)

    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None = None,
    ) -> None:
        run = tx.run if tx else executor.execute_query
        await run('DELETE FROM community_edges WHERE uuid = ANY($uuids)', uuids=uuids)

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> CommunityEdge:
        records, _, _ = await executor.execute_query(
            _SELECT + ' WHERE uuid = $uuid', uuid=uuid,
        )
        edges = [_parse(r) for r in records]
        if not edges:
            raise EdgeNotFoundError(uuid)
        return edges[0]

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[CommunityEdge]:
        records, _, _ = await executor.execute_query(
            _SELECT + ' WHERE uuid = ANY($uuids)', uuids=uuids,
        )
        return [_parse(r) for r in records]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[CommunityEdge]:
        cursor_clause = 'AND uuid < $uuid_cursor' if uuid_cursor else ''
        limit_clause = f'LIMIT {limit}' if limit is not None else ''
        query = f"""
            {_SELECT}
            WHERE group_id = ANY($group_ids)
            {cursor_clause}
            ORDER BY uuid DESC
            {limit_clause}
        """
        params: dict[str, Any] = {'group_ids': group_ids}
        if uuid_cursor:
            params['uuid_cursor'] = uuid_cursor
        records, _, _ = await executor.execute_query(query, **params)
        return [_parse(r) for r in records]


def _parse(record: dict) -> CommunityEdge:
    return CommunityEdge(
        uuid=record['uuid'],
        group_id=record['group_id'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        created_at=parse_db_date(record['created_at']),
    )
