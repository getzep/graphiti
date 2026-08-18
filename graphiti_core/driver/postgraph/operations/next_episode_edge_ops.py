from __future__ import annotations

import logging
from typing import Any

from graphiti_core.driver.operations.next_episode_edge_ops import NextEpisodeEdgeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.edges import NextEpisodeEdge
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.helpers import parse_db_date

logger = logging.getLogger(__name__)

_SELECT = """
    SELECT uuid, source_node_uuid, target_node_uuid, group_id, created_at
    FROM next_episode_edges
"""


class PGNextEpisodeEdgeOperations(NextEpisodeEdgeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        edge: NextEpisodeEdge,
        tx: Transaction | None = None,
    ) -> None:
        query = """
            INSERT INTO next_episode_edges (uuid, source_node_uuid, target_node_uuid, group_id, created_at)
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

    async def save_bulk(
        self,
        executor: QueryExecutor,
        edges: list[NextEpisodeEdge],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        for edge in edges:
            await self.save(executor, edge, tx=tx)

    async def delete(
        self,
        executor: QueryExecutor,
        edge: NextEpisodeEdge,
        tx: Transaction | None = None,
    ) -> None:
        run = tx.run if tx else executor.execute_query
        await run('DELETE FROM next_episode_edges WHERE uuid = $uuid', uuid=edge.uuid)

    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None = None,
    ) -> None:
        run = tx.run if tx else executor.execute_query
        await run('DELETE FROM next_episode_edges WHERE uuid = ANY($uuids)', uuids=uuids)

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> NextEpisodeEdge:
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
    ) -> list[NextEpisodeEdge]:
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
    ) -> list[NextEpisodeEdge]:
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


def _parse(record: dict) -> NextEpisodeEdge:
    return NextEpisodeEdge(
        uuid=record['uuid'],
        group_id=record['group_id'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        created_at=parse_db_date(record['created_at']),
    )
