from __future__ import annotations

import logging
from typing import Any

from graphiti_core.driver.operations.saga_node_ops import SagaNodeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.helpers import parse_db_date
from graphiti_core.nodes import SagaNode

logger = logging.getLogger(__name__)


class PGSagaNodeOperations(SagaNodeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        node: SagaNode,
        tx: Transaction | None = None,
    ) -> None:
        query = """
            INSERT INTO saga_nodes
                (uuid, name, group_id, summary, first_episode_uuid, last_episode_uuid,
                 last_summarized_at, last_summarized_episode_valid_at, created_at)
            VALUES ($uuid, $name, $group_id, $summary, $first_episode_uuid, $last_episode_uuid,
                    $last_summarized_at, $last_summarized_episode_valid_at, $created_at)
            ON CONFLICT (uuid) DO UPDATE SET
                name = EXCLUDED.name,
                group_id = EXCLUDED.group_id,
                summary = EXCLUDED.summary,
                first_episode_uuid = EXCLUDED.first_episode_uuid,
                last_episode_uuid = EXCLUDED.last_episode_uuid,
                last_summarized_at = EXCLUDED.last_summarized_at,
                last_summarized_episode_valid_at = EXCLUDED.last_summarized_episode_valid_at,
                created_at = EXCLUDED.created_at
        """
        run = tx.run if tx else executor.execute_query
        await run(
            query,
            uuid=node.uuid,
            name=node.name,
            group_id=node.group_id,
            summary=node.summary,
            first_episode_uuid=node.first_episode_uuid,
            last_episode_uuid=node.last_episode_uuid,
            last_summarized_at=node.last_summarized_at,
            last_summarized_episode_valid_at=node.last_summarized_episode_valid_at,
            created_at=node.created_at,
        )
        logger.debug(f'Saved Node to Graph: {node.uuid}')

    async def save_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[SagaNode],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        for node in nodes:
            await self.save(executor, node, tx=tx)

    async def delete(
        self,
        executor: QueryExecutor,
        node: SagaNode,
        tx: Transaction | None = None,
    ) -> None:
        run = tx.run if tx else executor.execute_query
        await run('DELETE FROM has_episode_edges WHERE source_node_uuid = $uuid', uuid=node.uuid)
        await run('DELETE FROM saga_nodes WHERE uuid = $uuid', uuid=node.uuid)
        logger.debug(f'Deleted Node: {node.uuid}')

    async def delete_by_group_id(
        self,
        executor: QueryExecutor,
        group_id: str,
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        run = tx.run if tx else executor.execute_query
        await run(
            'DELETE FROM has_episode_edges WHERE source_node_uuid IN (SELECT uuid FROM saga_nodes WHERE group_id = $group_id)',
            group_id=group_id,
        )
        await run('DELETE FROM saga_nodes WHERE group_id = $group_id', group_id=group_id)

    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        run = tx.run if tx else executor.execute_query
        await run('DELETE FROM has_episode_edges WHERE source_node_uuid = ANY($uuids)', uuids=uuids)
        await run('DELETE FROM saga_nodes WHERE uuid = ANY($uuids)', uuids=uuids)

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> SagaNode:
        records, _, _ = await executor.execute_query(
            """
            SELECT uuid, name, group_id, summary, first_episode_uuid, last_episode_uuid,
                   last_summarized_at, last_summarized_episode_valid_at, created_at
            FROM saga_nodes WHERE uuid = $uuid
            """,
            uuid=uuid,
        )
        nodes = [_parse(r) for r in records]
        if not nodes:
            raise NodeNotFoundError(uuid)
        return nodes[0]

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[SagaNode]:
        records, _, _ = await executor.execute_query(
            """
            SELECT uuid, name, group_id, summary, first_episode_uuid, last_episode_uuid,
                   last_summarized_at, last_summarized_episode_valid_at, created_at
            FROM saga_nodes WHERE uuid = ANY($uuids)
            """,
            uuids=uuids,
        )
        return [_parse(r) for r in records]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[SagaNode]:
        cursor_clause = 'AND uuid < $uuid_cursor' if uuid_cursor else ''
        limit_clause = f'LIMIT {limit}' if limit is not None else ''
        query = f"""
            SELECT uuid, name, group_id, summary, first_episode_uuid, last_episode_uuid,
                   last_summarized_at, last_summarized_episode_valid_at, created_at
            FROM saga_nodes
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


def _parse(record: dict) -> SagaNode:
    last_summarized_at = record.get('last_summarized_at')
    last_summarized_episode_valid_at = record.get('last_summarized_episode_valid_at')
    return SagaNode(
        uuid=record['uuid'],
        name=record['name'],
        group_id=record['group_id'],
        created_at=parse_db_date(record['created_at']),
        summary=record.get('summary', '') or '',
        first_episode_uuid=record.get('first_episode_uuid'),
        last_episode_uuid=record.get('last_episode_uuid'),
        last_summarized_at=parse_db_date(last_summarized_at) if last_summarized_at else None,
        last_summarized_episode_valid_at=(
            parse_db_date(last_summarized_episode_valid_at)
            if last_summarized_episode_valid_at
            else None
        ),
    )
