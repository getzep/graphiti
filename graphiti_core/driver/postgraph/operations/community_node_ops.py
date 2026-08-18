from __future__ import annotations

import logging

from graphiti_core.driver.operations.community_node_ops import CommunityNodeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.driver.record_parsers import community_node_from_record
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.nodes import CommunityNode

logger = logging.getLogger(__name__)


class PGCommunityNodeOperations(CommunityNodeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        node: CommunityNode,
        tx: Transaction | None = None,
    ) -> None:
        query = """
            INSERT INTO community_nodes (uuid, name, group_id, summary, name_embedding, created_at)
            VALUES ($uuid, $name, $group_id, $summary, $name_embedding, $created_at)
            ON CONFLICT (uuid) DO UPDATE SET
                name = EXCLUDED.name,
                group_id = EXCLUDED.group_id,
                summary = EXCLUDED.summary,
                name_embedding = EXCLUDED.name_embedding,
                created_at = EXCLUDED.created_at
        """
        run = tx.run if tx else executor.execute_query
        await run(
            query,
            uuid=node.uuid,
            name=node.name,
            group_id=node.group_id,
            summary=node.summary,
            name_embedding=_vec(node.name_embedding),
            created_at=node.created_at,
        )
        logger.debug(f'Saved Node to Graph: {node.uuid}')

    async def save_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[CommunityNode],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        for node in nodes:
            await self.save(executor, node, tx=tx)

    async def delete(
        self,
        executor: QueryExecutor,
        node: CommunityNode,
        tx: Transaction | None = None,
    ) -> None:
        run = tx.run if tx else executor.execute_query
        await run('DELETE FROM community_edges WHERE source_node_uuid = $uuid', uuid=node.uuid)
        await run('DELETE FROM community_nodes WHERE uuid = $uuid', uuid=node.uuid)
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
            'DELETE FROM community_edges WHERE source_node_uuid IN (SELECT uuid FROM community_nodes WHERE group_id = $group_id)',
            group_id=group_id,
        )
        await run('DELETE FROM community_nodes WHERE group_id = $group_id', group_id=group_id)

    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        run = tx.run if tx else executor.execute_query
        await run('DELETE FROM community_edges WHERE source_node_uuid = ANY($uuids)', uuids=uuids)
        await run('DELETE FROM community_nodes WHERE uuid = ANY($uuids)', uuids=uuids)

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> CommunityNode:
        records, _, _ = await executor.execute_query(
            """
            SELECT uuid, name, group_id, name_embedding, summary, created_at
            FROM community_nodes WHERE uuid = $uuid
            """,
            uuid=uuid,
        )
        nodes = [community_node_from_record(r) for r in records]
        if not nodes:
            raise NodeNotFoundError(uuid)
        return nodes[0]

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[CommunityNode]:
        records, _, _ = await executor.execute_query(
            """
            SELECT uuid, name, group_id, name_embedding, summary, created_at
            FROM community_nodes WHERE uuid = ANY($uuids)
            """,
            uuids=uuids,
        )
        return [community_node_from_record(r) for r in records]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[CommunityNode]:
        cursor_clause = 'AND uuid < $uuid_cursor' if uuid_cursor else ''
        limit_clause = f'LIMIT {limit}' if limit is not None else ''
        query = f"""
            SELECT uuid, name, group_id, name_embedding, summary, created_at
            FROM community_nodes
            WHERE group_id = ANY($group_ids)
            {cursor_clause}
            ORDER BY uuid DESC
            {limit_clause}
        """
        params: dict = {'group_ids': group_ids}
        if uuid_cursor:
            params['uuid_cursor'] = uuid_cursor
        records, _, _ = await executor.execute_query(query, **params)
        return [community_node_from_record(r) for r in records]

    async def load_name_embedding(
        self,
        executor: QueryExecutor,
        node: CommunityNode,
    ) -> None:
        records, _, _ = await executor.execute_query(
            'SELECT name_embedding FROM community_nodes WHERE uuid = $uuid',
            uuid=node.uuid,
        )
        if not records:
            raise NodeNotFoundError(node.uuid)
        emb = records[0]['name_embedding']
        node.name_embedding = _parse_vec(emb)


def _vec(embedding: list[float] | None) -> str | None:
    if embedding is None:
        return None
    return str(embedding)


def _parse_vec(value) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [float(x) for x in value.strip('[]').split(',')]
    return list(value)
