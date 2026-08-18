from __future__ import annotations

import logging
from datetime import datetime

from graphiti_core.driver.operations.episode_node_ops import EpisodeNodeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.helpers import parse_db_date
from graphiti_core.nodes import EpisodeType, EpisodicNode

logger = logging.getLogger(__name__)


class PGEpisodeNodeOperations(EpisodeNodeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        node: EpisodicNode,
        tx: Transaction | None = None,
    ) -> None:
        query = """
            INSERT INTO episodic_nodes
                (uuid, name, group_id, source, source_description, content, valid_at, entity_edges, created_at)
            VALUES ($uuid, $name, $group_id, $source, $source_description, $content, $valid_at, $entity_edges, $created_at)
            ON CONFLICT (uuid) DO UPDATE SET
                name = EXCLUDED.name,
                group_id = EXCLUDED.group_id,
                source = EXCLUDED.source,
                source_description = EXCLUDED.source_description,
                content = EXCLUDED.content,
                valid_at = EXCLUDED.valid_at,
                entity_edges = EXCLUDED.entity_edges,
                created_at = EXCLUDED.created_at
        """
        run = tx.run if tx else executor.execute_query
        await run(
            query,
            uuid=node.uuid,
            name=node.name,
            group_id=node.group_id,
            source=node.source.value,
            source_description=node.source_description,
            content=node.content,
            valid_at=node.valid_at,
            entity_edges=node.entity_edges,
            created_at=node.created_at,
        )
        logger.debug(f'Saved Node to Graph: {node.uuid}')

    async def save_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[EpisodicNode],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        for node in nodes:
            await self.save(executor, node, tx=tx)

    async def delete(
        self,
        executor: QueryExecutor,
        node: EpisodicNode,
        tx: Transaction | None = None,
    ) -> None:
        run = tx.run if tx else executor.execute_query
        await run('DELETE FROM episodic_edges WHERE source_node_uuid = $uuid', uuid=node.uuid)
        await run('DELETE FROM has_episode_edges WHERE target_node_uuid = $uuid', uuid=node.uuid)
        await run(
            'DELETE FROM next_episode_edges WHERE source_node_uuid = $uuid OR target_node_uuid = $uuid',
            uuid=node.uuid,
        )
        await run('DELETE FROM episodic_nodes WHERE uuid = $uuid', uuid=node.uuid)
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
            'DELETE FROM episodic_edges WHERE source_node_uuid IN (SELECT uuid FROM episodic_nodes WHERE group_id = $group_id)',
            group_id=group_id,
        )
        await run(
            'DELETE FROM has_episode_edges WHERE target_node_uuid IN (SELECT uuid FROM episodic_nodes WHERE group_id = $group_id)',
            group_id=group_id,
        )
        await run(
            'DELETE FROM next_episode_edges WHERE source_node_uuid IN (SELECT uuid FROM episodic_nodes WHERE group_id = $group_id) OR target_node_uuid IN (SELECT uuid FROM episodic_nodes WHERE group_id = $group_id)',
            group_id=group_id,
        )
        await run('DELETE FROM episodic_nodes WHERE group_id = $group_id', group_id=group_id)

    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        run = tx.run if tx else executor.execute_query
        await run('DELETE FROM episodic_edges WHERE source_node_uuid = ANY($uuids)', uuids=uuids)
        await run('DELETE FROM has_episode_edges WHERE target_node_uuid = ANY($uuids)', uuids=uuids)
        await run(
            'DELETE FROM next_episode_edges WHERE source_node_uuid = ANY($uuids) OR target_node_uuid = ANY($uuids)',
            uuids=uuids,
        )
        await run('DELETE FROM episodic_nodes WHERE uuid = ANY($uuids)', uuids=uuids)

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> EpisodicNode:
        records, _, _ = await executor.execute_query(
            """
            SELECT uuid, name, group_id, source, source_description, content,
                   valid_at, entity_edges, created_at
            FROM episodic_nodes WHERE uuid = $uuid
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
    ) -> list[EpisodicNode]:
        records, _, _ = await executor.execute_query(
            """
            SELECT uuid, name, group_id, source, source_description, content,
                   valid_at, entity_edges, created_at
            FROM episodic_nodes WHERE uuid = ANY($uuids)
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
    ) -> list[EpisodicNode]:
        cursor_clause = 'AND uuid < $uuid_cursor' if uuid_cursor else ''
        limit_clause = f'LIMIT {limit}' if limit is not None else ''
        query = f"""
            SELECT uuid, name, group_id, source, source_description, content,
                   valid_at, entity_edges, created_at
            FROM episodic_nodes
            WHERE group_id = ANY($group_ids)
            {cursor_clause}
            ORDER BY uuid DESC
            {limit_clause}
        """
        params: dict = {'group_ids': group_ids}
        if uuid_cursor:
            params['uuid_cursor'] = uuid_cursor
        records, _, _ = await executor.execute_query(query, **params)
        return [_parse(r) for r in records]

    async def get_by_entity_node_uuid(
        self,
        executor: QueryExecutor,
        entity_node_uuid: str,
    ) -> list[EpisodicNode]:
        records, _, _ = await executor.execute_query(
            """
            SELECT DISTINCT e.uuid, e.name, e.group_id, e.source, e.source_description,
                   e.content, e.valid_at, e.entity_edges, e.created_at
            FROM episodic_nodes e
            JOIN episodic_edges ee ON ee.source_node_uuid = e.uuid
            WHERE ee.target_node_uuid = $entity_node_uuid
            """,
            entity_node_uuid=entity_node_uuid,
        )
        return [_parse(r) for r in records]

    async def retrieve_episodes(
        self,
        executor: QueryExecutor,
        reference_time: datetime,
        last_n: int = 3,
        group_ids: list[str] | None = None,
        source: str | None = None,
        saga: str | None = None,
    ) -> list[EpisodicNode]:
        conditions = ['valid_at <= $reference_time']
        params: dict = {'reference_time': reference_time}

        if group_ids:
            conditions.append('group_id = ANY($group_ids)')
            params['group_ids'] = group_ids
        if source:
            conditions.append('source = $source')
            params['source'] = source

        where = ' AND '.join(conditions)

        if saga:
            query = f"""
                SELECT e.uuid, e.name, e.group_id, e.source, e.source_description,
                       e.content, e.valid_at, e.entity_edges, e.created_at
                FROM episodic_nodes e
                JOIN has_episode_edges he ON he.target_node_uuid = e.uuid
                WHERE he.source_node_uuid = $saga AND {where}
                ORDER BY e.valid_at DESC
                LIMIT {last_n}
            """
            params['saga'] = saga
        else:
            query = f"""
                SELECT uuid, name, group_id, source, source_description, content,
                       valid_at, entity_edges, created_at
                FROM episodic_nodes
                WHERE {where}
                ORDER BY valid_at DESC
                LIMIT {last_n}
            """

        records, _, _ = await executor.execute_query(query, **params)
        return [_parse(r) for r in records]


def _parse(record: dict) -> EpisodicNode:
    created_at = parse_db_date(record['created_at'])
    valid_at = parse_db_date(record['valid_at'])

    if created_at is None:
        raise ValueError(f'created_at cannot be None for episode {record.get("uuid", "unknown")}')
    if valid_at is None:
        raise ValueError(f'valid_at cannot be None for episode {record.get("uuid", "unknown")}')

    return EpisodicNode(
        content=record['content'],
        created_at=created_at,
        valid_at=valid_at,
        uuid=record['uuid'],
        group_id=record['group_id'],
        source=EpisodeType.from_str(record['source']),
        name=record['name'],
        source_description=record['source_description'],
        entity_edges=list(record.get('entity_edges', []) or []),
    )
