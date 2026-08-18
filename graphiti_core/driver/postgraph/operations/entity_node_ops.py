from __future__ import annotations

import json
import logging
from typing import Any

from graphiti_core.driver.operations.entity_node_ops import EntityNodeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.nodes import EntityNode

logger = logging.getLogger(__name__)


class PGEntityNodeOperations(EntityNodeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        node: EntityNode,
        tx: Transaction | None = None,
    ) -> None:
        attributes = dict(node.attributes or {})
        labels = list(set(node.labels + ['Entity']))
        query = """
            INSERT INTO entity_nodes (uuid, name, group_id, labels, summary, name_embedding, attributes, created_at)
            VALUES ($uuid, $name, $group_id, $labels, $summary, $name_embedding, $attributes, $created_at)
            ON CONFLICT (uuid) DO UPDATE SET
                name = EXCLUDED.name,
                group_id = EXCLUDED.group_id,
                labels = EXCLUDED.labels,
                summary = EXCLUDED.summary,
                name_embedding = EXCLUDED.name_embedding,
                attributes = EXCLUDED.attributes,
                created_at = EXCLUDED.created_at
        """
        run = tx.run if tx else executor.execute_query
        await run(
            query,
            uuid=node.uuid,
            name=node.name,
            group_id=node.group_id,
            labels=labels,
            summary=node.summary,
            name_embedding=_vec(node.name_embedding),
            attributes=json.dumps(attributes),
            created_at=node.created_at,
        )
        logger.debug(f'Saved Node to Graph: {node.uuid}')

    async def save_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[EntityNode],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        for node in nodes:
            await self.save(executor, node, tx=tx)

    async def delete(
        self,
        executor: QueryExecutor,
        node: EntityNode,
        tx: Transaction | None = None,
    ) -> None:
        run = tx.run if tx else executor.execute_query
        await run('DELETE FROM episodic_edges WHERE target_node_uuid = $uuid', uuid=node.uuid)
        await run(
            'DELETE FROM entity_edges WHERE source_node_uuid = $uuid OR target_node_uuid = $uuid',
            uuid=node.uuid,
        )
        await run('DELETE FROM community_edges WHERE target_node_uuid = $uuid', uuid=node.uuid)
        await run('DELETE FROM entity_nodes WHERE uuid = $uuid', uuid=node.uuid)
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
            """
            DELETE FROM episodic_edges WHERE target_node_uuid IN (
                SELECT uuid FROM entity_nodes WHERE group_id = $group_id
            )
            """,
            group_id=group_id,
        )
        await run('DELETE FROM entity_edges WHERE group_id = $group_id', group_id=group_id)
        await run(
            """
            DELETE FROM community_edges WHERE target_node_uuid IN (
                SELECT uuid FROM entity_nodes WHERE group_id = $group_id
            )
            """,
            group_id=group_id,
        )
        await run('DELETE FROM entity_nodes WHERE group_id = $group_id', group_id=group_id)

    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        run = tx.run if tx else executor.execute_query
        await run(
            'DELETE FROM episodic_edges WHERE target_node_uuid = ANY($uuids)',
            uuids=uuids,
        )
        await run(
            'DELETE FROM entity_edges WHERE source_node_uuid = ANY($uuids) OR target_node_uuid = ANY($uuids)',
            uuids=uuids,
        )
        await run(
            'DELETE FROM community_edges WHERE target_node_uuid = ANY($uuids)',
            uuids=uuids,
        )
        await run('DELETE FROM entity_nodes WHERE uuid = ANY($uuids)', uuids=uuids)

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> EntityNode:
        records, _, _ = await executor.execute_query(
            """
            SELECT uuid, name, group_id, labels, summary, attributes, created_at
            FROM entity_nodes WHERE uuid = $uuid
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
    ) -> list[EntityNode]:
        records, _, _ = await executor.execute_query(
            """
            SELECT uuid, name, group_id, labels, summary, attributes, created_at
            FROM entity_nodes WHERE uuid = ANY($uuids)
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
    ) -> list[EntityNode]:
        cursor_clause = 'AND uuid < $uuid_cursor' if uuid_cursor else ''
        limit_clause = f'LIMIT {limit}' if limit is not None else ''
        query = f"""
            SELECT uuid, name, group_id, labels, summary, attributes, created_at
            FROM entity_nodes
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

    async def load_embeddings(
        self,
        executor: QueryExecutor,
        node: EntityNode,
    ) -> None:
        records, _, _ = await executor.execute_query(
            'SELECT name_embedding FROM entity_nodes WHERE uuid = $uuid',
            uuid=node.uuid,
        )
        if not records:
            raise NodeNotFoundError(node.uuid)
        emb = records[0]['name_embedding']
        node.name_embedding = _parse_vec(emb)

    async def load_embeddings_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[EntityNode],
        batch_size: int = 100,
    ) -> None:
        uuids = [n.uuid for n in nodes]
        records, _, _ = await executor.execute_query(
            'SELECT DISTINCT uuid, name_embedding FROM entity_nodes WHERE uuid = ANY($uuids)',
            uuids=uuids,
        )
        embedding_map = {r['uuid']: _parse_vec(r['name_embedding']) for r in records}
        for node in nodes:
            if node.uuid in embedding_map:
                node.name_embedding = embedding_map[node.uuid]


def _parse(record: dict) -> EntityNode:
    from graphiti_core.helpers import parse_db_date

    attributes = record.get('attributes', {}) or {}
    if isinstance(attributes, str):
        attributes = json.loads(attributes)
    attributes.pop('uuid', None)
    attributes.pop('name', None)
    attributes.pop('group_id', None)
    attributes.pop('name_embedding', None)
    attributes.pop('summary', None)
    attributes.pop('created_at', None)
    attributes.pop('labels', None)

    labels = list(record.get('labels', []) or [])
    group_id = record.get('group_id', '')
    dynamic_label = 'Entity_' + group_id.replace('-', '')
    if dynamic_label in labels:
        labels.remove(dynamic_label)

    return EntityNode(
        uuid=record['uuid'],
        name=record['name'],
        name_embedding=record.get('name_embedding'),
        group_id=group_id,
        labels=labels,
        created_at=parse_db_date(record['created_at']),
        summary=record.get('summary', ''),
        attributes=attributes,
    )


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
