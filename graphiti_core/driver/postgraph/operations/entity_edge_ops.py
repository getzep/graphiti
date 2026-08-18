from __future__ import annotations

import json
import logging
from typing import Any

from graphiti_core.driver.operations.entity_edge_ops import EntityEdgeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.helpers import parse_db_date

logger = logging.getLogger(__name__)


class PGEntityEdgeOperations(EntityEdgeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        edge: EntityEdge,
        tx: Transaction | None = None,
    ) -> None:
        attributes = dict(edge.attributes or {})
        query = """
            INSERT INTO entity_edges
                (uuid, source_node_uuid, target_node_uuid, name, fact, fact_embedding,
                 group_id, episodes, created_at, expired_at, valid_at, invalid_at,
                 reference_time, attributes)
            VALUES ($uuid, $source_node_uuid, $target_node_uuid, $name, $fact, $fact_embedding,
                    $group_id, $episodes, $created_at, $expired_at, $valid_at, $invalid_at,
                    $reference_time, $attributes)
            ON CONFLICT (uuid) DO UPDATE SET
                source_node_uuid = EXCLUDED.source_node_uuid,
                target_node_uuid = EXCLUDED.target_node_uuid,
                name = EXCLUDED.name,
                fact = EXCLUDED.fact,
                fact_embedding = EXCLUDED.fact_embedding,
                group_id = EXCLUDED.group_id,
                episodes = EXCLUDED.episodes,
                created_at = EXCLUDED.created_at,
                expired_at = EXCLUDED.expired_at,
                valid_at = EXCLUDED.valid_at,
                invalid_at = EXCLUDED.invalid_at,
                reference_time = EXCLUDED.reference_time,
                attributes = EXCLUDED.attributes
        """
        run = tx.run if tx else executor.execute_query
        await run(
            query,
            uuid=edge.uuid,
            source_node_uuid=edge.source_node_uuid,
            target_node_uuid=edge.target_node_uuid,
            name=edge.name,
            fact=edge.fact,
            fact_embedding=_vec(edge.fact_embedding),
            group_id=edge.group_id,
            episodes=edge.episodes,
            created_at=edge.created_at,
            expired_at=edge.expired_at,
            valid_at=edge.valid_at,
            invalid_at=edge.invalid_at,
            reference_time=edge.reference_time,
            attributes=json.dumps(attributes),
        )
        logger.debug(f'Saved Edge to Graph: {edge.uuid}')

    async def save_bulk(
        self,
        executor: QueryExecutor,
        edges: list[EntityEdge],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        for edge in edges:
            await self.save(executor, edge, tx=tx)

    async def delete(
        self,
        executor: QueryExecutor,
        edge: EntityEdge,
        tx: Transaction | None = None,
    ) -> None:
        run = tx.run if tx else executor.execute_query
        await run('DELETE FROM entity_edges WHERE uuid = $uuid', uuid=edge.uuid)
        logger.debug(f'Deleted Edge: {edge.uuid}')

    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None = None,
    ) -> None:
        run = tx.run if tx else executor.execute_query
        await run('DELETE FROM entity_edges WHERE uuid = ANY($uuids)', uuids=uuids)

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> EntityEdge:
        records, _, _ = await executor.execute_query(
            _SELECT + ' WHERE uuid = $uuid',
            uuid=uuid,
        )
        edges = [_parse(r) for r in records]
        if not edges:
            raise EdgeNotFoundError(uuid)
        return edges[0]

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[EntityEdge]:
        if not uuids:
            return []
        records, _, _ = await executor.execute_query(
            _SELECT + ' WHERE uuid = ANY($uuids)',
            uuids=uuids,
        )
        return [_parse(r) for r in records]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EntityEdge]:
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

    async def get_between_nodes(
        self,
        executor: QueryExecutor,
        source_node_uuid: str,
        target_node_uuid: str,
    ) -> list[EntityEdge]:
        records, _, _ = await executor.execute_query(
            _SELECT + ' WHERE source_node_uuid = $source_node_uuid AND target_node_uuid = $target_node_uuid',
            source_node_uuid=source_node_uuid,
            target_node_uuid=target_node_uuid,
        )
        return [_parse(r) for r in records]

    async def get_by_node_uuid(
        self,
        executor: QueryExecutor,
        node_uuid: str,
    ) -> list[EntityEdge]:
        records, _, _ = await executor.execute_query(
            _SELECT + ' WHERE source_node_uuid = $node_uuid OR target_node_uuid = $node_uuid',
            node_uuid=node_uuid,
        )
        return [_parse(r) for r in records]

    async def load_embeddings(
        self,
        executor: QueryExecutor,
        edge: EntityEdge,
    ) -> None:
        records, _, _ = await executor.execute_query(
            'SELECT fact_embedding FROM entity_edges WHERE uuid = $uuid',
            uuid=edge.uuid,
        )
        if not records:
            raise EdgeNotFoundError(edge.uuid)
        edge.fact_embedding = _parse_vec(records[0]['fact_embedding'])

    async def load_embeddings_bulk(
        self,
        executor: QueryExecutor,
        edges: list[EntityEdge],
        batch_size: int = 100,
    ) -> None:
        uuids = [e.uuid for e in edges]
        records, _, _ = await executor.execute_query(
            'SELECT DISTINCT uuid, fact_embedding FROM entity_edges WHERE uuid = ANY($uuids)',
            uuids=uuids,
        )
        embedding_map = {r['uuid']: _parse_vec(r['fact_embedding']) for r in records}
        for edge in edges:
            if edge.uuid in embedding_map:
                edge.fact_embedding = embedding_map[edge.uuid]


_SELECT = """
    SELECT uuid, source_node_uuid, target_node_uuid, name, fact,
           group_id, episodes, created_at, expired_at, valid_at,
           invalid_at, reference_time, attributes
    FROM entity_edges
"""


def _parse(record: dict) -> EntityEdge:
    attributes = record.get('attributes', {}) or {}
    if isinstance(attributes, str):
        attributes = json.loads(attributes)
    attributes.pop('uuid', None)
    attributes.pop('source_node_uuid', None)
    attributes.pop('target_node_uuid', None)
    attributes.pop('fact', None)
    attributes.pop('fact_embedding', None)
    attributes.pop('name', None)
    attributes.pop('group_id', None)
    attributes.pop('episodes', None)
    attributes.pop('created_at', None)
    attributes.pop('expired_at', None)
    attributes.pop('valid_at', None)
    attributes.pop('invalid_at', None)
    attributes.pop('reference_time', None)

    return EntityEdge(
        uuid=record['uuid'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        fact=record['fact'],
        fact_embedding=record.get('fact_embedding'),
        name=record['name'],
        group_id=record['group_id'],
        episodes=list(record.get('episodes', []) or []),
        created_at=parse_db_date(record['created_at']),
        expired_at=parse_db_date(record.get('expired_at')),
        valid_at=parse_db_date(record.get('valid_at')),
        invalid_at=parse_db_date(record.get('invalid_at')),
        reference_time=parse_db_date(record.get('reference_time')),
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
