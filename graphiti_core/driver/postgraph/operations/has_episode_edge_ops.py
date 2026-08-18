from __future__ import annotations

import json
import logging
from typing import Any

from graphiti_core.driver.operations.has_episode_edge_ops import HasEpisodeEdgeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.edges import HasEpisodeEdge
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.helpers import parse_db_date

logger = logging.getLogger(__name__)

TABLE = 'has_episode_edges'
SOURCE_TABLE = 'saga_nodes'
TARGET_TABLE = 'episodic_nodes'


class PGHasEpisodeEdgeOperations(HasEpisodeEdgeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        edge: HasEpisodeEdge,
        _tx: Transaction | None = None,
    ) -> None:
        client = executor.client
        from_id = await executor._resolve_vertex_id(
            SOURCE_TABLE, edge.group_id, edge.source_node_uuid,
        )
        to_id = await executor._resolve_vertex_id(
            TARGET_TABLE, edge.group_id, edge.target_node_uuid,
        )
        if from_id is None or to_id is None:
            logger.warning(
                f'Cannot save edge {edge.uuid}: '
                f'source={edge.source_node_uuid} (id={from_id}) '
                f'target={edge.target_node_uuid} (id={to_id})',
            )
            return

        payload = {
            'uuid': edge.uuid,
            'source_node_uuid': edge.source_node_uuid,
            'target_node_uuid': edge.target_node_uuid,
            'created_at': edge.created_at.isoformat() if edge.created_at else None,
        }

        e_id = await executor._resolve_edge_id(TABLE, edge.group_id, edge.uuid)
        await client.upsert_edge(
            TABLE, edge.group_id,
            from_id=from_id,
            to_id=to_id,
            relation_type='HAS_EPISODE',
            edge_id=e_id,
            payload=payload,
        )
        logger.debug(f'Saved Edge to Graph: {edge.uuid}')

    async def save_bulk(
        self,
        executor: QueryExecutor,
        edges: list[HasEpisodeEdge],
        _tx: Transaction | None = None,
        _batch_size: int = 100,
    ) -> None:
        for edge in edges:
            await self.save(executor, edge)

    async def delete(
        self,
        executor: QueryExecutor,
        edge: HasEpisodeEdge,
        _tx: Transaction | None = None,
    ) -> None:
        client = executor.client
        e_id = await executor._resolve_edge_id(TABLE, edge.group_id, edge.uuid)
        if e_id is not None:
            await client._execute(
                f'DELETE FROM "{TABLE}" WHERE realm = $1 AND id = $2',
                edge.group_id, e_id,
            )
        logger.debug(f'Deleted Edge: {edge.uuid}')

    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        _tx: Transaction | None = None,
    ) -> None:
        if not uuids:
            return
        client = executor.client
        uuid_json_list = [json.dumps({'uuid': u}) for u in uuids]
        placeholders = ' OR '.join(
            f'payload @> ${i + 1}::jsonb' for i in range(len(uuids))
        )
        await client._execute(
            f'DELETE FROM "{TABLE}" WHERE {placeholders}',
            *uuid_json_list,
        )

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> HasEpisodeEdge:
        client = executor.client
        rows = await client._fetch(
            f'SELECT realm, id, from_id, to_id, payload, created_at FROM "{TABLE}" '
            'WHERE payload @> $1::jsonb',
            json.dumps({'uuid': uuid}),
        )
        if not rows:
            raise EdgeNotFoundError(uuid)
        return _row_to_has_episode_edge(dict(rows[0]))

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[HasEpisodeEdge]:
        if not uuids:
            return []
        client = executor.client
        placeholders = ' OR '.join(
            f'payload @> ${i + 1}::jsonb' for i in range(len(uuids))
        )
        uuid_json_list = [json.dumps({'uuid': u}) for u in uuids]
        rows = await client._fetch(
            f'SELECT realm, id, from_id, to_id, payload, created_at FROM "{TABLE}" '
            f'WHERE {placeholders}',
            *uuid_json_list,
        )
        return [_row_to_has_episode_edge(dict(r)) for r in rows]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[HasEpisodeEdge]:
        if not group_ids:
            return []
        client = executor.client
        conditions = ['realm = ANY($1)']
        params: list[Any] = [group_ids]
        if uuid_cursor:
            conditions.append(f"(payload->>'uuid') < $2")
            params.append(uuid_cursor)
        where = ' AND '.join(conditions)
        limit_clause = f' LIMIT {limit}' if limit is not None else ''
        rows = await client._fetch(
            f'SELECT realm, id, from_id, to_id, payload, created_at FROM "{TABLE}" '
            f"WHERE {where} ORDER BY payload->>'uuid' DESC{limit_clause}",
            *params,
        )
        return [_row_to_has_episode_edge(dict(r)) for r in rows]


def _row_to_has_episode_edge(row: dict) -> HasEpisodeEdge:
    p = row.get('payload', {})
    if isinstance(p, str):
        p = json.loads(p)
    return HasEpisodeEdge(
        uuid=p['uuid'],
        source_node_uuid=p['source_node_uuid'],
        target_node_uuid=p['target_node_uuid'],
        group_id=row.get('realm', ''),
        created_at=parse_db_date(p.get('created_at')) or row.get('created_at'),
    )
