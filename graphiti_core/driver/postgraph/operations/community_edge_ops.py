from __future__ import annotations

import json
import logging

from graphiti_core.driver.operations.community_edge_ops import (
    CommunityEdgeOperations,
)
from graphiti_core.driver.query_executor import (
    QueryExecutor,
    Transaction,
)
from graphiti_core.edges import CommunityEdge
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.helpers import parse_db_date

logger = logging.getLogger(__name__)

TABLE = 'community_edges'
SOURCE_TABLE = 'community_nodes'
TARGET_TABLE = 'entity_nodes'
RELATION_TYPE = 'HAS_MEMBER'

_EDGE_COLS = (
    'realm, id, space, fqid, from_id, to_id, '
    'relation_type, payload, created_at, updated_at, '
    'uuid::text AS uuid_text'
)


class PGCommunityEdgeOperations(CommunityEdgeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        edge: CommunityEdge,
        _tx: Transaction | None = None,
    ) -> None:
        client = executor.client
        from_id = await executor._resolve_vertex_id(
            SOURCE_TABLE,
            edge.group_id,
            edge.source_node_uuid,
        )
        to_id = await executor._resolve_vertex_id(
            TARGET_TABLE,
            edge.group_id,
            edge.target_node_uuid,
        )
        existing_id = await executor._resolve_edge_id(
            TABLE,
            edge.group_id,
            edge.uuid,
        )
        payload = _build_payload(edge)
        await client.upsert_edge(
            TABLE,
            realm=edge.group_id,
            from_id=str(from_id),
            to_id=str(to_id),
            relation_type=RELATION_TYPE,
            edge_id=existing_id,
            payload=payload,
        )
        logger.debug('Saved Edge to Graph: %s', edge.uuid)

    async def delete(
        self,
        executor: QueryExecutor,
        edge: CommunityEdge,
        _tx: Transaction | None = None,
    ) -> None:
        client = executor.client
        eid = await executor._resolve_edge_id(
            TABLE,
            edge.group_id,
            edge.uuid,
        )
        if eid is not None:
            await client.delete_edge(
                TABLE,
                edge.group_id,
                eid,
            )
        logger.debug('Deleted Edge: %s', edge.uuid)

    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        _tx: Transaction | None = None,
    ) -> None:
        if not uuids:
            return
        client = executor.client
        await client._execute(
            f'DELETE FROM "{TABLE}" WHERE payload->>\'uuid\' = ANY($1)',
            uuids,
        )

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> CommunityEdge:
        client = executor.client
        rows = await client._fetch(
            f'SELECT {_EDGE_COLS} FROM "{TABLE}" t WHERE payload @> $1::jsonb',
            json.dumps({'uuid': uuid}),
        )
        edges = [_parse(r) for r in rows]
        if not edges:
            raise EdgeNotFoundError(uuid)
        return edges[0]

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[CommunityEdge]:
        if not uuids:
            return []
        client = executor.client
        rows = await client._fetch(
            f'SELECT {_EDGE_COLS} FROM "{TABLE}" t WHERE payload->>\'uuid\' = ANY($1)',
            uuids,
        )
        return [_parse(r) for r in rows]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[CommunityEdge]:
        client = executor.client
        query = f'SELECT {_EDGE_COLS} FROM "{TABLE}" t WHERE realm = ANY($1)'
        args: list = [group_ids]
        if uuid_cursor:
            query += " AND payload->>'uuid' < $2"
            args.append(uuid_cursor)
        query += ' ORDER BY id DESC'
        if limit is not None:
            query += f' LIMIT {limit}'
        rows = await client._fetch(query, *args)
        return [_parse(r) for r in rows]


def _build_payload(edge: CommunityEdge) -> dict:
    return {
        'uuid': edge.uuid,
        'source_node_uuid': edge.source_node_uuid,
        'target_node_uuid': edge.target_node_uuid,
        'created_at': (edge.created_at.isoformat() if edge.created_at else None),
    }


def _parse(row) -> CommunityEdge:
    payload = row['payload']
    if isinstance(payload, str):
        payload = json.loads(payload)
    payload = payload or {}
    return CommunityEdge(
        uuid=payload['uuid'],
        group_id=row['realm'],
        source_node_uuid=payload['source_node_uuid'],
        target_node_uuid=payload['target_node_uuid'],
        created_at=parse_db_date(
            payload.get('created_at'),
        ),
    )
