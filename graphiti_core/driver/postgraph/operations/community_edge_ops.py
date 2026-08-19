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
        edge = _as_obj(edge)
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


_KNOWN_FIELDS = {
    'uuid', 'group_id', 'name', 'fact', 'fact_embedding', 'episodes',
    'source_node_uuid', 'target_node_uuid', 'created_at', 'expired_at',
    'valid_at', 'invalid_at', 'reference_time', 'attributes', 'summary',
    'name_embedding', 'labels', 'source', 'source_description', 'content',
    'entity_edges', 'level', 'saga_uuid',
}


def _as_obj(item):
    """Bulk paths pass model_dump() dicts; save() expects attribute access.

    bulk_utils serialises edges with `edge.model_dump()` because the Cypher
    drivers UNWIND a list of maps. save() here reads attributes, so the bulk
    path failed on the first add_episode(). The driver's tests call save()
    directly and never exercise it.
    """
    if not isinstance(item, dict):
        return item
    from datetime import datetime
    from types import SimpleNamespace

    known = {k: v for k, v in item.items() if k in _KNOWN_FIELDS}
    # bulk_utils flattens custom attributes to top level for the Cypher
    # drivers, which SET them as map properties. Invert that here.
    extra = {k: v for k, v in item.items() if k not in _KNOWN_FIELDS}
    data = dict(known)
    data['attributes'] = {**(known.get('attributes') or {}), **extra}
    for key, value in list(data.items()):
        if isinstance(value, str) and key.endswith(('_at', '_time')):
            try:
                data[key] = datetime.fromisoformat(value)
            except ValueError:
                pass
    return SimpleNamespace(**data)
