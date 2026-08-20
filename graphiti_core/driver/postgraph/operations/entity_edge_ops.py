from __future__ import annotations

import contextlib
import json
import logging

from graphiti_core.driver.operations.entity_edge_ops import (
    EntityEdgeOperations,
)
from graphiti_core.driver.query_executor import (
    QueryExecutor,
    Transaction,
)
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.helpers import parse_db_date

logger = logging.getLogger(__name__)

TABLE = 'entity_edges'
SOURCE_TABLE = 'entity_nodes'
TARGET_TABLE = 'entity_nodes'

_EDGE_COLS = (
    'realm, id, space, fqid, from_id, to_id, '
    'relation_type, payload, created_at, updated_at, '
    'uuid::text AS uuid_text, '
    "to_jsonb(t)->>'embedding' AS embedding_text"
)


class PGEntityEdgeOperations(EntityEdgeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        edge: EntityEdge,
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
        if from_id is None or to_id is None:
            logger.warning(
                'Cannot save edge %s: source=%s (id=%s) target=%s (id=%s)',
                edge.uuid,
                edge.source_node_uuid,
                from_id,
                edge.target_node_uuid,
                to_id,
            )
            return
        existing_id = await executor._resolve_edge_id(
            TABLE,
            edge.group_id,
            edge.uuid,
        )
        payload = _build_payload(edge)
        await client.upsert_edge(
            TABLE,
            realm=edge.group_id,
            from_id=from_id,
            to_id=to_id,
            relation_type=edge.name,
            edge_id=existing_id,
            payload=payload,
            embedding=edge.fact_embedding,
        )
        logger.debug('Saved Edge to Graph: %s', edge.uuid)

    async def save_bulk(
        self,
        executor: QueryExecutor,
        edges: list[EntityEdge],
        _tx: Transaction | None = None,
        _batch_size: int = 100,
    ) -> None:
        edges = [_as_obj(x) for x in edges]
        for edge in edges:
            await self.save(executor, edge)

    async def delete(
        self,
        executor: QueryExecutor,
        edge: EntityEdge,
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
    ) -> EntityEdge:
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
    ) -> list[EntityEdge]:
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
    ) -> list[EntityEdge]:
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

    async def get_between_nodes(
        self,
        executor: QueryExecutor,
        source_node_uuid: str,
        target_node_uuid: str,
    ) -> list[EntityEdge]:
        client = executor.client
        rows = await client._fetch(
            f'SELECT {_EDGE_COLS} FROM "{TABLE}" t '
            "WHERE payload->>'source_node_uuid' = $1 "
            "AND payload->>'target_node_uuid' = $2",
            source_node_uuid,
            target_node_uuid,
        )
        return [_parse(r) for r in rows]

    async def get_by_node_uuid(
        self,
        executor: QueryExecutor,
        node_uuid: str,
    ) -> list[EntityEdge]:
        client = executor.client
        rows = await client._fetch(
            f'SELECT {_EDGE_COLS} FROM "{TABLE}" t '
            "WHERE payload->>'source_node_uuid' = $1 "
            "OR payload->>'target_node_uuid' = $1",
            node_uuid,
        )
        return [_parse(r) for r in rows]

    async def load_embeddings(
        self,
        executor: QueryExecutor,
        edge: EntityEdge,
    ) -> None:
        client = executor.client
        rows = await client._fetch(
            'SELECT '
            "to_jsonb(t)->>'embedding' AS embedding_text "
            f'FROM "{TABLE}" t '
            'WHERE payload @> $1::jsonb',
            json.dumps({'uuid': edge.uuid}),
        )
        if not rows:
            raise EdgeNotFoundError(edge.uuid)
        edge.fact_embedding = _parse_embedding(
            rows[0]['embedding_text'],
        )

    async def load_embeddings_bulk(
        self,
        executor: QueryExecutor,
        edges: list[EntityEdge],
        _batch_size: int = 100,
    ) -> None:
        if not edges:
            return
        client = executor.client
        uuids = [e.uuid for e in edges]
        rows = await client._fetch(
            'SELECT '
            "payload->>'uuid' AS edge_uuid, "
            "to_jsonb(t)->>'embedding' AS embedding_text "
            f'FROM "{TABLE}" t '
            "WHERE payload->>'uuid' = ANY($1)",
            uuids,
        )
        emb_map = {
            r['edge_uuid']: _parse_embedding(
                r['embedding_text'],
            )
            for r in rows
        }
        for edge in edges:
            if edge.uuid in emb_map:
                edge.fact_embedding = emb_map[edge.uuid]


def _build_payload(edge: EntityEdge) -> dict:
    return {
        'uuid': edge.uuid,
        'source_node_uuid': edge.source_node_uuid,
        'target_node_uuid': edge.target_node_uuid,
        'name': edge.name,
        'fact': edge.fact,
        'episodes': edge.episodes,
        'created_at': (edge.created_at.isoformat() if edge.created_at else None),
        'expired_at': (edge.expired_at.isoformat() if edge.expired_at else None),
        'valid_at': (edge.valid_at.isoformat() if edge.valid_at else None),
        'invalid_at': (edge.invalid_at.isoformat() if edge.invalid_at else None),
        'reference_time': (edge.reference_time.isoformat() if edge.reference_time else None),
        'attributes': dict(edge.attributes or {}),
    }


def _parse(row) -> EntityEdge:
    payload = row['payload']
    if isinstance(payload, str):
        payload = json.loads(payload)
    payload = payload or {}
    return EntityEdge(
        uuid=payload['uuid'],
        source_node_uuid=payload['source_node_uuid'],
        target_node_uuid=payload['target_node_uuid'],
        name=payload['name'],
        fact=payload['fact'],
        fact_embedding=_parse_embedding(
            row.get('embedding_text'),
        ),
        group_id=row['realm'],
        episodes=list(payload.get('episodes', []) or []),
        created_at=parse_db_date(
            payload.get('created_at'),
        ),
        expired_at=parse_db_date(
            payload.get('expired_at'),
        ),
        valid_at=parse_db_date(
            payload.get('valid_at'),
        ),
        invalid_at=parse_db_date(
            payload.get('invalid_at'),
        ),
        reference_time=parse_db_date(
            payload.get('reference_time'),
        ),
        attributes=payload.get('attributes', {}),
    )


def _parse_embedding(value) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if value.startswith('['):
            return json.loads(value)
        return [float(x) for x in value.split(',') if x.strip()]
    return list(value)


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
            with contextlib.suppress(ValueError):
                data[key] = datetime.fromisoformat(value)
    return SimpleNamespace(**data)
