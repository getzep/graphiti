from __future__ import annotations

import contextlib
import json
import logging
from typing import Any

from post_graph import Vertex

from graphiti_core.driver.operations.community_node_ops import (
    CommunityNodeOperations,
)
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.helpers import parse_db_date
from graphiti_core.nodes import CommunityNode

logger = logging.getLogger(__name__)

TABLE = 'community_nodes'


class PGCommunityNodeOperations(CommunityNodeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        node: CommunityNode,
        _tx: Transaction | None = None,
    ) -> None:
        node = _as_obj(node)
        client = executor.client
        payload = {
            'uuid': node.uuid,
            'name': node.name,
            'summary': node.summary,
            'created_at': (node.created_at.isoformat() if node.created_at else None),
        }

        v_id = await executor._resolve_vertex_id(
            TABLE,
            node.group_id,
            node.uuid,
        )
        await client.upsert_vertex(
            TABLE,
            node.group_id,
            vertex_id=v_id,
            payload=payload,
            embedding=node.name_embedding,
        )
        logger.debug(f'Saved Node to Graph: {node.uuid}')

    async def save_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[CommunityNode],
        _tx: Transaction | None = None,
        _batch_size: int = 100,
    ) -> None:
        nodes = [_as_obj(x) for x in nodes]
        for node in nodes:
            await self.save(executor, node)

    async def delete(
        self,
        executor: QueryExecutor,
        node: CommunityNode,
        _tx: Transaction | None = None,
    ) -> None:
        client = executor.client
        uuid_json = json.dumps({'uuid': node.uuid})
        # Delete referencing community_edges first
        await client._execute(
            'DELETE FROM "community_edges" '
            'WHERE realm = $1 AND from_id IN ('
            '  SELECT id FROM "community_nodes" '
            '  WHERE realm = $1 AND payload @> $2::jsonb'
            ')',
            node.group_id,
            uuid_json,
        )
        v_id = await executor._resolve_vertex_id(
            TABLE,
            node.group_id,
            node.uuid,
        )
        if v_id is not None:
            await client.delete_vertex(
                TABLE,
                node.group_id,
                str(v_id),
            )
        logger.debug(f'Deleted Node: {node.uuid}')

    async def delete_by_group_id(
        self,
        executor: QueryExecutor,
        group_id: str,
        _tx: Transaction | None = None,
        _batch_size: int = 100,
    ) -> None:
        client = executor.client
        await client._execute(
            'DELETE FROM "community_edges" '
            'WHERE realm = $1 AND from_id IN ('
            '  SELECT id FROM "community_nodes" '
            '  WHERE realm = $1'
            ')',
            group_id,
        )
        vertices = await client.get_vertices(
            TABLE,
            group_id,
        )
        for v in vertices:
            await client.delete_vertex(
                TABLE,
                group_id,
                str(v.id),
            )

    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        _tx: Transaction | None = None,
        _batch_size: int = 100,
    ) -> None:
        if not uuids:
            return
        client = executor.client
        uuid_json_list = [json.dumps({'uuid': u}) for u in uuids]
        placeholders = ' OR '.join(f'payload @> ${i + 1}::jsonb' for i in range(len(uuids)))

        id_rows = await client._fetch(
            f'SELECT realm, id FROM "{TABLE}" WHERE {placeholders}',
            *uuid_json_list,
        )
        if not id_rows:
            return

        realm_ids: dict[str, list[int]] = {}
        for row in id_rows:
            realm_ids.setdefault(row['realm'], []).append(row['id'])

        for realm, ids in realm_ids.items():
            await client._execute(
                'DELETE FROM "community_edges" WHERE realm = $1 AND from_id = ANY($2)',
                realm,
                ids,
            )
            for vid in ids:
                await client.delete_vertex(
                    TABLE,
                    realm,
                    str(vid),
                )

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> CommunityNode:
        client = executor.client
        rows = await client._fetch(
            'SELECT realm, id, space, fqid, payload, '
            'created_at, updated_at, '
            'uuid::text AS uuid_text, '
            "to_jsonb(t)->>'embedding' "
            'AS embedding_text '
            f'FROM "{TABLE}" t '
            'WHERE payload @> $1::jsonb',
            json.dumps({'uuid': uuid}),
        )
        if not rows:
            raise NodeNotFoundError(uuid)
        return _row_to_community_node(dict(rows[0]))

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[CommunityNode]:
        if not uuids:
            return []
        client = executor.client
        placeholders = ' OR '.join(f'payload @> ${i + 1}::jsonb' for i in range(len(uuids)))
        uuid_json_list = [json.dumps({'uuid': u}) for u in uuids]
        rows = await client._fetch(
            'SELECT realm, id, space, fqid, payload, '
            'created_at, updated_at, '
            'uuid::text AS uuid_text, '
            "to_jsonb(t)->>'embedding' "
            'AS embedding_text '
            f'FROM "{TABLE}" t '
            f'WHERE {placeholders}',
            *uuid_json_list,
        )
        return [_row_to_community_node(dict(r)) for r in rows]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[CommunityNode]:
        if not group_ids:
            return []
        client = executor.client
        results: list[CommunityNode] = []
        for realm in group_ids:
            vertices = await client.get_vertices(
                TABLE,
                realm,
            )
            for v in vertices:
                results.append(_vertex_to_community_node(v))

        results.sort(
            key=lambda n: n.uuid,
            reverse=True,
        )

        if uuid_cursor:
            results = [n for n in results if n.uuid < uuid_cursor]
        if limit is not None:
            results = results[:limit]
        return results

    async def load_name_embedding(
        self,
        executor: QueryExecutor,
        node: CommunityNode,
    ) -> None:
        client = executor.client
        rows = await client._fetch(
            'SELECT realm, id, space, fqid, payload, '
            'created_at, updated_at, '
            'uuid::text AS uuid_text, '
            "to_jsonb(t)->>'embedding' "
            'AS embedding_text '
            f'FROM "{TABLE}" t '
            'WHERE payload @> $1::jsonb',
            json.dumps({'uuid': node.uuid}),
        )
        if not rows:
            raise NodeNotFoundError(node.uuid)
        emb_text = rows[0].get('embedding_text')
        node.name_embedding = _parse_vec(emb_text)


def _vertex_to_community_node(v: Vertex) -> CommunityNode:
    """Convert a post-graph Vertex to a CommunityNode."""
    p = v.payload or {}
    return CommunityNode(
        uuid=p.get('uuid', ''),
        name=p.get('name', ''),
        name_embedding=v.embedding,
        group_id=v.realm or '',
        created_at=parse_db_date(p.get('created_at')),
        summary=p.get('summary', ''),
    )


def _row_to_community_node(row: dict) -> CommunityNode:
    """Convert a raw DB row to a CommunityNode."""
    p = row.get('payload', {})
    if isinstance(p, str):
        p = json.loads(p)

    embedding = _parse_vec(row.get('embedding_text'))

    return CommunityNode(
        uuid=p.get('uuid', ''),
        name=p.get('name', ''),
        name_embedding=embedding,
        group_id=row.get('realm', ''),
        created_at=parse_db_date(p.get('created_at')),
        summary=p.get('summary', ''),
    )


def _parse_vec(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        cleaned = value.strip('[]')
        if not cleaned:
            return None
        return [float(x) for x in cleaned.split(',')]
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
