from __future__ import annotations

import json
import logging
from typing import Any

from post_graph import Vertex

from graphiti_core.driver.operations.entity_node_ops import EntityNodeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.helpers import parse_db_date
from graphiti_core.nodes import EntityNode

logger = logging.getLogger(__name__)

TABLE = 'entity_nodes'


class PGEntityNodeOperations(EntityNodeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        node: EntityNode,
        _tx: Transaction | None = None,
    ) -> None:
        client = executor.client
        attributes = dict(node.attributes or {})
        labels = list(set(node.labels + ['Entity']))
        payload = {
            'uuid': node.uuid,
            'name': node.name,
            'summary': node.summary,
            'labels': labels,
            'attributes': attributes,
            'created_at': (node.created_at.isoformat() if node.created_at else None),
        }

        # Resolve existing vertex id for upsert by payload UUID
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
        nodes: list[EntityNode],
        _tx: Transaction | None = None,
        _batch_size: int = 100,
    ) -> None:
        for node in nodes:
            await self.save(executor, node)

    async def delete(
        self,
        executor: QueryExecutor,
        node: EntityNode,
        _tx: Transaction | None = None,
    ) -> None:
        client = executor.client
        # Delete referencing edges first
        await client._execute(
            'DELETE FROM "episodic_edges" '
            'WHERE realm = $1 AND to_id IN ('
            '  SELECT id FROM "entity_nodes" '
            '  WHERE realm = $1 AND payload @> $2::jsonb'
            ')',
            node.group_id,
            json.dumps({'uuid': node.uuid}),
        )
        await client._execute(
            'DELETE FROM "entity_edges" '
            'WHERE realm = $1 AND ('
            '  from_id IN ('
            '    SELECT id FROM "entity_nodes" '
            '    WHERE realm = $1 AND payload @> $2::jsonb'
            '  ) OR to_id IN ('
            '    SELECT id FROM "entity_nodes" '
            '    WHERE realm = $1 AND payload @> $2::jsonb'
            '  )'
            ')',
            node.group_id,
            json.dumps({'uuid': node.uuid}),
        )
        await client._execute(
            'DELETE FROM "community_edges" '
            'WHERE realm = $1 AND to_id IN ('
            '  SELECT id FROM "entity_nodes" '
            '  WHERE realm = $1 AND payload @> $2::jsonb'
            ')',
            node.group_id,
            json.dumps({'uuid': node.uuid}),
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
        # Delete edges that reference entity_nodes in this realm
        await client._execute(
            'DELETE FROM "episodic_edges" '
            'WHERE realm = $1 AND to_id IN ('
            '  SELECT id FROM "entity_nodes" WHERE realm = $1'
            ')',
            group_id,
        )
        await client._execute(
            'DELETE FROM "entity_edges" WHERE realm = $1',
            group_id,
        )
        await client._execute(
            'DELETE FROM "community_edges" '
            'WHERE realm = $1 AND to_id IN ('
            '  SELECT id FROM "entity_nodes" WHERE realm = $1'
            ')',
            group_id,
        )
        # Delete all entity_nodes for the realm
        vertices = await client.get_vertices(TABLE, group_id)
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

        # Collect all realms that contain these nodes
        id_rows = await client._fetch(
            f'SELECT realm, id FROM "{TABLE}" WHERE {placeholders}',
            *uuid_json_list,
        )
        if not id_rows:
            return

        realm_ids: dict[str, list[int]] = {}
        for row in id_rows:
            realm_ids.setdefault(row['realm'], []).append(row['id'])

        # Delete referencing edges
        for realm, ids in realm_ids.items():
            id_arr = ids
            await client._execute(
                'DELETE FROM "episodic_edges" WHERE realm = $1 AND to_id = ANY($2)',
                realm,
                id_arr,
            )
            await client._execute(
                'DELETE FROM "entity_edges" '
                'WHERE realm = $1 AND '
                '(from_id = ANY($2) OR to_id = ANY($2))',
                realm,
                id_arr,
            )
            await client._execute(
                'DELETE FROM "community_edges" WHERE realm = $1 AND to_id = ANY($2)',
                realm,
                id_arr,
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
    ) -> EntityNode:
        client = executor.client
        # Cross-realm lookup via payload UUID
        rows = await client._fetch(
            'SELECT realm, id, space, fqid, payload, '
            'created_at, updated_at, '
            'uuid::text AS uuid_text, '
            "to_jsonb(t)->>'embedding' AS embedding_text "
            f'FROM "{TABLE}" t '
            'WHERE payload @> $1::jsonb',
            json.dumps({'uuid': uuid}),
        )
        if not rows:
            raise NodeNotFoundError(uuid)
        return _row_to_entity_node(dict(rows[0]))

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[EntityNode]:
        if not uuids:
            return []
        client = executor.client
        placeholders = ' OR '.join(f'payload @> ${i + 1}::jsonb' for i in range(len(uuids)))
        uuid_json_list = [json.dumps({'uuid': u}) for u in uuids]
        rows = await client._fetch(
            'SELECT realm, id, space, fqid, payload, '
            'created_at, updated_at, '
            'uuid::text AS uuid_text, '
            "to_jsonb(t)->>'embedding' AS embedding_text "
            f'FROM "{TABLE}" t '
            f'WHERE {placeholders}',
            *uuid_json_list,
        )
        return [_row_to_entity_node(dict(r)) for r in rows]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EntityNode]:
        if not group_ids:
            return []
        client = executor.client
        results: list[EntityNode] = []
        for realm in group_ids:
            vertices = await client.get_vertices(
                TABLE,
                realm,
            )
            for v in vertices:
                results.append(_vertex_to_entity_node(v))

        # Sort by payload uuid descending for cursor pagination
        results.sort(
            key=lambda n: n.uuid,
            reverse=True,
        )

        if uuid_cursor:
            results = [n for n in results if n.uuid < uuid_cursor]
        if limit is not None:
            results = results[:limit]
        return results

    async def load_embeddings(
        self,
        executor: QueryExecutor,
        node: EntityNode,
    ) -> None:
        client = executor.client
        rows = await client._fetch(
            'SELECT realm, id, space, fqid, payload, '
            'created_at, updated_at, '
            'uuid::text AS uuid_text, '
            "to_jsonb(t)->>'embedding' AS embedding_text "
            f'FROM "{TABLE}" t '
            'WHERE payload @> $1::jsonb',
            json.dumps({'uuid': node.uuid}),
        )
        if not rows:
            raise NodeNotFoundError(node.uuid)
        emb_text = rows[0].get('embedding_text')
        node.name_embedding = _parse_vec(emb_text)

    async def load_embeddings_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[EntityNode],
        _batch_size: int = 100,
    ) -> None:
        if not nodes:
            return
        client = executor.client
        uuids = [n.uuid for n in nodes]
        placeholders = ' OR '.join(f'payload @> ${i + 1}::jsonb' for i in range(len(uuids)))
        uuid_json_list = [json.dumps({'uuid': u}) for u in uuids]
        rows = await client._fetch(
            'SELECT payload, '
            "to_jsonb(t)->>'embedding' AS embedding_text "
            f'FROM "{TABLE}" t '
            f'WHERE {placeholders}',
            *uuid_json_list,
        )
        embedding_map: dict[str, list[float] | None] = {}
        for r in rows:
            p = r['payload']
            if isinstance(p, str):
                p = json.loads(p)
            row_uuid = p.get('uuid')
            if row_uuid:
                embedding_map[row_uuid] = _parse_vec(r.get('embedding_text'))
        for node in nodes:
            if node.uuid in embedding_map:
                node.name_embedding = embedding_map[node.uuid]


def _vertex_to_entity_node(v: Vertex) -> EntityNode:
    """Convert a post-graph Vertex to a Graphiti EntityNode."""
    p = v.payload or {}
    attributes = dict(p.get('attributes', {}) or {})
    # Remove keys that are top-level fields, not attributes
    for key in (
        'uuid',
        'name',
        'group_id',
        'name_embedding',
        'summary',
        'created_at',
        'labels',
    ):
        attributes.pop(key, None)

    labels = list(p.get('labels', []) or [])
    realm = v.realm or ''
    dynamic_label = 'Entity_' + realm.replace('-', '')
    if dynamic_label in labels:
        labels.remove(dynamic_label)

    return EntityNode(
        uuid=p.get('uuid', ''),
        name=p.get('name', ''),
        name_embedding=v.embedding,
        group_id=realm,
        labels=labels,
        created_at=parse_db_date(p.get('created_at')),
        summary=p.get('summary', ''),
        attributes=attributes,
    )


def _row_to_entity_node(row: dict) -> EntityNode:
    """Convert a raw DB row to a Graphiti EntityNode."""
    p = row.get('payload', {})
    if isinstance(p, str):
        p = json.loads(p)

    attributes = dict(p.get('attributes', {}) or {})
    for key in (
        'uuid',
        'name',
        'group_id',
        'name_embedding',
        'summary',
        'created_at',
        'labels',
    ):
        attributes.pop(key, None)

    labels = list(p.get('labels', []) or [])
    realm = row.get('realm', '')
    dynamic_label = 'Entity_' + realm.replace('-', '')
    if dynamic_label in labels:
        labels.remove(dynamic_label)

    embedding = _parse_vec(row.get('embedding_text'))

    return EntityNode(
        uuid=p.get('uuid', ''),
        name=p.get('name', ''),
        name_embedding=embedding,
        group_id=realm,
        labels=labels,
        created_at=parse_db_date(p.get('created_at')),
        summary=p.get('summary', ''),
        attributes=attributes,
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
