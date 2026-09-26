from __future__ import annotations

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any

from post_graph import Vertex

from graphiti_core.driver.operations.episode_node_ops import (
    EpisodeNodeOperations,
)
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.helpers import parse_db_date
from graphiti_core.nodes import EpisodeType, EpisodicNode

logger = logging.getLogger(__name__)

TABLE = 'episodic_nodes'


class PGEpisodeNodeOperations(EpisodeNodeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        node: Any,
        _tx: Transaction | None = None,
    ) -> None:
        node = _as_episodic_node(node)
        client = executor.client
        payload = {
            'uuid': node.uuid,
            'name': node.name,
            'source': node.source.value if hasattr(node.source, 'value') else node.source,
            'source_description': node.source_description,
            'content': node.content,
            'valid_at': (node.valid_at.isoformat() if node.valid_at else None),
            'entity_edges': node.entity_edges,
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
        )
        logger.debug(f'Saved Node to Graph: {node.uuid}')

    async def save_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[Any],
        _tx: Transaction | None = None,
        _batch_size: int = 100,
    ) -> None:
        """Accepts EpisodicNode objects or the dicts the bulk path supplies.

        bulk_utils.add_nodes_and_edges_bulk_tx converts every node with
        `dict(episode)` before calling this — the Cypher drivers UNWIND a list
        of maps — and it stringifies `source` on the way. save() expects an
        object, so the bulk path failed with "'dict' object has no attribute
        'uuid'" on the first add_episode(). The driver's own tests call save()
        directly and never see it.
        """
        for node in nodes:
            await self.save(executor, _as_episodic_node(node))

    async def delete(
        self,
        executor: QueryExecutor,
        node: EpisodicNode,
        _tx: Transaction | None = None,
    ) -> None:
        client = executor.client
        uuid_json = json.dumps({'uuid': node.uuid})
        # Delete referencing edges
        await client._execute(
            'DELETE FROM "episodic_edges" '
            'WHERE realm = $1 AND from_id IN ('
            '  SELECT id FROM "episodic_nodes" '
            '  WHERE realm = $1 AND payload @> $2::jsonb'
            ')',
            node.group_id,
            uuid_json,
        )
        await client._execute(
            'DELETE FROM "has_episode_edges" '
            'WHERE realm = $1 AND to_id IN ('
            '  SELECT id FROM "episodic_nodes" '
            '  WHERE realm = $1 AND payload @> $2::jsonb'
            ')',
            node.group_id,
            uuid_json,
        )
        await client._execute(
            'DELETE FROM "next_episode_edges" '
            'WHERE realm = $1 AND ('
            '  from_id IN ('
            '    SELECT id FROM "episodic_nodes" '
            '    WHERE realm = $1 AND payload @> $2::jsonb'
            '  ) OR to_id IN ('
            '    SELECT id FROM "episodic_nodes" '
            '    WHERE realm = $1 AND payload @> $2::jsonb'
            '  )'
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
            'DELETE FROM "episodic_edges" '
            'WHERE realm = $1 AND from_id IN ('
            '  SELECT id FROM "episodic_nodes" '
            '  WHERE realm = $1'
            ')',
            group_id,
        )
        await client._execute(
            'DELETE FROM "has_episode_edges" '
            'WHERE realm = $1 AND to_id IN ('
            '  SELECT id FROM "episodic_nodes" '
            '  WHERE realm = $1'
            ')',
            group_id,
        )
        await client._execute(
            'DELETE FROM "next_episode_edges" '
            'WHERE realm = $1 AND ('
            '  from_id IN ('
            '    SELECT id FROM "episodic_nodes" '
            '    WHERE realm = $1'
            '  ) OR to_id IN ('
            '    SELECT id FROM "episodic_nodes" '
            '    WHERE realm = $1'
            '  )'
            ')',
            group_id,
        )
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
            id_arr = ids
            await client._execute(
                'DELETE FROM "episodic_edges" WHERE realm = $1 AND from_id = ANY($2)',
                realm,
                id_arr,
            )
            await client._execute(
                'DELETE FROM "has_episode_edges" WHERE realm = $1 AND to_id = ANY($2)',
                realm,
                id_arr,
            )
            await client._execute(
                'DELETE FROM "next_episode_edges" '
                'WHERE realm = $1 AND '
                '(from_id = ANY($2) OR to_id = ANY($2))',
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
    ) -> EpisodicNode:
        client = executor.client
        rows = await client._fetch(
            'SELECT realm, id, space, fqid, payload, '
            'created_at, updated_at, '
            'uuid::text AS uuid_text '
            f'FROM "{TABLE}" t '
            'WHERE payload @> $1::jsonb',
            json.dumps({'uuid': uuid}),
        )
        if not rows:
            raise NodeNotFoundError(uuid)
        return _row_to_episodic_node(dict(rows[0]))

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[EpisodicNode]:
        if not uuids:
            return []
        client = executor.client
        placeholders = ' OR '.join(f'payload @> ${i + 1}::jsonb' for i in range(len(uuids)))
        uuid_json_list = [json.dumps({'uuid': u}) for u in uuids]
        rows = await client._fetch(
            'SELECT realm, id, space, fqid, payload, '
            'created_at, updated_at, '
            'uuid::text AS uuid_text '
            f'FROM "{TABLE}" t '
            f'WHERE {placeholders}',
            *uuid_json_list,
        )
        return [_row_to_episodic_node(dict(r)) for r in rows]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EpisodicNode]:
        if not group_ids:
            return []
        client = executor.client
        results: list[EpisodicNode] = []
        for realm in group_ids:
            vertices = await client.get_vertices(
                TABLE,
                realm,
            )
            for v in vertices:
                results.append(_vertex_to_episodic_node(v))

        results.sort(
            key=lambda n: n.uuid,
            reverse=True,
        )

        if uuid_cursor:
            results = [n for n in results if n.uuid < uuid_cursor]
        if limit is not None:
            results = results[:limit]
        return results

    async def get_by_entity_node_uuid(
        self,
        executor: QueryExecutor,
        entity_node_uuid: str,
    ) -> list[EpisodicNode]:
        client = executor.client
        # Find episodes linked to entity via episodic_edges
        rows = await client._fetch(
            'SELECT DISTINCT e.realm, e.id, e.space, '
            'e.fqid, e.payload, '
            'e.created_at, e.updated_at, '
            'e.uuid::text AS uuid_text '
            'FROM "episodic_nodes" e '
            'JOIN "episodic_edges" ee '
            '  ON ee.from_id = e.id AND ee.realm = e.realm '
            'JOIN "entity_nodes" en '
            '  ON ee.to_id = en.id AND ee.realm = en.realm '
            'WHERE en.payload @> $1::jsonb',
            json.dumps({'uuid': entity_node_uuid}),
        )
        return [_row_to_episodic_node(dict(r)) for r in rows]

    async def retrieve_episodes(
        self,
        executor: QueryExecutor,
        reference_time: datetime,
        last_n: int = 3,
        group_ids: list[str] | None = None,
        source: str | None = None,
        saga: str | None = None,
    ) -> list[EpisodicNode]:
        client = executor.client
        ref_iso = reference_time.isoformat()
        params: list[Any] = [ref_iso]

        # `source` is annotated `str` but Graphiti's own callers pass the
        # EpisodeType enum, and both filter branches below json.dumps it — which
        # raises TypeError on the first add_episode(). The save path already
        # coerces with `.value`; this makes the read path agree, and accepts
        # either form so a caller of any vintage works.
        if isinstance(source, Enum):
            source = source.value

        if saga:
            # Build query joining through has_episode_edges
            # to filter by saga node
            conditions = [
                "(e.payload->>'valid_at') <= $1",
            ]
            if group_ids:
                realm_ph = ', '.join(f'${i + len(params) + 1}' for i in range(len(group_ids)))
                conditions.append(f'e.realm IN ({realm_ph})')
                params.extend(group_ids)
            if source:
                params.append(json.dumps({'source': source}))
                conditions.append(f'e.payload @> ${len(params)}::jsonb')

            params.append(json.dumps({'uuid': saga}))
            saga_param = f'${len(params)}'
            where = ' AND '.join(conditions)

            query = (
                'SELECT e.realm, e.id, e.space, '
                'e.fqid, e.payload, '
                'e.created_at, e.updated_at, '
                'e.uuid::text AS uuid_text '
                'FROM "episodic_nodes" e '
                'JOIN "has_episode_edges" he '
                '  ON he.to_id = e.id '
                '  AND he.realm = e.realm '
                'JOIN "saga_nodes" s '
                '  ON he.from_id = s.id '
                '  AND he.realm = s.realm '
                f'WHERE s.payload @> '
                f'{saga_param}::jsonb '
                f'AND {where} '
                "ORDER BY e.payload->>'valid_at' "
                f'DESC LIMIT {last_n}'
            )
        else:
            conditions = [
                "(payload->>'valid_at') <= $1",
            ]
            if group_ids:
                realm_ph = ', '.join(f'${i + len(params) + 1}' for i in range(len(group_ids)))
                conditions.append(f'realm IN ({realm_ph})')
                params.extend(group_ids)
            if source:
                params.append(json.dumps({'source': source}))
                conditions.append(f'payload @> ${len(params)}::jsonb')

            where = ' AND '.join(conditions)
            query = (
                'SELECT realm, id, space, fqid, '
                'payload, created_at, updated_at, '
                'uuid::text AS uuid_text '
                f'FROM "{TABLE}" '
                f'WHERE {where} '
                "ORDER BY payload->>'valid_at' "
                f'DESC LIMIT {last_n}'
            )

        rows = await client._fetch(query, *params)
        return [_row_to_episodic_node(dict(r)) for r in rows]


def _vertex_to_episodic_node(v: Vertex) -> EpisodicNode:
    """Convert a post-graph Vertex to an EpisodicNode."""
    p = v.payload or {}
    created_at = parse_db_date(p.get('created_at'))
    valid_at = parse_db_date(p.get('valid_at'))

    if created_at is None:
        raise ValueError(f'created_at cannot be None for episode {p.get("uuid", "unknown")}')
    if valid_at is None:
        raise ValueError(f'valid_at cannot be None for episode {p.get("uuid", "unknown")}')

    return EpisodicNode(
        content=p.get('content', ''),
        created_at=created_at,
        valid_at=valid_at,
        uuid=p.get('uuid', ''),
        group_id=v.realm or '',
        source=EpisodeType.from_str(p.get('source', 'text')),
        name=p.get('name', ''),
        source_description=p.get('source_description', ''),
        entity_edges=list(p.get('entity_edges', []) or []),
    )


def _row_to_episodic_node(row: dict) -> EpisodicNode:
    """Convert a raw DB row to an EpisodicNode."""
    p = row.get('payload', {})
    if isinstance(p, str):
        p = json.loads(p)

    created_at = parse_db_date(p.get('created_at'))
    valid_at = parse_db_date(p.get('valid_at'))

    if created_at is None:
        raise ValueError(f'created_at cannot be None for episode {p.get("uuid", "unknown")}')
    if valid_at is None:
        raise ValueError(f'valid_at cannot be None for episode {p.get("uuid", "unknown")}')

    return EpisodicNode(
        content=p.get('content', ''),
        created_at=created_at,
        valid_at=valid_at,
        uuid=p.get('uuid', ''),
        group_id=row.get('realm', ''),
        source=EpisodeType.from_str(p.get('source', 'text')),
        name=p.get('name', ''),
        source_description=p.get('source_description', ''),
        entity_edges=list(p.get('entity_edges', []) or []),
    )


def _as_episodic_node(node: Any) -> Any:
    """Normalise a bulk-path dict into something save() can read.

    A SimpleNamespace rather than a real EpisodicNode: the dict has already
    been through `dict(episode)` and had `source` stringified, so re-validating
    it through the model would reject its own output.
    """
    if not isinstance(node, dict):
        return node
    from types import SimpleNamespace

    data = dict(node)
    for key in ('valid_at', 'created_at'):
        value = data.get(key)
        if isinstance(value, str):
            from datetime import datetime

            try:
                data[key] = datetime.fromisoformat(value)
            except ValueError:
                data[key] = None
    data.setdefault('entity_edges', [])
    return SimpleNamespace(**data)
