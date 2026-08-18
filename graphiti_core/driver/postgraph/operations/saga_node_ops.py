from __future__ import annotations

import json
import logging
from typing import Any

from post_graph import Vertex

from graphiti_core.driver.operations.saga_node_ops import SagaNodeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.helpers import parse_db_date
from graphiti_core.nodes import SagaNode

logger = logging.getLogger(__name__)

TABLE = 'saga_nodes'


class PGSagaNodeOperations(SagaNodeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        node: SagaNode,
        _tx: Transaction | None = None,
    ) -> None:
        client = executor.client
        payload = {
            'uuid': node.uuid,
            'name': node.name,
            'summary': node.summary,
            'first_episode_uuid': node.first_episode_uuid,
            'last_episode_uuid': node.last_episode_uuid,
            'last_summarized_at': (
                node.last_summarized_at.isoformat()
                if node.last_summarized_at
                else None
            ),
            'last_summarized_episode_valid_at': (
                node.last_summarized_episode_valid_at.isoformat()
                if node.last_summarized_episode_valid_at
                else None
            ),
            'created_at': (
                node.created_at.isoformat()
                if node.created_at
                else None
            ),
        }

        v_id = await executor._resolve_vertex_id(
            TABLE, node.group_id, node.uuid,
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
        nodes: list[SagaNode],
        _tx: Transaction | None = None,
        _batch_size: int = 100,
    ) -> None:
        for node in nodes:
            await self.save(executor, node)

    async def delete(
        self,
        executor: QueryExecutor,
        node: SagaNode,
        _tx: Transaction | None = None,
    ) -> None:
        client = executor.client
        uuid_json = json.dumps({'uuid': node.uuid})
        # Delete referencing has_episode_edges first
        await client._execute(
            'DELETE FROM "has_episode_edges" '
            'WHERE realm = $1 AND from_id IN ('
            '  SELECT id FROM "saga_nodes" '
            '  WHERE realm = $1 AND payload @> $2::jsonb'
            ')',
            node.group_id, uuid_json,
        )
        v_id = await executor._resolve_vertex_id(
            TABLE, node.group_id, node.uuid,
        )
        if v_id is not None:
            await client.delete_vertex(
                TABLE, node.group_id, str(v_id),
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
            'DELETE FROM "has_episode_edges" '
            'WHERE realm = $1 AND from_id IN ('
            '  SELECT id FROM "saga_nodes" '
            '  WHERE realm = $1'
            ')',
            group_id,
        )
        vertices = await client.get_vertices(
            TABLE, group_id,
        )
        for v in vertices:
            await client.delete_vertex(
                TABLE, group_id, str(v.id),
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
        uuid_json_list = [
            json.dumps({'uuid': u}) for u in uuids
        ]
        placeholders = ' OR '.join(
            f'payload @> ${i + 1}::jsonb'
            for i in range(len(uuids))
        )

        id_rows = await client._fetch(
            f'SELECT realm, id FROM "{TABLE}" '
            f'WHERE {placeholders}',
            *uuid_json_list,
        )
        if not id_rows:
            return

        realm_ids: dict[str, list[int]] = {}
        for row in id_rows:
            realm_ids.setdefault(row['realm'], []).append(
                row['id']
            )

        for realm, ids in realm_ids.items():
            await client._execute(
                'DELETE FROM "has_episode_edges" '
                'WHERE realm = $1 AND '
                'from_id = ANY($2)',
                realm, ids,
            )
            for vid in ids:
                await client.delete_vertex(
                    TABLE, realm, str(vid),
                )

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> SagaNode:
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
        return _row_to_saga_node(dict(rows[0]))

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[SagaNode]:
        if not uuids:
            return []
        client = executor.client
        placeholders = ' OR '.join(
            f'payload @> ${i + 1}::jsonb'
            for i in range(len(uuids))
        )
        uuid_json_list = [
            json.dumps({'uuid': u}) for u in uuids
        ]
        rows = await client._fetch(
            'SELECT realm, id, space, fqid, payload, '
            'created_at, updated_at, '
            'uuid::text AS uuid_text '
            f'FROM "{TABLE}" t '
            f'WHERE {placeholders}',
            *uuid_json_list,
        )
        return [
            _row_to_saga_node(dict(r)) for r in rows
        ]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[SagaNode]:
        if not group_ids:
            return []
        client = executor.client
        results: list[SagaNode] = []
        for realm in group_ids:
            vertices = await client.get_vertices(
                TABLE, realm,
            )
            for v in vertices:
                results.append(
                    _vertex_to_saga_node(v)
                )

        results.sort(
            key=lambda n: n.uuid, reverse=True,
        )

        if uuid_cursor:
            results = [
                n for n in results
                if n.uuid < uuid_cursor
            ]
        if limit is not None:
            results = results[:limit]
        return results


def _vertex_to_saga_node(v: Vertex) -> SagaNode:
    """Convert a post-graph Vertex to a SagaNode."""
    p = v.payload or {}
    last_summarized_at = p.get('last_summarized_at')
    last_ep_valid_at = p.get(
        'last_summarized_episode_valid_at'
    )
    return SagaNode(
        uuid=p.get('uuid', ''),
        name=p.get('name', ''),
        group_id=v.realm or '',
        created_at=parse_db_date(p.get('created_at')),
        summary=p.get('summary', '') or '',
        first_episode_uuid=p.get('first_episode_uuid'),
        last_episode_uuid=p.get('last_episode_uuid'),
        last_summarized_at=(
            parse_db_date(last_summarized_at)
            if last_summarized_at
            else None
        ),
        last_summarized_episode_valid_at=(
            parse_db_date(last_ep_valid_at)
            if last_ep_valid_at
            else None
        ),
    )


def _row_to_saga_node(row: dict) -> SagaNode:
    """Convert a raw DB row to a SagaNode."""
    p = row.get('payload', {})
    if isinstance(p, str):
        p = json.loads(p)

    last_summarized_at = p.get('last_summarized_at')
    last_ep_valid_at = p.get(
        'last_summarized_episode_valid_at'
    )
    return SagaNode(
        uuid=p.get('uuid', ''),
        name=p.get('name', ''),
        group_id=row.get('realm', ''),
        created_at=parse_db_date(p.get('created_at')),
        summary=p.get('summary', '') or '',
        first_episode_uuid=p.get('first_episode_uuid'),
        last_episode_uuid=p.get('last_episode_uuid'),
        last_summarized_at=(
            parse_db_date(last_summarized_at)
            if last_summarized_at
            else None
        ),
        last_summarized_episode_valid_at=(
            parse_db_date(last_ep_valid_at)
            if last_ep_valid_at
            else None
        ),
    )
