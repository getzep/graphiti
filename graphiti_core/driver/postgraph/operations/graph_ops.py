from __future__ import annotations

import json
import logging
from typing import Any

from post_graph import AsyncPostGraph, Vertex

from graphiti_core.driver.operations.graph_ops import GraphMaintenanceOperations
from graphiti_core.driver.operations.graph_utils import Neighbor, label_propagation
from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.helpers import parse_db_date
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode

logger = logging.getLogger(__name__)

VERTEX_TABLES = ['entity_nodes', 'episodic_nodes', 'community_nodes', 'saga_nodes']

EDGE_DEFS = [
    {
        'name': 'entity_edges',
        'from': 'entity_nodes',
        'to': 'entity_nodes',
        'vector': True,
    },
    {'name': 'episodic_edges', 'from': 'episodic_nodes', 'to': 'entity_nodes'},
    {'name': 'community_edges', 'from': 'community_nodes', 'to': 'entity_nodes'},
    {'name': 'has_episode_edges', 'from': 'saga_nodes', 'to': 'episodic_nodes'},
    {'name': 'next_episode_edges', 'from': 'episodic_nodes', 'to': 'episodic_nodes'},
]

ALL_TABLES = VERTEX_TABLES + [e['name'] for e in EDGE_DEFS]


def _tsvector_ddl() -> list[str]:
    """Graphiti-specific generated tsvector columns for fulltext search on payload."""
    return [
        """ALTER TABLE "entity_nodes" ADD COLUMN IF NOT EXISTS search_vector TSVECTOR
           GENERATED ALWAYS AS (
               to_tsvector('simple',
                   coalesce(payload->>'name', '') || ' ' || coalesce(payload->>'summary', ''))
           ) STORED""",
        """ALTER TABLE "episodic_nodes" ADD COLUMN IF NOT EXISTS search_vector TSVECTOR
           GENERATED ALWAYS AS (
               to_tsvector('simple', coalesce(payload->>'content', ''))
           ) STORED""",
        """ALTER TABLE "community_nodes" ADD COLUMN IF NOT EXISTS search_vector TSVECTOR
           GENERATED ALWAYS AS (
               to_tsvector('simple',
                   coalesce(payload->>'name', '') || ' ' || coalesce(payload->>'summary', ''))
           ) STORED""",
        """ALTER TABLE "entity_edges" ADD COLUMN IF NOT EXISTS search_vector TSVECTOR
           GENERATED ALWAYS AS (
               to_tsvector('simple',
                   coalesce(payload->>'name', '') || ' ' || coalesce(payload->>'fact', ''))
           ) STORED""",
    ]


def _extra_index_ddl() -> list[str]:
    """Additional indexes beyond what post-graph creates by default."""
    return [
        'CREATE INDEX IF NOT EXISTS idx_entity_nodes_search ON "entity_nodes" USING GIN (search_vector)',
        'CREATE INDEX IF NOT EXISTS idx_episodic_nodes_search ON "episodic_nodes" USING GIN (search_vector)',
        'CREATE INDEX IF NOT EXISTS idx_community_nodes_search ON "community_nodes" USING GIN (search_vector)',
        'CREATE INDEX IF NOT EXISTS idx_entity_edges_search ON "entity_edges" USING GIN (search_vector)',
        'CREATE INDEX IF NOT EXISTS idx_episodic_nodes_valid_at ON "episodic_nodes" ((payload->>\'valid_at\'))',
    ]


class PGGraphMaintenanceOperations(GraphMaintenanceOperations):
    def __init__(self, embedding_dim: int = 1024):
        self._embedding_dim = embedding_dim

    async def build_indices_and_constraints_pg(
        self, client: AsyncPostGraph, embedding_dim: int,
    ) -> None:
        from post_graph.errors import TableExistsError

        for vt in VERTEX_TABLES:
            vdim = embedding_dim if vt in ('entity_nodes', 'community_nodes') else None
            try:
                await client.create_vertex_table(vt, vector_dim=vdim)
            except (TableExistsError, Exception):
                pass

        for edef in EDGE_DEFS:
            vdim = embedding_dim if edef.get('vector') else None
            try:
                await client.create_edge_table(
                    edef['name'],
                    from_vertex_table=edef['from'],
                    to_vertex_table=edef['to'],
                    vector_dim=vdim,
                    cascade_delete_from=True,
                    cascade_delete_to=True,
                )
            except (TableExistsError, Exception):
                pass

        for stmt in _tsvector_ddl():
            try:
                await client._execute(stmt)
            except Exception:
                pass

        for stmt in _extra_index_ddl():
            try:
                await client._execute(stmt)
            except Exception:
                pass

    async def build_indices_and_constraints_raw(self, _conn) -> None:
        pass

    async def delete_all_indexes_raw(self, _conn) -> None:
        pass

    async def clear_data(
        self,
        executor: QueryExecutor,
        group_ids: list[str] | None = None,
    ) -> None:
        client = executor.client
        if group_ids is None:
            for table in ALL_TABLES:
                try:
                    await client._execute(f'DELETE FROM "{table}"')
                except Exception:
                    pass
        else:
            for gid in group_ids:
                await client.delete_realm(gid)

    async def build_indices_and_constraints(
        self,
        executor: QueryExecutor,
        delete_existing: bool = False,  # noqa: ARG002
    ) -> None:
        client = executor.client
        await self.build_indices_and_constraints_pg(client, self._embedding_dim)

    async def delete_all_indexes(
        self,
        executor: QueryExecutor,
    ) -> None:
        client = executor.client
        rows = await client._fetch(
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'public' AND indexname LIKE 'idx_%'"
        )
        for r in rows:
            await client._execute(f'DROP INDEX IF EXISTS {r["indexname"]}')

    async def get_community_clusters(
        self,
        executor: QueryExecutor,
        group_ids: list[str] | None = None,
    ) -> list[Any]:
        client = executor.client
        community_clusters: list[list[EntityNode]] = []

        if group_ids is None:
            all_verts = await client._fetch(
                'SELECT DISTINCT realm FROM "entity_nodes" WHERE realm IS NOT NULL'
            )
            group_ids = [r['realm'] for r in all_verts]

        resolved_group_ids: list[str] = group_ids or []
        for group_id in resolved_group_ids:
            projection: dict[str, list[Neighbor]] = {}

            node_verts = await client.get_vertices('entity_nodes', group_id)
            nodes = [_vertex_to_entity_node(v) for v in node_verts]

            for node in nodes:
                v_id = await executor._resolve_vertex_id('entity_nodes', group_id, node.uuid)
                if v_id is None:
                    continue
                rows = await client._fetch(
                    """
                    SELECT
                        CASE WHEN from_id = $1 THEN to_id ELSE from_id END AS other_id,
                        count(*) AS count
                    FROM "entity_edges"
                    WHERE (from_id = $1 OR to_id = $1) AND realm = $2
                    GROUP BY 1
                    """,
                    v_id, group_id,
                )
                neighbors = []
                for r in rows:
                    other_rows = await client._fetch(
                        'SELECT payload->>\'uuid\' AS uuid FROM "entity_nodes" WHERE realm = $1 AND id = $2',
                        group_id, r['other_id'],
                    )
                    if other_rows:
                        neighbors.append(
                            Neighbor(node_uuid=other_rows[0]['uuid'], edge_count=r['count'])
                        )
                projection[node.uuid] = neighbors

            cluster_uuids = label_propagation(projection)

            for cluster in cluster_uuids:
                if not cluster:
                    continue
                cluster_nodes = []
                for uuid in cluster:
                    v_rows = await client._fetch(
                        'SELECT realm, id, space, fqid, payload, created_at, updated_at, '
                        'uuid::text AS uuid_text, to_jsonb(t)->>\'embedding\' AS embedding_text '
                        'FROM "entity_nodes" t WHERE realm = $1 AND payload @> $2::jsonb',
                        group_id, json.dumps({'uuid': uuid}),
                    )
                    if v_rows:
                        cluster_nodes.append(_row_to_entity_node(v_rows[0]))
                if cluster_nodes:
                    community_clusters.append(cluster_nodes)

        return community_clusters

    async def remove_communities(
        self,
        executor: QueryExecutor,
        group_ids: list[str] | None = None,
    ) -> None:
        client = executor.client
        if group_ids:
            for gid in group_ids:
                await client._execute(
                    'DELETE FROM "community_edges" WHERE realm = $1', gid
                )
                await client._execute(
                    'DELETE FROM "community_nodes" WHERE realm = $1', gid
                )
        else:
            await client._execute('DELETE FROM "community_edges"')
            await client._execute('DELETE FROM "community_nodes"')

    async def determine_entity_community(
        self,
        executor: QueryExecutor,
        entity: EntityNode,
    ) -> None:
        client = executor.client
        v_id = await executor._resolve_vertex_id('entity_nodes', entity.group_id, entity.uuid)
        if v_id is None:
            return

        rows = await client._fetch(
            """
            SELECT c.payload FROM "community_nodes" c
            JOIN "community_edges" ce ON ce.from_id = c.id AND ce.realm = c.realm
            WHERE ce.to_id = $1 AND c.realm = $2
            """,
            v_id, entity.group_id,
        )
        if rows:
            return

    async def get_mentioned_nodes(
        self,
        executor: QueryExecutor,
        episodes: list[EpisodicNode],
    ) -> list[EntityNode]:
        client = executor.client
        results = []
        group_ids = list({ep.group_id for ep in episodes})

        for ep in episodes:
            ep_vid = await executor._resolve_vertex_id('episodic_nodes', ep.group_id, ep.uuid)
            if ep_vid is None:
                continue
            rows = await client._fetch(
                """
                SELECT DISTINCT n.realm, n.id, n.space, n.fqid, n.payload,
                       n.created_at, n.updated_at,
                       n.uuid::text AS uuid_text,
                       to_jsonb(n)->>'embedding' AS embedding_text
                FROM "entity_nodes" n
                JOIN "episodic_edges" ee ON ee.to_id = n.id AND ee.realm = n.realm
                WHERE ee.from_id = $1 AND n.realm = ANY($2)
                """,
                ep_vid, group_ids,
            )
            for r in rows:
                node = _row_to_entity_node(r)
                if not any(n.uuid == node.uuid for n in results):
                    results.append(node)
        return results

    async def get_communities_by_nodes(
        self,
        executor: QueryExecutor,
        nodes: list[EntityNode],
    ) -> list[CommunityNode]:
        client = executor.client
        results = []
        group_ids = list({n.group_id for n in nodes})

        for node in nodes:
            v_id = await executor._resolve_vertex_id('entity_nodes', node.group_id, node.uuid)
            if v_id is None:
                continue
            rows = await client._fetch(
                """
                SELECT DISTINCT c.realm, c.id, c.space, c.fqid, c.payload,
                       c.created_at, c.updated_at,
                       c.uuid::text AS uuid_text,
                       to_jsonb(c)->>'embedding' AS embedding_text
                FROM "community_nodes" c
                JOIN "community_edges" ce ON ce.from_id = c.id AND ce.realm = c.realm
                WHERE ce.to_id = $1 AND c.realm = ANY($2)
                """,
                v_id, group_ids,
            )
            for r in rows:
                cn = _row_to_community_node(r)
                if not any(c.uuid == cn.uuid for c in results):
                    results.append(cn)
        return results


def _vertex_to_entity_node(v: Vertex) -> EntityNode:
    p = v.payload
    labels = list(p.get('labels', []))
    group_id = v.realm
    dynamic_label = 'Entity_' + group_id.replace('-', '')
    if dynamic_label in labels:
        labels.remove(dynamic_label)

    attributes = dict(p.get('attributes', {}))
    return EntityNode(
        uuid=p['uuid'],
        name=p.get('name', ''),
        name_embedding=v.embedding,
        group_id=group_id,
        labels=labels,
        created_at=parse_db_date(p.get('created_at')) or v.created_at,
        summary=p.get('summary', ''),
        attributes=attributes,
    )


def _row_to_entity_node(row: dict) -> EntityNode:
    p = row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload'])
    labels = list(p.get('labels', []))
    group_id = row['realm']
    dynamic_label = 'Entity_' + group_id.replace('-', '')
    if dynamic_label in labels:
        labels.remove(dynamic_label)

    emb = None
    if row.get('embedding_text'):
        emb = [float(x) for x in row['embedding_text'].strip('[]').split(',') if x.strip()]

    attributes = dict(p.get('attributes', {}))
    return EntityNode(
        uuid=p['uuid'],
        name=p.get('name', ''),
        name_embedding=emb,
        group_id=group_id,
        labels=labels,
        created_at=parse_db_date(p.get('created_at')) or row.get('created_at'),
        summary=p.get('summary', ''),
        attributes=attributes,
    )


def _row_to_community_node(row: dict) -> CommunityNode:
    p = row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload'])
    emb = None
    if row.get('embedding_text'):
        emb = [float(x) for x in row['embedding_text'].strip('[]').split(',') if x.strip()]

    return CommunityNode(
        uuid=p['uuid'],
        name=p.get('name', ''),
        group_id=row['realm'],
        name_embedding=emb,
        created_at=parse_db_date(p.get('created_at')) or row.get('created_at'),
        summary=p.get('summary', ''),
    )
