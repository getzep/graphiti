from __future__ import annotations

import logging
from typing import Any

from graphiti_core.driver.operations.graph_ops import GraphMaintenanceOperations
from graphiti_core.driver.operations.graph_utils import Neighbor, label_propagation
from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.driver.record_parsers import community_node_from_record
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode

logger = logging.getLogger(__name__)

TABLES = [
    'entity_nodes',
    'episodic_nodes',
    'community_nodes',
    'saga_nodes',
    'entity_edges',
    'episodic_edges',
    'community_edges',
    'has_episode_edges',
    'next_episode_edges',
]


def _ddl(embedding_dim: int) -> list[str]:
    return [
        'CREATE EXTENSION IF NOT EXISTS vector',
        f"""
        CREATE TABLE IF NOT EXISTS entity_nodes (
            uuid            TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            group_id        TEXT NOT NULL,
            labels          TEXT[] NOT NULL DEFAULT '{{}}',
            summary         TEXT NOT NULL DEFAULT '',
            name_embedding  vector({embedding_dim}),
            attributes      JSONB NOT NULL DEFAULT '{{}}',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            search_vector   TSVECTOR GENERATED ALWAYS AS (
                to_tsvector('simple', coalesce(name, '') || ' ' || coalesce(summary, ''))
            ) STORED
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS episodic_nodes (
            uuid                TEXT PRIMARY KEY,
            name                TEXT NOT NULL,
            group_id            TEXT NOT NULL,
            source              TEXT NOT NULL,
            source_description  TEXT NOT NULL DEFAULT '',
            content             TEXT NOT NULL DEFAULT '',
            valid_at            TIMESTAMPTZ NOT NULL,
            entity_edges        TEXT[] NOT NULL DEFAULT '{{}}',
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            search_vector       TSVECTOR GENERATED ALWAYS AS (
                to_tsvector('simple', coalesce(content, ''))
            ) STORED
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS community_nodes (
            uuid            TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            group_id        TEXT NOT NULL,
            summary         TEXT NOT NULL DEFAULT '',
            name_embedding  vector({embedding_dim}),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            search_vector   TSVECTOR GENERATED ALWAYS AS (
                to_tsvector('simple', coalesce(name, '') || ' ' || coalesce(summary, ''))
            ) STORED
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS saga_nodes (
            uuid                            TEXT PRIMARY KEY,
            name                            TEXT NOT NULL,
            group_id                        TEXT NOT NULL,
            summary                         TEXT NOT NULL DEFAULT '',
            first_episode_uuid              TEXT,
            last_episode_uuid               TEXT,
            last_summarized_at              TIMESTAMPTZ,
            last_summarized_episode_valid_at TIMESTAMPTZ,
            created_at                      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS entity_edges (
            uuid                TEXT PRIMARY KEY,
            source_node_uuid    TEXT NOT NULL,
            target_node_uuid    TEXT NOT NULL,
            name                TEXT NOT NULL,
            fact                TEXT NOT NULL DEFAULT '',
            fact_embedding      vector({embedding_dim}),
            group_id            TEXT NOT NULL,
            episodes            TEXT[] NOT NULL DEFAULT '{{}}',
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            expired_at          TIMESTAMPTZ,
            valid_at            TIMESTAMPTZ,
            invalid_at          TIMESTAMPTZ,
            reference_time      TIMESTAMPTZ,
            attributes          JSONB NOT NULL DEFAULT '{{}}',
            search_vector       TSVECTOR GENERATED ALWAYS AS (
                to_tsvector('simple', coalesce(name, '') || ' ' || coalesce(fact, ''))
            ) STORED
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS episodic_edges (
            uuid                TEXT PRIMARY KEY,
            source_node_uuid    TEXT NOT NULL,
            target_node_uuid    TEXT NOT NULL,
            group_id            TEXT NOT NULL,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS community_edges (
            uuid                TEXT PRIMARY KEY,
            source_node_uuid    TEXT NOT NULL,
            target_node_uuid    TEXT NOT NULL,
            group_id            TEXT NOT NULL,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS has_episode_edges (
            uuid                TEXT PRIMARY KEY,
            source_node_uuid    TEXT NOT NULL,
            target_node_uuid    TEXT NOT NULL,
            group_id            TEXT NOT NULL,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS next_episode_edges (
            uuid                TEXT PRIMARY KEY,
            source_node_uuid    TEXT NOT NULL,
            target_node_uuid    TEXT NOT NULL,
            group_id            TEXT NOT NULL,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
    ]


def _index_ddl(embedding_dim: int) -> list[str]:
    return [
        'CREATE INDEX IF NOT EXISTS idx_entity_nodes_group_id ON entity_nodes (group_id)',
        'CREATE INDEX IF NOT EXISTS idx_entity_nodes_search ON entity_nodes USING GIN (search_vector)',
        'CREATE INDEX IF NOT EXISTS idx_entity_nodes_embedding ON entity_nodes USING hnsw (name_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)',
        'CREATE INDEX IF NOT EXISTS idx_episodic_nodes_group_id ON episodic_nodes (group_id)',
        'CREATE INDEX IF NOT EXISTS idx_episodic_nodes_search ON episodic_nodes USING GIN (search_vector)',
        'CREATE INDEX IF NOT EXISTS idx_episodic_nodes_valid_at ON episodic_nodes (valid_at DESC)',
        'CREATE INDEX IF NOT EXISTS idx_community_nodes_group_id ON community_nodes (group_id)',
        'CREATE INDEX IF NOT EXISTS idx_community_nodes_search ON community_nodes USING GIN (search_vector)',
        'CREATE INDEX IF NOT EXISTS idx_community_nodes_embedding ON community_nodes USING hnsw (name_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)',
        'CREATE INDEX IF NOT EXISTS idx_saga_nodes_group_id ON saga_nodes (group_id)',
        'CREATE INDEX IF NOT EXISTS idx_entity_edges_group_id ON entity_edges (group_id)',
        'CREATE INDEX IF NOT EXISTS idx_entity_edges_source ON entity_edges (source_node_uuid)',
        'CREATE INDEX IF NOT EXISTS idx_entity_edges_target ON entity_edges (target_node_uuid)',
        'CREATE INDEX IF NOT EXISTS idx_entity_edges_search ON entity_edges USING GIN (search_vector)',
        'CREATE INDEX IF NOT EXISTS idx_entity_edges_embedding ON entity_edges USING hnsw (fact_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)',
        'CREATE INDEX IF NOT EXISTS idx_episodic_edges_source ON episodic_edges (source_node_uuid)',
        'CREATE INDEX IF NOT EXISTS idx_episodic_edges_target ON episodic_edges (target_node_uuid)',
        'CREATE INDEX IF NOT EXISTS idx_episodic_edges_group_id ON episodic_edges (group_id)',
        'CREATE INDEX IF NOT EXISTS idx_community_edges_source ON community_edges (source_node_uuid)',
        'CREATE INDEX IF NOT EXISTS idx_community_edges_target ON community_edges (target_node_uuid)',
        'CREATE INDEX IF NOT EXISTS idx_community_edges_group_id ON community_edges (group_id)',
        'CREATE INDEX IF NOT EXISTS idx_has_episode_edges_source ON has_episode_edges (source_node_uuid)',
        'CREATE INDEX IF NOT EXISTS idx_has_episode_edges_target ON has_episode_edges (target_node_uuid)',
        'CREATE INDEX IF NOT EXISTS idx_has_episode_edges_group_id ON has_episode_edges (group_id)',
        'CREATE INDEX IF NOT EXISTS idx_next_episode_edges_source ON next_episode_edges (source_node_uuid)',
        'CREATE INDEX IF NOT EXISTS idx_next_episode_edges_target ON next_episode_edges (target_node_uuid)',
        'CREATE INDEX IF NOT EXISTS idx_next_episode_edges_group_id ON next_episode_edges (group_id)',
    ]


class PGGraphMaintenanceOperations(GraphMaintenanceOperations):
    def __init__(self, embedding_dim: int = 1024):
        self._embedding_dim = embedding_dim

    async def build_indices_and_constraints_raw(self, conn) -> None:
        for stmt in _ddl(self._embedding_dim):
            await conn.execute(stmt)
        for stmt in _index_ddl(self._embedding_dim):
            await conn.execute(stmt)

    async def delete_all_indexes_raw(self, conn) -> None:
        rows = await conn.fetch(
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'public' AND indexname LIKE 'idx_%'"
        )
        for row in rows:
            await conn.execute(f'DROP INDEX IF EXISTS {row["indexname"]}')

    async def clear_data(
        self,
        executor: QueryExecutor,
        group_ids: list[str] | None = None,
    ) -> None:
        if group_ids is None:
            for table in TABLES:
                await executor.execute_query(f'DELETE FROM {table}')
        else:
            for table in TABLES:
                await executor.execute_query(
                    f'DELETE FROM {table} WHERE group_id = ANY($group_ids)',
                    group_ids=group_ids,
                )

    async def build_indices_and_constraints(
        self,
        executor: QueryExecutor,
        delete_existing: bool = False,
    ) -> None:
        if delete_existing:
            await self.delete_all_indexes(executor)
        for stmt in _ddl(self._embedding_dim):
            await executor.execute_query(stmt)
        for stmt in _index_ddl(self._embedding_dim):
            await executor.execute_query(stmt)

    async def delete_all_indexes(
        self,
        executor: QueryExecutor,
    ) -> None:
        records, _, _ = await executor.execute_query(
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'public' AND indexname LIKE 'idx_%'"
        )
        for r in records:
            await executor.execute_query(f'DROP INDEX IF EXISTS {r["indexname"]}')

    async def get_community_clusters(
        self,
        executor: QueryExecutor,
        group_ids: list[str] | None = None,
    ) -> list[Any]:
        community_clusters: list[list[EntityNode]] = []

        if group_ids is None:
            records, _, _ = await executor.execute_query(
                'SELECT DISTINCT group_id FROM entity_nodes WHERE group_id IS NOT NULL'
            )
            group_ids = [r['group_id'] for r in records]

        resolved_group_ids: list[str] = group_ids or []
        for group_id in resolved_group_ids:
            projection: dict[str, list[Neighbor]] = {}

            node_records, _, _ = await executor.execute_query(
                """
                SELECT uuid, name, group_id, labels, summary, attributes, created_at
                FROM entity_nodes
                WHERE group_id = $group_id
                """,
                group_id=group_id,
            )
            nodes = [_entity_node_from_pg(r) for r in node_records]

            for node in nodes:
                records, _, _ = await executor.execute_query(
                    """
                    SELECT
                        CASE WHEN source_node_uuid = $uuid THEN target_node_uuid
                             ELSE source_node_uuid END AS uuid,
                        count(*) AS count
                    FROM entity_edges
                    WHERE (source_node_uuid = $uuid OR target_node_uuid = $uuid)
                      AND group_id = $group_id
                    GROUP BY 1
                    """,
                    uuid=node.uuid,
                    group_id=group_id,
                )
                projection[node.uuid] = [
                    Neighbor(node_uuid=r['uuid'], edge_count=r['count']) for r in records
                ]

            cluster_uuids = label_propagation(projection)

            for cluster in cluster_uuids:
                if not cluster:
                    continue
                cluster_records, _, _ = await executor.execute_query(
                    """
                    SELECT uuid, name, group_id, labels, summary, attributes, created_at
                    FROM entity_nodes
                    WHERE uuid = ANY($uuids)
                    """,
                    uuids=cluster,
                )
                community_clusters.append([_entity_node_from_pg(r) for r in cluster_records])

        return community_clusters

    async def remove_communities(
        self,
        executor: QueryExecutor,
        group_ids: list[str] | None = None,
    ) -> None:
        if group_ids:
            await executor.execute_query(
                'DELETE FROM community_edges WHERE group_id = ANY($group_ids)',
                group_ids=group_ids,
            )
            await executor.execute_query(
                'DELETE FROM community_nodes WHERE group_id = ANY($group_ids)',
                group_ids=group_ids,
            )
        else:
            await executor.execute_query('DELETE FROM community_edges')
            await executor.execute_query('DELETE FROM community_nodes')

    async def determine_entity_community(
        self,
        executor: QueryExecutor,
        entity: EntityNode,
    ) -> None:
        records, _, _ = await executor.execute_query(
            """
            SELECT c.uuid, c.name, c.group_id, c.name_embedding, c.summary, c.created_at
            FROM community_nodes c
            JOIN community_edges ce ON ce.source_node_uuid = c.uuid
                AND ce.group_id = c.group_id
            WHERE ce.target_node_uuid = $entity_uuid
              AND c.group_id = $group_id
            """,
            entity_uuid=entity.uuid,
            group_id=entity.group_id,
        )

        if len(records) > 0:
            return

        await executor.execute_query(
            """
            SELECT c.uuid, c.name, c.group_id, c.name_embedding, c.summary, c.created_at
            FROM community_nodes c
            JOIN community_edges ce ON ce.source_node_uuid = c.uuid
                AND ce.group_id = c.group_id
            JOIN entity_nodes m ON ce.target_node_uuid = m.uuid
                AND m.group_id = c.group_id
            JOIN entity_edges ee ON (
                (ee.source_node_uuid = m.uuid AND ee.target_node_uuid = $entity_uuid)
                OR (ee.target_node_uuid = m.uuid AND ee.source_node_uuid = $entity_uuid)
            ) AND ee.group_id = c.group_id
            WHERE c.group_id = $group_id
            """,
            entity_uuid=entity.uuid,
            group_id=entity.group_id,
        )

    async def get_mentioned_nodes(
        self,
        executor: QueryExecutor,
        episodes: list[EpisodicNode],
    ) -> list[EntityNode]:
        episode_uuids = [ep.uuid for ep in episodes]
        group_ids = list({ep.group_id for ep in episodes})

        records, _, _ = await executor.execute_query(
            """
            SELECT DISTINCT n.uuid, n.name, n.group_id, n.labels, n.summary,
                   n.attributes, n.created_at
            FROM entity_nodes n
            JOIN episodic_edges ee ON ee.target_node_uuid = n.uuid
                AND ee.group_id = n.group_id
            WHERE ee.source_node_uuid = ANY($uuids)
              AND n.group_id = ANY($group_ids)
            """,
            uuids=episode_uuids,
            group_ids=group_ids,
        )

        return [_entity_node_from_pg(r) for r in records]

    async def get_communities_by_nodes(
        self,
        executor: QueryExecutor,
        nodes: list[EntityNode],
    ) -> list[CommunityNode]:
        node_uuids = [n.uuid for n in nodes]
        group_ids = list({n.group_id for n in nodes})

        records, _, _ = await executor.execute_query(
            """
            SELECT DISTINCT c.uuid, c.name, c.group_id, c.name_embedding,
                   c.summary, c.created_at
            FROM community_nodes c
            JOIN community_edges ce ON ce.source_node_uuid = c.uuid
                AND ce.group_id = c.group_id
            WHERE ce.target_node_uuid = ANY($uuids)
              AND c.group_id = ANY($group_ids)
            """,
            uuids=node_uuids,
            group_ids=group_ids,
        )

        return [community_node_from_record(r) for r in records]


def _entity_node_from_pg(record: dict) -> EntityNode:
    attributes = record.get('attributes', {}) or {}
    if isinstance(attributes, str):
        import json

        attributes = json.loads(attributes)
    attributes.pop('uuid', None)
    attributes.pop('name', None)
    attributes.pop('group_id', None)
    attributes.pop('name_embedding', None)
    attributes.pop('summary', None)
    attributes.pop('created_at', None)
    attributes.pop('labels', None)

    labels = list(record.get('labels', []) or [])
    group_id = record.get('group_id', '')
    dynamic_label = 'Entity_' + group_id.replace('-', '')
    if dynamic_label in labels:
        labels.remove(dynamic_label)

    from graphiti_core.helpers import parse_db_date

    return EntityNode(
        uuid=record['uuid'],
        name=record['name'],
        name_embedding=record.get('name_embedding'),
        group_id=group_id,
        labels=labels,
        created_at=parse_db_date(record['created_at']),
        summary=record.get('summary', ''),
        attributes=attributes,
    )
