from __future__ import annotations

import json
import logging
from typing import Any

from graphiti_core.driver.operations.search_ops import SearchOperations
from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.driver.record_parsers import community_node_from_record
from graphiti_core.edges import EntityEdge
from graphiti_core.helpers import parse_db_date
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search.search_filters import SearchFilters

logger = logging.getLogger(__name__)


class PGSearchOperations(SearchOperations):
    # --- Node search ---

    async def node_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityNode]:
        ts_query = _build_ts_query(query)
        if not ts_query:
            return []

        conditions, params = _node_filter_conditions(search_filter, group_ids)
        params['ts_query'] = ts_query

        where = _where(conditions)

        sql = f"""
            SELECT uuid, name, group_id, labels, summary, attributes, created_at,
                   ts_rank(search_vector, to_tsquery('simple', $ts_query)) AS score
            FROM entity_nodes
            {where}
              {'AND' if conditions else 'WHERE'} search_vector @@ to_tsquery('simple', $ts_query)
            ORDER BY score DESC
            LIMIT {limit}
        """
        records, _, _ = await executor.execute_query(sql, **params)
        return [_parse_entity(r) for r in records]

    async def node_similarity_search(
        self,
        executor: QueryExecutor,
        search_vector: list[float],
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.6,
    ) -> list[EntityNode]:
        conditions, params = _node_filter_conditions(search_filter, group_ids)
        params['search_vector'] = str(search_vector)
        params['min_score'] = min_score

        where = _where(conditions)

        sql = f"""
            SELECT uuid, name, group_id, labels, summary, attributes, created_at,
                   1 - (name_embedding <=> $search_vector::vector) AS score
            FROM entity_nodes
            {where}
              {'AND' if conditions else 'WHERE'} name_embedding IS NOT NULL
            HAVING 1 - (name_embedding <=> $search_vector::vector) > $min_score
            ORDER BY score DESC
            LIMIT {limit}
        """
        # HAVING not valid with non-aggregate; use subquery
        sql = f"""
            SELECT * FROM (
                SELECT uuid, name, group_id, labels, summary, attributes, created_at,
                       1 - (name_embedding <=> $search_vector::vector) AS score
                FROM entity_nodes
                {where}
                  {'AND' if conditions else 'WHERE'} name_embedding IS NOT NULL
            ) sub
            WHERE score > $min_score
            ORDER BY score DESC
            LIMIT {limit}
        """
        records, _, _ = await executor.execute_query(sql, **params)
        return [_parse_entity(r) for r in records]

    async def node_bfs_search(
        self,
        executor: QueryExecutor,
        origin_uuids: list[str],
        search_filter: SearchFilters,
        max_depth: int,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityNode]:
        if not origin_uuids or max_depth < 1:
            return []

        conditions, params = _node_filter_conditions(search_filter, group_ids)
        params['origin_uuids'] = origin_uuids

        filter_clause = ''
        if conditions:
            filter_clause = ' AND ' + ' AND '.join(conditions)

        sql = f"""
            WITH RECURSIVE bfs AS (
                SELECT uuid, 0 AS depth
                FROM entity_nodes
                WHERE uuid = ANY($origin_uuids)
                UNION
                SELECT DISTINCT
                    CASE WHEN ee.source_node_uuid = bfs.uuid THEN ee.target_node_uuid
                         ELSE ee.source_node_uuid END,
                    bfs.depth + 1
                FROM bfs
                JOIN entity_edges ee ON (ee.source_node_uuid = bfs.uuid OR ee.target_node_uuid = bfs.uuid)
                WHERE bfs.depth < {max_depth}
            )
            SELECT DISTINCT n.uuid, n.name, n.group_id, n.labels, n.summary,
                   n.attributes, n.created_at
            FROM bfs
            JOIN entity_nodes n ON n.uuid = bfs.uuid
            WHERE bfs.depth > 0
            {filter_clause}
            LIMIT {limit}
        """
        records, _, _ = await executor.execute_query(sql, **params)
        return [_parse_entity(r) for r in records]

    # --- Edge search ---

    async def edge_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityEdge]:
        ts_query = _build_ts_query(query)
        if not ts_query:
            return []

        conditions, params = _edge_filter_conditions(search_filter, group_ids)
        params['ts_query'] = ts_query

        where = _where(conditions)

        sql = f"""
            SELECT uuid, source_node_uuid, target_node_uuid, name, fact,
                   group_id, episodes, created_at, expired_at, valid_at,
                   invalid_at, reference_time, attributes,
                   ts_rank(search_vector, to_tsquery('simple', $ts_query)) AS score
            FROM entity_edges
            {where}
              {'AND' if conditions else 'WHERE'} search_vector @@ to_tsquery('simple', $ts_query)
            ORDER BY score DESC
            LIMIT {limit}
        """
        records, _, _ = await executor.execute_query(sql, **params)
        return [_parse_edge(r) for r in records]

    async def edge_similarity_search(
        self,
        executor: QueryExecutor,
        search_vector: list[float],
        source_node_uuid: str | None,
        target_node_uuid: str | None,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.6,
    ) -> list[EntityEdge]:
        conditions, params = _edge_filter_conditions(search_filter, group_ids)
        params['search_vector'] = str(search_vector)
        params['min_score'] = min_score

        if source_node_uuid is not None:
            conditions.append('source_node_uuid = $source_uuid')
            params['source_uuid'] = source_node_uuid
        if target_node_uuid is not None:
            conditions.append('target_node_uuid = $target_uuid')
            params['target_uuid'] = target_node_uuid

        where = _where(conditions)

        sql = f"""
            SELECT * FROM (
                SELECT uuid, source_node_uuid, target_node_uuid, name, fact,
                       group_id, episodes, created_at, expired_at, valid_at,
                       invalid_at, reference_time, attributes,
                       1 - (fact_embedding <=> $search_vector::vector) AS score
                FROM entity_edges
                {where}
                  {'AND' if conditions else 'WHERE'} fact_embedding IS NOT NULL
            ) sub
            WHERE score > $min_score
            ORDER BY score DESC
            LIMIT {limit}
        """
        records, _, _ = await executor.execute_query(sql, **params)
        return [_parse_edge(r) for r in records]

    async def edge_bfs_search(
        self,
        executor: QueryExecutor,
        origin_uuids: list[str],
        max_depth: int,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityEdge]:
        if not origin_uuids:
            return []

        conditions, params = _edge_filter_conditions(search_filter, group_ids)
        params['origin_uuids'] = origin_uuids

        filter_clause = ''
        if conditions:
            filter_clause = ' AND ' + ' AND '.join(
                c.replace('group_id', 'ee.group_id')
                .replace('name ', 'ee.name ')
                .replace('uuid ', 'ee.uuid ')
                for c in conditions
            )

        sql = f"""
            WITH RECURSIVE bfs AS (
                SELECT uuid, 0 AS depth
                FROM entity_nodes
                WHERE uuid = ANY($origin_uuids)
                UNION
                SELECT DISTINCT
                    CASE WHEN ee2.source_node_uuid = bfs.uuid THEN ee2.target_node_uuid
                         ELSE ee2.source_node_uuid END,
                    bfs.depth + 1
                FROM bfs
                JOIN entity_edges ee2 ON (ee2.source_node_uuid = bfs.uuid OR ee2.target_node_uuid = bfs.uuid)
                WHERE bfs.depth < {max_depth}
            )
            SELECT DISTINCT ee.uuid, ee.source_node_uuid, ee.target_node_uuid, ee.name, ee.fact,
                   ee.group_id, ee.episodes, ee.created_at, ee.expired_at, ee.valid_at,
                   ee.invalid_at, ee.reference_time, ee.attributes
            FROM bfs
            JOIN entity_edges ee ON (ee.source_node_uuid = bfs.uuid OR ee.target_node_uuid = bfs.uuid)
            WHERE bfs.depth > 0
            {filter_clause}
            LIMIT {limit}
        """
        records, _, _ = await executor.execute_query(sql, **params)
        return [_parse_edge(r) for r in records]

    # --- Episode search ---

    async def episode_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EpisodicNode]:
        ts_query = _build_ts_query(query)
        if not ts_query:
            return []

        conditions: list[str] = []
        params: dict[str, Any] = {'ts_query': ts_query}

        if group_ids is not None:
            conditions.append('group_id = ANY($group_ids)')
            params['group_ids'] = group_ids

        where = _where(conditions)

        sql = f"""
            SELECT uuid, name, group_id, source, source_description, content,
                   valid_at, entity_edges, created_at,
                   ts_rank(search_vector, to_tsquery('simple', $ts_query)) AS score
            FROM episodic_nodes
            {where}
              {'AND' if conditions else 'WHERE'} search_vector @@ to_tsquery('simple', $ts_query)
            ORDER BY score DESC
            LIMIT {limit}
        """
        records, _, _ = await executor.execute_query(sql, **params)
        return [_parse_episode(r) for r in records]

    # --- Community search ---

    async def community_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[CommunityNode]:
        ts_query = _build_ts_query(query)
        if not ts_query:
            return []

        conditions: list[str] = []
        params: dict[str, Any] = {'ts_query': ts_query}

        if group_ids is not None:
            conditions.append('group_id = ANY($group_ids)')
            params['group_ids'] = group_ids

        where = _where(conditions)

        sql = f"""
            SELECT uuid, name, group_id, name_embedding, summary, created_at,
                   ts_rank(search_vector, to_tsquery('simple', $ts_query)) AS score
            FROM community_nodes
            {where}
              {'AND' if conditions else 'WHERE'} search_vector @@ to_tsquery('simple', $ts_query)
            ORDER BY score DESC
            LIMIT {limit}
        """
        records, _, _ = await executor.execute_query(sql, **params)
        return [community_node_from_record(r) for r in records]

    async def community_similarity_search(
        self,
        executor: QueryExecutor,
        search_vector: list[float],
        group_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.6,
    ) -> list[CommunityNode]:
        conditions: list[str] = []
        params: dict[str, Any] = {
            'search_vector': str(search_vector),
            'min_score': min_score,
        }

        if group_ids is not None:
            conditions.append('group_id = ANY($group_ids)')
            params['group_ids'] = group_ids

        where = _where(conditions)

        sql = f"""
            SELECT * FROM (
                SELECT uuid, name, group_id, name_embedding, summary, created_at,
                       1 - (name_embedding <=> $search_vector::vector) AS score
                FROM community_nodes
                {where}
                  {'AND' if conditions else 'WHERE'} name_embedding IS NOT NULL
            ) sub
            WHERE score > $min_score
            ORDER BY score DESC
            LIMIT {limit}
        """
        records, _, _ = await executor.execute_query(sql, **params)
        return [community_node_from_record(r) for r in records]

    # --- Rerankers ---

    async def node_distance_reranker(
        self,
        executor: QueryExecutor,
        node_uuids: list[str],
        center_node_uuid: str,
        min_score: float = 0,
    ) -> list[EntityNode]:
        filtered_uuids = [u for u in node_uuids if u != center_node_uuid]
        scores: dict[str, float] = {center_node_uuid: 0.0}

        records, _, _ = await executor.execute_query(
            """
            SELECT 1 AS score, n.uuid
            FROM entity_nodes n
            JOIN entity_edges ee ON (
                (ee.source_node_uuid = $center_uuid AND ee.target_node_uuid = n.uuid)
                OR (ee.target_node_uuid = $center_uuid AND ee.source_node_uuid = n.uuid)
            )
            WHERE n.uuid = ANY($node_uuids)
            """,
            node_uuids=filtered_uuids,
            center_uuid=center_node_uuid,
        )

        for r in records:
            scores[r['uuid']] = r['score']

        for uuid in filtered_uuids:
            if uuid not in scores:
                scores[uuid] = float('inf')

        filtered_uuids.sort(key=lambda u: scores[u])

        if center_node_uuid in node_uuids:
            scores[center_node_uuid] = 0.1
            filtered_uuids = [center_node_uuid] + filtered_uuids

        reranked_uuids = [u for u in filtered_uuids if (1 / scores[u]) >= min_score]

        if not reranked_uuids:
            return []

        get_records, _, _ = await executor.execute_query(
            """
            SELECT uuid, name, group_id, labels, summary, attributes, created_at
            FROM entity_nodes WHERE uuid = ANY($uuids)
            """,
            uuids=reranked_uuids,
        )
        node_map = {r['uuid']: _parse_entity(r) for r in get_records}
        return [node_map[u] for u in reranked_uuids if u in node_map]

    async def episode_mentions_reranker(
        self,
        executor: QueryExecutor,
        node_uuids: list[str],
        min_score: float = 0,
    ) -> list[EntityNode]:
        if not node_uuids:
            return []

        scores: dict[str, float] = {}

        records, _, _ = await executor.execute_query(
            """
            SELECT count(*) AS score, n.uuid
            FROM entity_nodes n
            JOIN episodic_edges ee ON ee.target_node_uuid = n.uuid
            WHERE n.uuid = ANY($node_uuids)
            GROUP BY n.uuid
            """,
            node_uuids=node_uuids,
        )

        for r in records:
            scores[r['uuid']] = r['score']

        for uuid in node_uuids:
            if uuid not in scores:
                scores[uuid] = float('inf')

        sorted_uuids = list(node_uuids)
        sorted_uuids.sort(key=lambda u: scores[u])

        reranked_uuids = [u for u in sorted_uuids if scores[u] >= min_score]

        if not reranked_uuids:
            return []

        get_records, _, _ = await executor.execute_query(
            """
            SELECT uuid, name, group_id, labels, summary, attributes, created_at
            FROM entity_nodes WHERE uuid = ANY($uuids)
            """,
            uuids=reranked_uuids,
        )
        node_map = {r['uuid']: _parse_entity(r) for r in get_records}
        return [node_map[u] for u in reranked_uuids if u in node_map]

    # --- Filter builders ---

    def build_node_search_filters(self, search_filters: SearchFilters) -> Any:
        conditions, params = _node_filter_conditions(search_filters, None)
        return {'filter_queries': conditions, 'filter_params': params}

    def build_edge_search_filters(self, search_filters: SearchFilters) -> Any:
        conditions, params = _edge_filter_conditions(search_filters, None)
        return {'filter_queries': conditions, 'filter_params': params}

    def build_fulltext_query(
        self,
        query: str,
        group_ids: list[str] | None = None,
        max_query_length: int = 8000,
    ) -> str:
        return _build_ts_query(query, max_query_length)


# --- helpers ---


def _build_ts_query(query: str, max_length: int = 128) -> str:
    words = query.strip().split()[:max_length]
    terms = [w for w in words if w]
    if not terms:
        return ''
    return ' & '.join(terms)


def _where(conditions: list[str]) -> str:
    if not conditions:
        return ''
    return 'WHERE ' + ' AND '.join(conditions)


def _node_filter_conditions(
    search_filter: SearchFilters,
    group_ids: list[str] | None,
) -> tuple[list[str], dict[str, Any]]:
    conditions: list[str] = []
    params: dict[str, Any] = {}

    if search_filter.node_labels is not None:
        conditions.append('labels && $labels')
        params['labels'] = search_filter.node_labels

    if group_ids is not None:
        conditions.append('group_id = ANY($group_ids)')
        params['group_ids'] = group_ids

    return conditions, params


def _edge_filter_conditions(
    search_filter: SearchFilters,
    group_ids: list[str] | None,
) -> tuple[list[str], dict[str, Any]]:
    conditions: list[str] = []
    params: dict[str, Any] = {}

    if search_filter.edge_types is not None:
        conditions.append('name = ANY($edge_types)')
        params['edge_types'] = search_filter.edge_types

    if search_filter.edge_uuids is not None:
        conditions.append('uuid = ANY($edge_uuids)')
        params['edge_uuids'] = search_filter.edge_uuids

    if group_ids is not None:
        conditions.append('group_id = ANY($group_ids)')
        params['group_ids'] = group_ids

    _add_date_filters(conditions, params, search_filter)

    return conditions, params


def _add_date_filters(
    conditions: list[str],
    params: dict[str, Any],
    filters: SearchFilters,
) -> None:
    for field_name in ('valid_at', 'invalid_at', 'created_at', 'expired_at'):
        date_lists = getattr(filters, field_name, None)
        if date_lists is None:
            continue
        or_parts = []
        for i, or_list in enumerate(date_lists):
            and_parts = []
            for j, df in enumerate(or_list):
                param_key = f'{field_name}_{i}_{j}'
                op = df.comparison_operator.value
                if op in ('IS NULL', 'IS NOT NULL'):
                    and_parts.append(f'({field_name} {op})')
                else:
                    and_parts.append(f'({field_name} {op} ${param_key})')
                    params[param_key] = df.date
            or_parts.append('(' + ' AND '.join(and_parts) + ')')
        conditions.append('(' + ' OR '.join(or_parts) + ')')


def _parse_entity(record: dict) -> EntityNode:
    attributes = record.get('attributes', {}) or {}
    if isinstance(attributes, str):
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


def _parse_edge(record: dict) -> EntityEdge:
    attributes = record.get('attributes', {}) or {}
    if isinstance(attributes, str):
        attributes = json.loads(attributes)
    attributes.pop('uuid', None)
    attributes.pop('source_node_uuid', None)
    attributes.pop('target_node_uuid', None)
    attributes.pop('fact', None)
    attributes.pop('fact_embedding', None)
    attributes.pop('name', None)
    attributes.pop('group_id', None)
    attributes.pop('episodes', None)
    attributes.pop('created_at', None)
    attributes.pop('expired_at', None)
    attributes.pop('valid_at', None)
    attributes.pop('invalid_at', None)
    attributes.pop('reference_time', None)

    return EntityEdge(
        uuid=record['uuid'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        fact=record['fact'],
        fact_embedding=record.get('fact_embedding'),
        name=record['name'],
        group_id=record['group_id'],
        episodes=list(record.get('episodes', []) or []),
        created_at=parse_db_date(record['created_at']),
        expired_at=parse_db_date(record.get('expired_at')),
        valid_at=parse_db_date(record.get('valid_at')),
        invalid_at=parse_db_date(record.get('invalid_at')),
        reference_time=parse_db_date(record.get('reference_time')),
        attributes=attributes,
    )


def _parse_episode(record: dict) -> EpisodicNode:
    created_at = parse_db_date(record['created_at'])
    valid_at = parse_db_date(record['valid_at'])

    if created_at is None:
        raise ValueError(f'created_at cannot be None for episode {record.get("uuid", "unknown")}')
    if valid_at is None:
        raise ValueError(f'valid_at cannot be None for episode {record.get("uuid", "unknown")}')

    return EpisodicNode(
        content=record['content'],
        created_at=created_at,
        valid_at=valid_at,
        uuid=record['uuid'],
        group_id=record['group_id'],
        source=EpisodeType.from_str(record['source']),
        name=record['name'],
        source_description=record['source_description'],
        entity_edges=list(record.get('entity_edges', []) or []),
    )
