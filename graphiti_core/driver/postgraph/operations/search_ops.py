from __future__ import annotations

import json
import logging
from typing import Any

from graphiti_core.driver.operations.search_ops import SearchOperations
from graphiti_core.driver.query_executor import QueryExecutor
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

        client = executor.client
        conditions, params = _node_filter_conditions(search_filter, group_ids)
        params.append(ts_query)
        ts_idx = len(params)

        where = _where(conditions)
        and_kw = 'AND' if conditions else 'WHERE'

        sql = f"""
            SELECT realm, id, payload, created_at,
                   to_jsonb(t)->>'embedding' AS embedding_text,
                   ts_rank(search_vector, to_tsquery('simple', ${ts_idx})) AS score
            FROM "entity_nodes" t
            {where}
              {and_kw} search_vector @@ to_tsquery('simple', ${ts_idx})
            ORDER BY score DESC
            LIMIT {limit}
        """
        rows = await client._fetch(sql, *params)
        return [_parse_entity(r) for r in rows]

    async def node_similarity_search(
        self,
        executor: QueryExecutor,
        search_vector: list[float],
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.6,
    ) -> list[EntityNode]:
        client = executor.client

        results = []
        realms = group_ids or []
        if not realms:
            realm_rows = await client._fetch(
                'SELECT DISTINCT realm FROM "entity_nodes"'
            )
            realms = [r['realm'] for r in realm_rows]

        for realm in realms:
            hits = await client.vector_search(
                'entity_nodes', realm, search_vector,
                top_k=limit, distance_metric='cosine',
            )
            for vertex, distance in hits:
                score = 1.0 - distance
                if score > min_score:
                    node = _vertex_to_entity(vertex)
                    if _matches_node_filter(node, search_filter):
                        results.append((score, node))

        results.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in results[:limit]]

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

        client = executor.client
        results = []

        for origin_uuid in origin_uuids:
            origin_row = await client._fetch(
                'SELECT realm, id FROM "entity_nodes" '
                'WHERE payload @> $1::jsonb',
                json.dumps({'uuid': origin_uuid}),
            )
            if not origin_row:
                continue
            realm = origin_row[0]['realm']
            v_id = str(origin_row[0]['id'])

            if group_ids and realm not in group_ids:
                continue

            traversal = await client.traverse(
                realm=realm,
                start_table='entity_nodes',
                start_id=v_id,
                edge_tables=['entity_edges'],
                max_depth=max_depth,
                direction='both',
            )

            for step in traversal:
                if step['depth'] == 0:
                    continue
                step_id = step['id']
                v_rows = await client._fetch(
                    'SELECT realm, id, payload, created_at, '
                    'to_jsonb(t)->>\'embedding\' AS embedding_text '
                    'FROM "entity_nodes" t WHERE realm = $1 AND id = $2',
                    realm, int(step_id),
                )
                if v_rows:
                    node = _parse_entity(v_rows[0])
                    if _matches_node_filter(node, search_filter):
                        if not any(n.uuid == node.uuid for n in results):
                            results.append(node)

        return results[:limit]

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

        client = executor.client
        conditions, params = _edge_filter_conditions(search_filter, group_ids)
        params.append(ts_query)
        ts_idx = len(params)

        where = _where(conditions)
        and_kw = 'AND' if conditions else 'WHERE'

        sql = f"""
            SELECT realm, id, from_id, to_id, relation_type, payload,
                   created_at, to_jsonb(t)->>'embedding' AS embedding_text,
                   ts_rank(search_vector, to_tsquery('simple', ${ts_idx})) AS score
            FROM "entity_edges" t
            {where}
              {and_kw} search_vector @@ to_tsquery('simple', ${ts_idx})
            ORDER BY score DESC
            LIMIT {limit}
        """
        rows = await client._fetch(sql, *params)
        return [_parse_edge(r) for r in rows]

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
        client = executor.client

        results = []
        realms = group_ids or []
        if not realms:
            realm_rows = await client._fetch(
                'SELECT DISTINCT realm FROM "entity_edges"'
            )
            realms = [r['realm'] for r in realm_rows]

        for realm in realms:
            hits = await client.vector_search_edges(
                'entity_edges', realm, search_vector,
                top_k=limit, distance_metric='cosine',
            )
            for edge_obj, distance in hits:
                score = 1.0 - distance
                if score <= min_score:
                    continue
                p = edge_obj.payload
                if source_node_uuid and p.get('source_node_uuid') != source_node_uuid:
                    continue
                if target_node_uuid and p.get('target_node_uuid') != target_node_uuid:
                    continue
                edge = _edge_from_pg_edge(edge_obj)
                if _matches_edge_filter(edge, search_filter):
                    results.append((score, edge))

        results.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in results[:limit]]

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

        client = executor.client
        results = []

        for origin_uuid in origin_uuids:
            origin_row = await client._fetch(
                'SELECT realm, id FROM "entity_nodes" '
                'WHERE payload @> $1::jsonb',
                json.dumps({'uuid': origin_uuid}),
            )
            if not origin_row:
                continue
            realm = origin_row[0]['realm']
            v_id = str(origin_row[0]['id'])

            if group_ids and realm not in group_ids:
                continue

            traversal = await client.traverse(
                realm=realm,
                start_table='entity_nodes',
                start_id=v_id,
                edge_tables=['entity_edges'],
                max_depth=max_depth,
                direction='both',
            )

            visited_edge_ids = set()
            for step in traversal:
                for eid in (step.get('edge_ids') or []):
                    if eid in visited_edge_ids:
                        continue
                    visited_edge_ids.add(eid)
                    edge_obj = await client.get_edge(
                        'entity_edges', realm, eid
                    )
                    if edge_obj:
                        edge = _edge_from_pg_edge(edge_obj)
                        if _matches_edge_filter(edge, search_filter):
                            results.append(edge)

        return results[:limit]

    # --- Episode search ---

    async def episode_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        _search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EpisodicNode]:
        ts_query = _build_ts_query(query)
        if not ts_query:
            return []

        client = executor.client
        conditions: list[str] = []
        params: list[Any] = []

        if group_ids is not None:
            params.append(group_ids)
            conditions.append(f'realm = ANY(${len(params)})')

        params.append(ts_query)
        ts_idx = len(params)

        where = _where(conditions)
        and_kw = 'AND' if conditions else 'WHERE'

        sql = f"""
            SELECT realm, id, payload, created_at,
                   ts_rank(search_vector, to_tsquery('simple', ${ts_idx})) AS score
            FROM "episodic_nodes" t
            {where}
              {and_kw} search_vector @@ to_tsquery('simple', ${ts_idx})
            ORDER BY score DESC
            LIMIT {limit}
        """
        rows = await client._fetch(sql, *params)
        return [_parse_episode(r) for r in rows]

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

        client = executor.client
        conditions: list[str] = []
        params: list[Any] = []

        if group_ids is not None:
            params.append(group_ids)
            conditions.append(f'realm = ANY(${len(params)})')

        params.append(ts_query)
        ts_idx = len(params)

        where = _where(conditions)
        and_kw = 'AND' if conditions else 'WHERE'

        sql = f"""
            SELECT realm, id, payload, created_at,
                   to_jsonb(t)->>'embedding' AS embedding_text,
                   ts_rank(search_vector, to_tsquery('simple', ${ts_idx})) AS score
            FROM "community_nodes" t
            {where}
              {and_kw} search_vector @@ to_tsquery('simple', ${ts_idx})
            ORDER BY score DESC
            LIMIT {limit}
        """
        rows = await client._fetch(sql, *params)
        return [_parse_community(r) for r in rows]

    async def community_similarity_search(
        self,
        executor: QueryExecutor,
        search_vector: list[float],
        group_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.6,
    ) -> list[CommunityNode]:
        client = executor.client

        results = []
        realms = group_ids or []
        if not realms:
            realm_rows = await client._fetch(
                'SELECT DISTINCT realm FROM "community_nodes"'
            )
            realms = [r['realm'] for r in realm_rows]

        for realm in realms:
            hits = await client.vector_search(
                'community_nodes', realm, search_vector,
                top_k=limit, distance_metric='cosine',
            )
            for vertex, distance in hits:
                score = 1.0 - distance
                if score > min_score:
                    results.append((score, _vertex_to_community(vertex)))

        results.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in results[:limit]]

    # --- Rerankers ---

    async def node_distance_reranker(
        self,
        executor: QueryExecutor,
        node_uuids: list[str],
        center_node_uuid: str,
        min_score: float = 0,
    ) -> list[EntityNode]:
        client = executor.client
        filtered_uuids = [u for u in node_uuids if u != center_node_uuid]
        scores: dict[str, float] = {center_node_uuid: 0.0}

        center_row = await client._fetch(
            'SELECT realm, id FROM "entity_nodes" WHERE payload @> $1::jsonb',
            json.dumps({'uuid': center_node_uuid}),
        )
        if not center_row:
            return []
        realm = center_row[0]['realm']
        center_vid = center_row[0]['id']

        for uuid in filtered_uuids:
            vid_rows = await client._fetch(
                'SELECT id FROM "entity_nodes" '
                'WHERE realm = $1 AND payload @> $2::jsonb',
                realm, json.dumps({'uuid': uuid}),
            )
            if not vid_rows:
                scores[uuid] = float('inf')
                continue

            vid = vid_rows[0]['id']
            edge_rows = await client._fetch(
                """
                SELECT 1 FROM "entity_edges"
                WHERE realm = $1
                  AND ((from_id = $2 AND to_id = $3)
                    OR (from_id = $3 AND to_id = $2))
                LIMIT 1
                """,
                realm, center_vid, vid,
            )
            scores[uuid] = 1.0 if edge_rows else float('inf')

        filtered_uuids.sort(key=lambda u: scores.get(u, float('inf')))

        if center_node_uuid in node_uuids:
            scores[center_node_uuid] = 0.1
            filtered_uuids = [center_node_uuid] + filtered_uuids

        reranked_uuids = [
            u for u in filtered_uuids if (1 / scores.get(u, float('inf'))) >= min_score
        ]

        if not reranked_uuids:
            return []

        result_nodes = []
        for uuid in reranked_uuids:
            rows = await client._fetch(
                'SELECT realm, id, payload, created_at, '
                'to_jsonb(t)->>\'embedding\' AS embedding_text '
                'FROM "entity_nodes" t WHERE payload @> $1::jsonb',
                json.dumps({'uuid': uuid}),
            )
            if rows:
                result_nodes.append(_parse_entity(rows[0]))
        return result_nodes

    async def episode_mentions_reranker(
        self,
        executor: QueryExecutor,
        node_uuids: list[str],
        min_score: float = 0,
    ) -> list[EntityNode]:
        if not node_uuids:
            return []

        client = executor.client
        scores: dict[str, float] = {}

        for uuid in node_uuids:
            vid_rows = await client._fetch(
                'SELECT realm, id FROM "entity_nodes" '
                'WHERE payload @> $1::jsonb',
                json.dumps({'uuid': uuid}),
            )
            if not vid_rows:
                scores[uuid] = float('inf')
                continue

            realm = vid_rows[0]['realm']
            vid = vid_rows[0]['id']
            count_rows = await client._fetch(
                'SELECT count(*) AS cnt FROM "episodic_edges" '
                'WHERE realm = $1 AND to_id = $2',
                realm, vid,
            )
            scores[uuid] = float(count_rows[0]['cnt']) if count_rows else float('inf')

        sorted_uuids = list(node_uuids)
        sorted_uuids.sort(key=lambda u: scores.get(u, float('inf')))

        reranked_uuids = [u for u in sorted_uuids if scores.get(u, 0) >= min_score]

        if not reranked_uuids:
            return []

        result_nodes = []
        for uuid in reranked_uuids:
            rows = await client._fetch(
                'SELECT realm, id, payload, created_at, '
                'to_jsonb(t)->>\'embedding\' AS embedding_text '
                'FROM "entity_nodes" t WHERE payload @> $1::jsonb',
                json.dumps({'uuid': uuid}),
            )
            if rows:
                result_nodes.append(_parse_entity(rows[0]))
        return result_nodes

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
        _group_ids: list[str] | None = None,
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
) -> tuple[list[str], list[Any]]:
    conditions: list[str] = []
    params: list[Any] = []

    if search_filter.node_labels is not None:
        params.append(search_filter.node_labels)
        conditions.append(f"payload->'labels' ?| ${len(params)}")

    if group_ids is not None:
        params.append(group_ids)
        conditions.append(f'realm = ANY(${len(params)})')

    return conditions, params


def _edge_filter_conditions(
    search_filter: SearchFilters,
    group_ids: list[str] | None,
) -> tuple[list[str], list[Any]]:
    conditions: list[str] = []
    params: list[Any] = []

    if search_filter.edge_types is not None:
        params.append(search_filter.edge_types)
        conditions.append(f"payload->>'name' = ANY(${len(params)})")

    if search_filter.edge_uuids is not None:
        params.append(search_filter.edge_uuids)
        conditions.append(f"payload->>'uuid' = ANY(${len(params)})")

    if group_ids is not None:
        params.append(group_ids)
        conditions.append(f'realm = ANY(${len(params)})')

    _add_date_filters(conditions, params, search_filter)

    return conditions, params


def _add_date_filters(
    conditions: list[str],
    params: list[Any],
    filters: SearchFilters,
) -> None:
    for field_name in ('valid_at', 'invalid_at', 'created_at', 'expired_at'):
        date_lists = getattr(filters, field_name, None)
        if date_lists is None:
            continue
        or_parts = []
        for or_list in date_lists:
            and_parts = []
            for df in or_list:
                op = df.comparison_operator.value
                if op in ('IS NULL', 'IS NOT NULL'):
                    and_parts.append(
                        f"((payload->>'{field_name}') {op})"
                    )
                else:
                    params.append(df.date.isoformat() if hasattr(df.date, 'isoformat') else str(df.date))
                    and_parts.append(
                        f"((payload->>'{field_name}')::timestamptz {op} ${len(params)}::timestamptz)"
                    )
            or_parts.append('(' + ' AND '.join(and_parts) + ')')
        conditions.append('(' + ' OR '.join(or_parts) + ')')


def _vertex_to_entity(v) -> EntityNode:
    p = v.payload
    labels = list(p.get('labels', []))
    group_id = v.realm
    dynamic_label = 'Entity_' + group_id.replace('-', '')
    if dynamic_label in labels:
        labels.remove(dynamic_label)
    return EntityNode(
        uuid=p['uuid'],
        name=p.get('name', ''),
        name_embedding=v.embedding,
        group_id=group_id,
        labels=labels,
        created_at=parse_db_date(p.get('created_at')) or v.created_at,
        summary=p.get('summary', ''),
        attributes=dict(p.get('attributes', {})),
    )


def _vertex_to_community(v) -> CommunityNode:
    p = v.payload
    return CommunityNode(
        uuid=p['uuid'],
        name=p.get('name', ''),
        group_id=v.realm,
        name_embedding=v.embedding,
        created_at=parse_db_date(p.get('created_at')) or v.created_at,
        summary=p.get('summary', ''),
    )


def _parse_entity(row: dict) -> EntityNode:
    p = row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload'])
    labels = list(p.get('labels', []))
    group_id = row['realm']
    dynamic_label = 'Entity_' + group_id.replace('-', '')
    if dynamic_label in labels:
        labels.remove(dynamic_label)

    emb = None
    if row.get('embedding_text'):
        emb = [float(x) for x in row['embedding_text'].strip('[]').split(',') if x.strip()]

    return EntityNode(
        uuid=p['uuid'],
        name=p.get('name', ''),
        name_embedding=emb,
        group_id=group_id,
        labels=labels,
        created_at=parse_db_date(p.get('created_at')) or row.get('created_at'),
        summary=p.get('summary', ''),
        attributes=dict(p.get('attributes', {})),
    )


def _parse_edge(row: dict) -> EntityEdge:
    p = row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload'])

    emb = None
    if row.get('embedding_text'):
        emb = [float(x) for x in row['embedding_text'].strip('[]').split(',') if x.strip()]

    return EntityEdge(
        uuid=p['uuid'],
        source_node_uuid=p['source_node_uuid'],
        target_node_uuid=p['target_node_uuid'],
        fact=p.get('fact', ''),
        fact_embedding=emb,
        name=p.get('name', ''),
        group_id=row['realm'],
        episodes=list(p.get('episodes', [])),
        created_at=parse_db_date(p.get('created_at')) or row.get('created_at'),
        expired_at=parse_db_date(p.get('expired_at')),
        valid_at=parse_db_date(p.get('valid_at')),
        invalid_at=parse_db_date(p.get('invalid_at')),
        reference_time=parse_db_date(p.get('reference_time')),
        attributes=dict(p.get('attributes', {})),
    )


def _edge_from_pg_edge(edge_obj) -> EntityEdge:
    p = edge_obj.payload
    return EntityEdge(
        uuid=p['uuid'],
        source_node_uuid=p['source_node_uuid'],
        target_node_uuid=p['target_node_uuid'],
        fact=p.get('fact', ''),
        fact_embedding=edge_obj.embedding,
        name=p.get('name', ''),
        group_id=edge_obj.realm,
        episodes=list(p.get('episodes', [])),
        created_at=parse_db_date(p.get('created_at')) or edge_obj.created_at,
        expired_at=parse_db_date(p.get('expired_at')),
        valid_at=parse_db_date(p.get('valid_at')),
        invalid_at=parse_db_date(p.get('invalid_at')),
        reference_time=parse_db_date(p.get('reference_time')),
        attributes=dict(p.get('attributes', {})),
    )


def _parse_episode(row: dict) -> EpisodicNode:
    p = row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload'])
    created_at = parse_db_date(p.get('created_at')) or row.get('created_at')
    valid_at = parse_db_date(p.get('valid_at'))

    if created_at is None:
        raise ValueError(
            f'created_at cannot be None for episode {p.get("uuid", "unknown")}'
        )
    if valid_at is None:
        raise ValueError(
            f'valid_at cannot be None for episode {p.get("uuid", "unknown")}'
        )

    return EpisodicNode(
        content=p.get('content', ''),
        created_at=created_at,
        valid_at=valid_at,
        uuid=p['uuid'],
        group_id=row['realm'],
        source=EpisodeType.from_str(p.get('source', 'text')),
        name=p.get('name', ''),
        source_description=p.get('source_description', ''),
        entity_edges=list(p.get('entity_edges', [])),
    )


def _parse_community(row: dict) -> CommunityNode:
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


def _matches_node_filter(node: EntityNode, sf: SearchFilters) -> bool:
    if sf.node_labels is not None:
        if not set(sf.node_labels) & set(node.labels):
            return False
    return True


def _matches_edge_filter(edge: EntityEdge, sf: SearchFilters) -> bool:
    if sf.edge_types is not None:
        if edge.name not in sf.edge_types:
            return False
    if sf.edge_uuids is not None:
        if edge.uuid not in sf.edge_uuids:
            return False
    return True
