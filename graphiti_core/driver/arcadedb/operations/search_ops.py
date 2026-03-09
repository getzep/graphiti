"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from typing import Any

import numpy as np

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.operations.search_ops import SearchOperations
from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.driver.record_parsers import (
    community_node_from_record,
    entity_edge_from_record,
    entity_node_from_record,
    episodic_node_from_record,
)
from graphiti_core.edges import EntityEdge
from graphiti_core.helpers import lucene_sanitize
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_return_query
from graphiti_core.models.nodes.node_db_queries import (
    COMMUNITY_NODE_RETURN,
    EPISODIC_NODE_RETURN,
    get_entity_node_return_query,
)
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.search.search_filters import (
    SearchFilters,
    edge_search_filter_query_constructor,
    node_search_filter_query_constructor,
)

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 128


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _build_arcadedb_fulltext_query(
    query: str,
    group_ids: list[str] | None = None,
    max_query_length: int = MAX_QUERY_LENGTH,
) -> str:
    """Build a Lucene query string for ArcadeDB fulltext search."""
    group_ids_filter_list = [f'group_id:"{g}"' for g in group_ids] if group_ids is not None else []
    group_ids_filter = ''
    for f in group_ids_filter_list:
        group_ids_filter += f if not group_ids_filter else f' OR {f}'

    group_ids_filter += ' AND ' if group_ids_filter else ''

    lucene_query = lucene_sanitize(query)
    if len(lucene_query.split(' ')) + len(group_ids or '') >= max_query_length:
        return ''

    full_query = group_ids_filter + '(' + lucene_query + ')'
    return full_query


class ArcadeDBSearchOperations(SearchOperations):
    """Search operations for ArcadeDB.

    ArcadeDB uses Lucene-based fulltext indexes and HNSW vector indexes.
    Fulltext search uses CONTAINS predicate with Lucene syntax.
    Vector similarity is computed in Python using numpy after fetching
    candidate nodes with their embeddings.
    """

    # --- Node search ---

    async def node_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityNode]:
        fuzzy_query = _build_arcadedb_fulltext_query(query, group_ids)
        if fuzzy_query == '':
            return []

        filter_queries, filter_params = node_search_filter_query_constructor(
            search_filter, GraphProvider.ARCADEDB
        )

        if group_ids is not None:
            filter_queries.append('n.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

        filter_query = ''
        if filter_queries:
            filter_query = ' AND ' + (' AND '.join(filter_queries))

        # Use toLower/CONTAINS for fulltext search (compatible with ArcadeDB Cypher)
        search_terms = lucene_sanitize(query).split()
        if not search_terms:
            return []

        search_clauses = []
        for i, term in enumerate(search_terms[:5]):
            param_key = f'term_{i}'
            filter_params[param_key] = term.lower()
            search_clauses.append(
                f'(toLower(n.name) CONTAINS ${param_key}'
                f' OR toLower(n.summary) CONTAINS ${param_key})'
            )

        search_query = ' OR '.join(search_clauses)

        cypher = (
            'MATCH (n:Entity)'
            + f' WHERE ({search_query})'
            + filter_query
            + """
            RETURN
            """
            + get_entity_node_return_query(GraphProvider.ARCADEDB)
            + """
            LIMIT $limit
            """
        )

        records, _, _ = await executor.execute_query(
            cypher,
            limit=limit,
            routing_='r',
            **filter_params,
        )

        return [entity_node_from_record(r) for r in records]

    async def node_similarity_search(
        self,
        executor: QueryExecutor,
        search_vector: list[float],
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.6,
    ) -> list[EntityNode]:
        filter_queries, filter_params = node_search_filter_query_constructor(
            search_filter, GraphProvider.ARCADEDB
        )

        if group_ids is not None:
            filter_queries.append('n.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

        filter_query = ''
        if filter_queries:
            filter_query = ' WHERE ' + (' AND '.join(filter_queries))

        # Fetch candidate nodes with embeddings
        cypher = (
            'MATCH (n:Entity)'
            + filter_query
            + """
            WHERE n.name_embedding IS NOT NULL
            RETURN
            """
            + get_entity_node_return_query(GraphProvider.ARCADEDB)
            + """,
            n.name_embedding AS name_embedding
            """
        )

        records, _, _ = await executor.execute_query(
            cypher,
            routing_='r',
            **filter_params,
        )

        # Compute cosine similarity in Python and filter/sort
        scored_records = []
        for r in records:
            embedding = r.get('name_embedding')
            if embedding is not None:
                score = _cosine_similarity(embedding, search_vector)
                if score > min_score:
                    scored_records.append((score, r))

        scored_records.sort(key=lambda x: x[0], reverse=True)
        scored_records = scored_records[:limit]

        return [entity_node_from_record(r) for _, r in scored_records]

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

        filter_queries, filter_params = node_search_filter_query_constructor(
            search_filter, GraphProvider.ARCADEDB
        )

        if group_ids is not None:
            filter_queries.append('n.group_id IN $group_ids')
            filter_queries.append('origin.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

        filter_query = ''
        if filter_queries:
            filter_query = ' AND ' + (' AND '.join(filter_queries))

        cypher = (
            f"""
            UNWIND $bfs_origin_node_uuids AS origin_uuid
            MATCH (origin {{uuid: origin_uuid}})-[:RELATES_TO|MENTIONS*1..{max_depth}]->(n:Entity)
            WHERE n.group_id = origin.group_id
            """
            + filter_query
            + """
            RETURN
            """
            + get_entity_node_return_query(GraphProvider.ARCADEDB)
            + """
            LIMIT $limit
            """
        )

        records, _, _ = await executor.execute_query(
            cypher,
            bfs_origin_node_uuids=origin_uuids,
            limit=limit,
            routing_='r',
            **filter_params,
        )

        return [entity_node_from_record(r) for r in records]

    # --- Edge search ---

    async def edge_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityEdge]:
        fuzzy_query = _build_arcadedb_fulltext_query(query, group_ids)
        if fuzzy_query == '':
            return []

        filter_queries, filter_params = edge_search_filter_query_constructor(
            search_filter, GraphProvider.ARCADEDB
        )

        if group_ids is not None:
            filter_queries.append('e.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

        filter_query = ''
        if filter_queries:
            filter_query = ' AND ' + (' AND '.join(filter_queries))

        # Use CONTAINS for fulltext search
        search_terms = lucene_sanitize(query).split()
        if not search_terms:
            return []

        search_clauses = []
        for i, term in enumerate(search_terms[:5]):
            param_key = f'term_{i}'
            filter_params[param_key] = term.lower()
            search_clauses.append(
                f'(toLower(e.name) CONTAINS ${param_key} OR toLower(e.fact) CONTAINS ${param_key})'
            )

        search_query = ' OR '.join(search_clauses)

        cypher = (
            'MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)'
            + f' WHERE ({search_query})'
            + filter_query
            + """
            RETURN
            """
            + get_entity_edge_return_query(GraphProvider.ARCADEDB)
            + """
            LIMIT $limit
            """
        )

        records, _, _ = await executor.execute_query(
            cypher,
            limit=limit,
            routing_='r',
            **filter_params,
        )

        return [entity_edge_from_record(r) for r in records]

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
        filter_queries, filter_params = edge_search_filter_query_constructor(
            search_filter, GraphProvider.ARCADEDB
        )

        if group_ids is not None:
            filter_queries.append('e.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

            if source_node_uuid is not None:
                filter_params['source_uuid'] = source_node_uuid
                filter_queries.append('n.uuid = $source_uuid')

            if target_node_uuid is not None:
                filter_params['target_uuid'] = target_node_uuid
                filter_queries.append('m.uuid = $target_uuid')

        filter_query = ''
        if filter_queries:
            filter_query = ' WHERE ' + (' AND '.join(filter_queries))

        # Fetch candidate edges with embeddings
        cypher = (
            'MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)'
            + filter_query
            + """
            WHERE e.fact_embedding IS NOT NULL
            RETURN DISTINCT
            """
            + get_entity_edge_return_query(GraphProvider.ARCADEDB)
            + """,
            e.fact_embedding AS fact_embedding
            """
        )

        records, _, _ = await executor.execute_query(
            cypher,
            routing_='r',
            **filter_params,
        )

        # Compute cosine similarity in Python and filter/sort
        scored_records = []
        for r in records:
            embedding = r.get('fact_embedding')
            if embedding is not None:
                score = _cosine_similarity(embedding, search_vector)
                if score > min_score:
                    scored_records.append((score, r))

        scored_records.sort(key=lambda x: x[0], reverse=True)
        scored_records = scored_records[:limit]

        return [entity_edge_from_record(r) for _, r in scored_records]

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

        filter_queries, filter_params = edge_search_filter_query_constructor(
            search_filter, GraphProvider.ARCADEDB
        )

        if group_ids is not None:
            filter_queries.append('e.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

        filter_query = ''
        if filter_queries:
            filter_query = ' WHERE ' + (' AND '.join(filter_queries))

        cypher = (
            f"""
            UNWIND $bfs_origin_node_uuids AS origin_uuid
            MATCH path = (origin {{uuid: origin_uuid}})-[:RELATES_TO|MENTIONS*1..{max_depth}]->(:Entity)
            UNWIND relationships(path) AS rel
            MATCH (n:Entity)-[e:RELATES_TO {{uuid: rel.uuid}}]-(m:Entity)
            """
            + filter_query
            + """
            RETURN DISTINCT
            """
            + get_entity_edge_return_query(GraphProvider.ARCADEDB)
            + """
            LIMIT $limit
            """
        )

        records, _, _ = await executor.execute_query(
            cypher,
            bfs_origin_node_uuids=origin_uuids,
            depth=max_depth,
            limit=limit,
            routing_='r',
            **filter_params,
        )

        return [entity_edge_from_record(r) for r in records]

    # --- Episode search ---

    async def episode_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,  # noqa: ARG002
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EpisodicNode]:
        fuzzy_query = _build_arcadedb_fulltext_query(query, group_ids)
        if fuzzy_query == '':
            return []

        filter_params: dict[str, Any] = {}
        group_filter_query = ''
        if group_ids is not None:
            group_filter_query += '\nAND e.group_id IN $group_ids'
            filter_params['group_ids'] = group_ids

        # Use CONTAINS for fulltext search
        search_terms = lucene_sanitize(query).split()
        if not search_terms:
            return []

        search_clauses = []
        for i, term in enumerate(search_terms[:5]):
            param_key = f'term_{i}'
            filter_params[param_key] = term.lower()
            search_clauses.append(f'toLower(e.content) CONTAINS ${param_key}')

        search_query = ' OR '.join(search_clauses)

        cypher = (
            'MATCH (e:Episodic)'
            + f' WHERE ({search_query})'
            + group_filter_query
            + """
            RETURN
            """
            + EPISODIC_NODE_RETURN
            + """
            LIMIT $limit
            """
        )

        records, _, _ = await executor.execute_query(
            cypher, limit=limit, routing_='r', **filter_params
        )

        return [episodic_node_from_record(r) for r in records]

    # --- Community search ---

    async def community_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[CommunityNode]:
        fuzzy_query = _build_arcadedb_fulltext_query(query, group_ids)
        if fuzzy_query == '':
            return []

        filter_params: dict[str, Any] = {}
        group_filter_query = ''
        if group_ids is not None:
            group_filter_query = 'AND c.group_id IN $group_ids'
            filter_params['group_ids'] = group_ids

        # Use CONTAINS for fulltext search
        search_terms = lucene_sanitize(query).split()
        if not search_terms:
            return []

        search_clauses = []
        for i, term in enumerate(search_terms[:5]):
            param_key = f'term_{i}'
            filter_params[param_key] = term.lower()
            search_clauses.append(f'toLower(c.name) CONTAINS ${param_key}')

        search_query = ' OR '.join(search_clauses)

        cypher = (
            'MATCH (c:Community)'
            + f' WHERE ({search_query})'
            + (' ' + group_filter_query if group_filter_query else '')
            + """
            RETURN
            """
            + COMMUNITY_NODE_RETURN
            + """
            LIMIT $limit
            """
        )

        records, _, _ = await executor.execute_query(
            cypher, limit=limit, routing_='r', **filter_params
        )

        return [community_node_from_record(r) for r in records]

    async def community_similarity_search(
        self,
        executor: QueryExecutor,
        search_vector: list[float],
        group_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.6,
    ) -> list[CommunityNode]:
        query_params: dict[str, Any] = {}

        group_filter_query = ''
        if group_ids is not None:
            group_filter_query += ' WHERE c.group_id IN $group_ids'
            query_params['group_ids'] = group_ids

        # Fetch candidate communities with embeddings
        cypher = (
            'MATCH (c:Community)'
            + group_filter_query
            + """
            WHERE c.name_embedding IS NOT NULL
            RETURN
            """
            + COMMUNITY_NODE_RETURN
        )

        records, _, _ = await executor.execute_query(
            cypher,
            routing_='r',
            **query_params,
        )

        # Compute cosine similarity in Python and filter/sort
        scored_records = []
        for r in records:
            embedding = r.get('name_embedding')
            if embedding is not None:
                score = _cosine_similarity(embedding, search_vector)
                if score > min_score:
                    scored_records.append((score, r))

        scored_records.sort(key=lambda x: x[0], reverse=True)
        scored_records = scored_records[:limit]

        return [community_node_from_record(r) for _, r in scored_records]

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

        cypher = """
        UNWIND $node_uuids AS node_uuid
        MATCH (center:Entity {uuid: $center_uuid})-[:RELATES_TO]-(n:Entity {uuid: node_uuid})
        RETURN 1 AS score, node_uuid AS uuid
        """

        results, _, _ = await executor.execute_query(
            cypher,
            node_uuids=filtered_uuids,
            center_uuid=center_node_uuid,
            routing_='r',
        )

        for result in results:
            scores[result['uuid']] = result['score']

        for uuid in filtered_uuids:
            if uuid not in scores:
                scores[uuid] = float('inf')

        filtered_uuids.sort(key=lambda cur_uuid: scores[cur_uuid])

        if center_node_uuid in node_uuids:
            scores[center_node_uuid] = 0.1
            filtered_uuids = [center_node_uuid] + filtered_uuids

        reranked_uuids = [u for u in filtered_uuids if (1 / scores[u]) >= min_score]

        if not reranked_uuids:
            return []

        get_query = """
            MATCH (n:Entity)
            WHERE n.uuid IN $uuids
            RETURN
            """ + get_entity_node_return_query(GraphProvider.ARCADEDB)

        records, _, _ = await executor.execute_query(get_query, uuids=reranked_uuids, routing_='r')

        node_map = {r['uuid']: entity_node_from_record(r) for r in records}
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

        results, _, _ = await executor.execute_query(
            """
            UNWIND $node_uuids AS node_uuid
            MATCH (episode:Episodic)-[r:MENTIONS]->(n:Entity {uuid: node_uuid})
            RETURN count(*) AS score, n.uuid AS uuid
            """,
            node_uuids=node_uuids,
            routing_='r',
        )

        for result in results:
            scores[result['uuid']] = result['score']

        for uuid in node_uuids:
            if uuid not in scores:
                scores[uuid] = float('inf')

        sorted_uuids = list(node_uuids)
        sorted_uuids.sort(key=lambda cur_uuid: scores[cur_uuid])

        reranked_uuids = [u for u in sorted_uuids if scores[u] >= min_score]

        if not reranked_uuids:
            return []

        get_query = """
            MATCH (n:Entity)
            WHERE n.uuid IN $uuids
            RETURN
            """ + get_entity_node_return_query(GraphProvider.ARCADEDB)

        records, _, _ = await executor.execute_query(get_query, uuids=reranked_uuids, routing_='r')

        node_map = {r['uuid']: entity_node_from_record(r) for r in records}
        return [node_map[u] for u in reranked_uuids if u in node_map]

    # --- Filter builders ---

    def build_node_search_filters(self, search_filters: SearchFilters) -> Any:
        filter_queries, filter_params = node_search_filter_query_constructor(
            search_filters, GraphProvider.ARCADEDB
        )
        return {'filter_queries': filter_queries, 'filter_params': filter_params}

    def build_edge_search_filters(self, search_filters: SearchFilters) -> Any:
        filter_queries, filter_params = edge_search_filter_query_constructor(
            search_filters, GraphProvider.ARCADEDB
        )
        return {'filter_queries': filter_queries, 'filter_params': filter_params}

    # --- Fulltext query builder ---

    def build_fulltext_query(
        self,
        query: str,
        group_ids: list[str] | None = None,
        max_query_length: int = 8000,
    ) -> str:
        return _build_arcadedb_fulltext_query(query, group_ids, max_query_length)
