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

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.kuzu.operations.record_parsers import (
    parse_kuzu_entity_edge,
    parse_kuzu_entity_node,
)
from graphiti_core.driver.operations.search_ops import SearchOperations
from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.driver.record_parsers import (
    community_node_from_record,
    episodic_node_from_record,
)
from graphiti_core.edges import EntityEdge
from graphiti_core.graph_queries import (
    get_nodes_query,
    get_relationships_query,
    get_vector_cosine_func_query,
)
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


def _build_kuzu_fulltext_query(
    query: str,
    group_ids: list[str] | None = None,  # noqa: ARG001
    max_query_length: int = MAX_QUERY_LENGTH,
) -> str:
    """Build a fulltext query string for Kuzu.

    Kuzu does not use Lucene syntax. The raw query is returned, truncated if it
    exceeds *max_query_length* words.
    """
    words = query.split()
    if len(words) >= max_query_length:
        words = words[:max_query_length]
    truncated = ' '.join(words)
    return truncated


class KuzuSearchOperations(SearchOperations):
    # --- Node search ---

    async def node_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityNode]:
        fuzzy_query = _build_kuzu_fulltext_query(query, group_ids)
        if fuzzy_query == '':
            return []

        filter_queries, filter_params = node_search_filter_query_constructor(
            search_filter, GraphProvider.KUZU
        )

        if group_ids is not None:
            filter_queries.append('n.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

        filter_query = ''
        if filter_queries:
            filter_query = ' WHERE ' + (' AND '.join(filter_queries))

        cypher = (
            get_nodes_query(
                'node_name_and_summary', '$query', limit=limit, provider=GraphProvider.KUZU
            )
            + ' WITH node AS n, score'
            + filter_query
            + """
            WITH n, score
            ORDER BY score DESC
            LIMIT $limit
            RETURN
            """
            + get_entity_node_return_query(GraphProvider.KUZU)
        )

        records, _, _ = await executor.execute_query(
            cypher,
            query=fuzzy_query,
            limit=limit,
            **filter_params,
        )

        return [parse_kuzu_entity_node(r) for r in records]

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
            search_filter, GraphProvider.KUZU
        )

        if group_ids is not None:
            filter_queries.append('n.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

        filter_query = ''
        if filter_queries:
            filter_query = ' WHERE ' + (' AND '.join(filter_queries))

        search_vector_var = f'CAST($search_vector AS FLOAT[{len(search_vector)}])'

        cypher = (
            'MATCH (n:Entity)'
            + filter_query
            + """
            WITH n, """
            + get_vector_cosine_func_query(
                'n.name_embedding', search_vector_var, GraphProvider.KUZU
            )
            + """ AS score
            WHERE score > $min_score
            RETURN
            """
            + get_entity_node_return_query(GraphProvider.KUZU)
            + """
            ORDER BY score DESC
            LIMIT $limit
            """
        )

        records, _, _ = await executor.execute_query(
            cypher,
            search_vector=search_vector,
            limit=limit,
            min_score=min_score,
            **filter_params,
        )

        return [parse_kuzu_entity_node(r) for r in records]

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
            search_filter, GraphProvider.KUZU
        )

        if group_ids is not None:
            filter_queries.append('n.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

        filter_query = ''
        if filter_queries:
            filter_query = ' AND ' + (' AND '.join(filter_queries))

        # Kuzu uses RelatesToNode_ as an intermediate node for edges, so each
        # logical hop is actually 2 hops in the graph.  We need 3 separate
        # MATCH queries UNIONed together:
        # 1. Episodic -> MENTIONS -> Entity (direct mention)
        # 2. Entity -> RELATES_TO*{2..depth*2} -> Entity (entity traversal)
        # 3. Episodic -> MENTIONS -> Entity -> RELATES_TO*{2..(depth-1)*2} -> Entity (combined)

        all_records: list[Any] = []

        for origin_uuid in origin_uuids:
            # Query 1: From Episodic origins via MENTIONS
            cypher_episodic = (
                """
                MATCH (origin:Episodic {uuid: $origin_uuid})-[:MENTIONS]->(n:Entity)
                WHERE n.group_id = origin.group_id
                """
                + filter_query
                + """
                RETURN
                """
                + get_entity_node_return_query(GraphProvider.KUZU)
                + """
                LIMIT $limit
                """
            )

            records, _, _ = await executor.execute_query(
                cypher_episodic,
                origin_uuid=origin_uuid,
                limit=limit,
                **filter_params,
            )
            all_records.extend(records)

            # Query 2: From Entity origins via RELATES_TO (doubled depth)
            doubled_depth = max_depth * 2
            cypher_entity = (
                f"""
                MATCH (origin:Entity {{uuid: $origin_uuid}})-[:RELATES_TO*2..{doubled_depth}]->(n:Entity)
                WHERE n.group_id = origin.group_id
                """
                + filter_query
                + """
                RETURN
                """
                + get_entity_node_return_query(GraphProvider.KUZU)
                + """
                LIMIT $limit
                """
            )

            records, _, _ = await executor.execute_query(
                cypher_entity,
                origin_uuid=origin_uuid,
                limit=limit,
                **filter_params,
            )
            all_records.extend(records)

            # Query 3: From Episodic through Entity (only if max_depth > 1)
            if max_depth > 1:
                combined_depth = (max_depth - 1) * 2
                cypher_combined = (
                    f"""
                    MATCH (origin:Episodic {{uuid: $origin_uuid}})-[:MENTIONS]->(:Entity)-[:RELATES_TO*2..{combined_depth}]->(n:Entity)
                    WHERE n.group_id = origin.group_id
                    """
                    + filter_query
                    + """
                    RETURN
                    """
                    + get_entity_node_return_query(GraphProvider.KUZU)
                    + """
                    LIMIT $limit
                    """
                )

                records, _, _ = await executor.execute_query(
                    cypher_combined,
                    origin_uuid=origin_uuid,
                    limit=limit,
                    **filter_params,
                )
                all_records.extend(records)

        # Deduplicate by uuid and limit
        seen: set[str] = set()
        unique_nodes: list[EntityNode] = []
        for r in all_records:
            node = parse_kuzu_entity_node(r)
            if node.uuid not in seen:
                seen.add(node.uuid)
                unique_nodes.append(node)
            if len(unique_nodes) >= limit:
                break

        return unique_nodes

    # --- Edge search ---

    async def edge_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityEdge]:
        fuzzy_query = _build_kuzu_fulltext_query(query, group_ids)
        if fuzzy_query == '':
            return []

        filter_queries, filter_params = edge_search_filter_query_constructor(
            search_filter, GraphProvider.KUZU
        )

        if group_ids is not None:
            filter_queries.append('e.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

        filter_query = ''
        if filter_queries:
            filter_query = ' WHERE ' + (' AND '.join(filter_queries))

        # Kuzu FTS for edges queries the RelatesToNode_ label, then we match
        # the full pattern to get source (n) and target (m) Entity nodes.
        cypher = (
            get_relationships_query('edge_name_and_fact', limit=limit, provider=GraphProvider.KUZU)
            + """
            WITH node AS e, score
            MATCH (n:Entity)-[:RELATES_TO]->(e)-[:RELATES_TO]->(m:Entity)
            """
            + filter_query
            + """
            WITH e, score, n, m
            RETURN
            """
            + get_entity_edge_return_query(GraphProvider.KUZU)
            + """
            ORDER BY score DESC
            LIMIT $limit
            """
        )

        records, _, _ = await executor.execute_query(
            cypher,
            query=fuzzy_query,
            limit=limit,
            **filter_params,
        )

        return [parse_kuzu_entity_edge(r) for r in records]

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
            search_filter, GraphProvider.KUZU
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

        search_vector_var = f'CAST($search_vector AS FLOAT[{len(search_vector)}])'

        cypher = (
            'MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m:Entity)'
            + filter_query
            + """
            WITH DISTINCT e, n, m, """
            + get_vector_cosine_func_query(
                'e.fact_embedding', search_vector_var, GraphProvider.KUZU
            )
            + """ AS score
            WHERE score > $min_score
            RETURN
            """
            + get_entity_edge_return_query(GraphProvider.KUZU)
            + """
            ORDER BY score DESC
            LIMIT $limit
            """
        )

        records, _, _ = await executor.execute_query(
            cypher,
            search_vector=search_vector,
            limit=limit,
            min_score=min_score,
            **filter_params,
        )

        return [parse_kuzu_entity_edge(r) for r in records]

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
            search_filter, GraphProvider.KUZU
        )

        if group_ids is not None:
            filter_queries.append('e.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

        filter_query = ''
        if filter_queries:
            filter_query = ' WHERE ' + (' AND '.join(filter_queries))

        # Because RelatesToNode_ doubles every hop, we need separate queries
        # similar to node BFS.
        all_records: list[Any] = []
        doubled_depth = max_depth * 2

        for origin_uuid in origin_uuids:
            # From Entity origins: traverse doubled depth to reach RelatesToNode_ edges
            cypher_entity = (
                f"""
                MATCH (origin:Entity {{uuid: $origin_uuid}})-[:RELATES_TO*2..{doubled_depth}]->(e:RelatesToNode_)
                MATCH (n:Entity)-[:RELATES_TO]->(e)-[:RELATES_TO]->(m:Entity)
                """
                + filter_query
                + """
                RETURN DISTINCT
                """
                + get_entity_edge_return_query(GraphProvider.KUZU)
                + """
                LIMIT $limit
                """
            )

            records, _, _ = await executor.execute_query(
                cypher_entity,
                origin_uuid=origin_uuid,
                limit=limit,
                **filter_params,
            )
            all_records.extend(records)

            # From Episodic origins: go through MENTIONS to Entity, then traverse
            cypher_episodic = (
                """
                MATCH (origin:Episodic {uuid: $origin_uuid})-[:MENTIONS]->(start:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m:Entity)
                MATCH (n:Entity)-[:RELATES_TO]->(e)
                """
                + filter_query
                + """
                RETURN DISTINCT
                """
                + get_entity_edge_return_query(GraphProvider.KUZU)
                + """
                LIMIT $limit
                """
            )

            records, _, _ = await executor.execute_query(
                cypher_episodic,
                origin_uuid=origin_uuid,
                limit=limit,
                **filter_params,
            )
            all_records.extend(records)

        # Deduplicate by uuid and limit
        seen: set[str] = set()
        unique_edges: list[EntityEdge] = []
        for r in all_records:
            edge = parse_kuzu_entity_edge(r)
            if edge.uuid not in seen:
                seen.add(edge.uuid)
                unique_edges.append(edge)
            if len(unique_edges) >= limit:
                break

        return unique_edges

    # --- Episode search ---

    async def episode_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,  # noqa: ARG002
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EpisodicNode]:
        fuzzy_query = _build_kuzu_fulltext_query(query, group_ids)
        if fuzzy_query == '':
            return []

        filter_params: dict[str, Any] = {}
        group_filter_query = ''
        if group_ids is not None:
            group_filter_query += '\nAND e.group_id IN $group_ids'
            filter_params['group_ids'] = group_ids

        cypher = (
            get_nodes_query('episode_content', '$query', limit=limit, provider=GraphProvider.KUZU)
            + """
            WITH node AS episode, score
            MATCH (e:Episodic)
            WHERE e.uuid = episode.uuid
            """
            + group_filter_query
            + """
            RETURN
            """
            + EPISODIC_NODE_RETURN
            + """
            ORDER BY score DESC
            LIMIT $limit
            """
        )

        records, _, _ = await executor.execute_query(
            cypher, query=fuzzy_query, limit=limit, **filter_params
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
        fuzzy_query = _build_kuzu_fulltext_query(query, group_ids)
        if fuzzy_query == '':
            return []

        filter_params: dict[str, Any] = {}
        group_filter_query = ''
        if group_ids is not None:
            group_filter_query = 'WHERE c.group_id IN $group_ids'
            filter_params['group_ids'] = group_ids

        cypher = (
            get_nodes_query('community_name', '$query', limit=limit, provider=GraphProvider.KUZU)
            + """
            WITH node AS c, score
            WITH c, score
            """
            + group_filter_query
            + """
            RETURN
            """
            + COMMUNITY_NODE_RETURN
            + """
            ORDER BY score DESC
            LIMIT $limit
            """
        )

        records, _, _ = await executor.execute_query(
            cypher, query=fuzzy_query, limit=limit, **filter_params
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

        search_vector_var = f'CAST($search_vector AS FLOAT[{len(search_vector)}])'

        cypher = (
            'MATCH (c:Community)'
            + group_filter_query
            + """
            WITH c,
            """
            + get_vector_cosine_func_query(
                'c.name_embedding', search_vector_var, GraphProvider.KUZU
            )
            + """ AS score
            WHERE score > $min_score
            RETURN
            """
            + COMMUNITY_NODE_RETURN
            + """
            ORDER BY score DESC
            LIMIT $limit
            """
        )

        records, _, _ = await executor.execute_query(
            cypher,
            search_vector=search_vector,
            limit=limit,
            min_score=min_score,
            **query_params,
        )

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

        # Kuzu does not support UNWIND, so query each UUID individually
        cypher = """
        MATCH (center:Entity {uuid: $center_uuid})-[:RELATES_TO]->(:RelatesToNode_)-[:RELATES_TO]-(n:Entity {uuid: $node_uuid})
        RETURN 1 AS score, n.uuid AS uuid
        """

        for node_uuid in filtered_uuids:
            results, _, _ = await executor.execute_query(
                cypher,
                node_uuid=node_uuid,
                center_uuid=center_node_uuid,
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

        # Fetch the actual EntityNode objects
        get_query = """
            MATCH (n:Entity)
            WHERE n.uuid IN $uuids
            RETURN
            """ + get_entity_node_return_query(GraphProvider.KUZU)

        records, _, _ = await executor.execute_query(get_query, uuids=reranked_uuids)

        node_map = {r['uuid']: parse_kuzu_entity_node(r) for r in records}
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

        # Kuzu does not support UNWIND, so query each UUID individually
        cypher = """
            MATCH (episode:Episodic)-[r:MENTIONS]->(n:Entity {uuid: $node_uuid})
            RETURN count(*) AS score, n.uuid AS uuid
        """
        for node_uuid in node_uuids:
            results, _, _ = await executor.execute_query(
                cypher,
                node_uuid=node_uuid,
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

        # Fetch the actual EntityNode objects
        get_query = """
            MATCH (n:Entity)
            WHERE n.uuid IN $uuids
            RETURN
            """ + get_entity_node_return_query(GraphProvider.KUZU)

        records, _, _ = await executor.execute_query(get_query, uuids=reranked_uuids)

        node_map = {r['uuid']: parse_kuzu_entity_node(r) for r in records}
        return [node_map[u] for u in reranked_uuids if u in node_map]

    # --- Filter builders ---

    def build_node_search_filters(self, search_filters: SearchFilters) -> Any:
        filter_queries, filter_params = node_search_filter_query_constructor(
            search_filters, GraphProvider.KUZU
        )
        return {'filter_queries': filter_queries, 'filter_params': filter_params}

    def build_edge_search_filters(self, search_filters: SearchFilters) -> Any:
        filter_queries, filter_params = edge_search_filter_query_constructor(
            search_filters, GraphProvider.KUZU
        )
        return {'filter_queries': filter_queries, 'filter_params': filter_params}

    # --- Fulltext query builder ---

    def build_fulltext_query(
        self,
        query: str,
        group_ids: list[str] | None = None,
        max_query_length: int = 8000,
    ) -> str:
        return _build_kuzu_fulltext_query(query, group_ids, max_query_length)
