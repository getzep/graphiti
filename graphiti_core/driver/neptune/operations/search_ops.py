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

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_return_query
from graphiti_core.models.nodes.node_db_queries import (
    COMMUNITY_NODE_RETURN_NEPTUNE,
    EPISODIC_NODE_RETURN_NEPTUNE,
    get_entity_node_return_query,
)
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.search.search_filters import (
    SearchFilters,
    edge_search_filter_query_constructor,
    node_search_filter_query_constructor,
)
from graphiti_core.search.search_utils import calculate_cosine_similarity

if TYPE_CHECKING:
    from graphiti_core.driver.neptune_driver import NeptuneDriver

logger = logging.getLogger(__name__)


class NeptuneSearchOperations(SearchOperations):
    def __init__(self, driver: NeptuneDriver | None = None):
        self._driver = driver

    # --- Node search ---

    async def node_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityNode]:
        if self._driver is None:
            return []
        driver = self._driver
        res = driver.run_aoss_query('node_name_and_summary', query, limit=limit)
        if not res or res.get('hits', {}).get('total', {}).get('value', 0) == 0:
            return []

        input_ids = []
        for r in res['hits']['hits']:
            input_ids.append({'id': r['_source']['uuid'], 'score': r['_score']})

        cypher = (
            """
            UNWIND $ids as i
            MATCH (n:Entity)
            WHERE n.uuid=i.id
            RETURN
            """
            + get_entity_node_return_query(GraphProvider.NEPTUNE)
            + """
            ORDER BY i.score DESC
            LIMIT $limit
            """
        )

        records, _, _ = await executor.execute_query(
            cypher,
            ids=input_ids,
            limit=limit,
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
            search_filter, GraphProvider.NEPTUNE
        )

        if group_ids is not None:
            filter_queries.append('n.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

        filter_query = ''
        if filter_queries:
            filter_query = ' WHERE ' + (' AND '.join(filter_queries))

        # Neptune: fetch all embeddings, compute cosine in Python
        query = (
            'MATCH (n:Entity)'
            + filter_query
            + """
            RETURN DISTINCT id(n) as id, n.name_embedding as embedding
            """
        )
        resp, _, _ = await executor.execute_query(
            query,
            **filter_params,
        )

        if not resp:
            return []

        input_ids = []
        for r in resp:
            if r['embedding']:
                score = calculate_cosine_similarity(
                    search_vector, list(map(float, r['embedding'].split(',')))
                )
                if score > min_score:
                    input_ids.append({'id': r['id'], 'score': score})

        if not input_ids:
            return []

        cypher = (
            """
            UNWIND $ids as i
            MATCH (n:Entity)
            WHERE id(n)=i.id
            RETURN
            """
            + get_entity_node_return_query(GraphProvider.NEPTUNE)
            + """
            ORDER BY i.score DESC
            LIMIT $limit
            """
        )
        records, _, _ = await executor.execute_query(
            cypher,
            ids=input_ids,
            limit=limit,
        )

        return [entity_node_from_record(r) for r in records]

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
            search_filter, GraphProvider.NEPTUNE
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
            MATCH (origin {{uuid: origin_uuid}})-[e:RELATES_TO|MENTIONS*1..{max_depth}]->(n:Entity)
            WHERE (origin:Entity OR origin:Episodic)
            AND n.group_id = origin.group_id
            """
            + filter_query
            + """
            RETURN
            """
            + get_entity_node_return_query(GraphProvider.NEPTUNE)
            + """
            LIMIT $limit
            """
        )

        records, _, _ = await executor.execute_query(
            cypher,
            bfs_origin_node_uuids=origin_uuids,
            limit=limit,
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
        if self._driver is None:
            return []
        driver = self._driver
        res = driver.run_aoss_query('edge_name_and_fact', query)
        if not res or res.get('hits', {}).get('total', {}).get('value', 0) == 0:
            return []

        filter_queries, filter_params = edge_search_filter_query_constructor(
            search_filter, GraphProvider.NEPTUNE
        )

        if group_ids is not None:
            filter_queries.append('e.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

        filter_query = ''
        if filter_queries:
            filter_query = ' AND ' + (' AND '.join(filter_queries))

        input_ids = []
        for r in res['hits']['hits']:
            input_ids.append({'id': r['_source']['uuid'], 'score': r['_score']})

        cypher = (
            """
            UNWIND $ids as id
            MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
            WHERE e.uuid = id.id
            """
            + filter_query
            + """
            WITH e, id.score as score, n, m
            RETURN
            """
            + get_entity_edge_return_query(GraphProvider.NEPTUNE)
            + """
            ORDER BY score DESC LIMIT $limit
            """
        )

        records, _, _ = await executor.execute_query(
            cypher,
            ids=input_ids,
            limit=limit,
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
            search_filter, GraphProvider.NEPTUNE
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

        # Fetch all embeddings, compute cosine similarity in Python
        query = (
            'MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)'
            + filter_query
            + """
            RETURN DISTINCT id(e) as id, e.fact_embedding as embedding
            """
        )
        resp, _, _ = await executor.execute_query(
            query,
            **filter_params,
        )

        if not resp:
            return []

        input_ids = []
        for r in resp:
            if r['embedding']:
                score = calculate_cosine_similarity(
                    search_vector, list(map(float, r['embedding'].split(',')))
                )
                if score > min_score:
                    input_ids.append({'id': r['id'], 'score': score})

        if not input_ids:
            return []

        cypher = """
            UNWIND $ids as i
            MATCH ()-[r]->()
            WHERE id(r) = i.id
            RETURN
                r.uuid AS uuid,
                r.group_id AS group_id,
                startNode(r).uuid AS source_node_uuid,
                endNode(r).uuid AS target_node_uuid,
                r.created_at AS created_at,
                r.name AS name,
                r.fact AS fact,
                split(r.episodes, ",") AS episodes,
                r.expired_at AS expired_at,
                r.valid_at AS valid_at,
                r.invalid_at AS invalid_at,
                properties(r) AS attributes
            ORDER BY i.score DESC
            LIMIT $limit
        """
        records, _, _ = await executor.execute_query(
            cypher,
            ids=input_ids,
            limit=limit,
        )

        return [entity_edge_from_record(r) for r in records]

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
            search_filter, GraphProvider.NEPTUNE
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
            MATCH path = (origin {{uuid: origin_uuid}})-[:RELATES_TO|MENTIONS *1..{max_depth}]->(n:Entity)
            WHERE origin:Entity OR origin:Episodic
            UNWIND relationships(path) AS rel
            MATCH (n:Entity)-[e:RELATES_TO {{uuid: rel.uuid}}]-(m:Entity)
            """
            + filter_query
            + """
            RETURN DISTINCT
                e.uuid AS uuid,
                e.group_id AS group_id,
                startNode(e).uuid AS source_node_uuid,
                endNode(e).uuid AS target_node_uuid,
                e.created_at AS created_at,
                e.name AS name,
                e.fact AS fact,
                split(e.episodes, ',') AS episodes,
                e.expired_at AS expired_at,
                e.valid_at AS valid_at,
                e.invalid_at AS invalid_at,
                properties(e) AS attributes
            LIMIT $limit
            """
        )

        records, _, _ = await executor.execute_query(
            cypher,
            bfs_origin_node_uuids=origin_uuids,
            limit=limit,
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
        if self._driver is None:
            return []
        driver = self._driver
        res = driver.run_aoss_query('episode_content', query, limit=limit)
        if not res or res.get('hits', {}).get('total', {}).get('value', 0) == 0:
            return []

        input_ids = []
        for r in res['hits']['hits']:
            input_ids.append({'id': r['_source']['uuid'], 'score': r['_score']})

        cypher = (
            """
            UNWIND $ids as i
            MATCH (e:Episodic)
            WHERE e.uuid=i.id
            RETURN
            """
            + EPISODIC_NODE_RETURN_NEPTUNE
            + """
            ORDER BY i.score DESC
            LIMIT $limit
            """
        )

        records, _, _ = await executor.execute_query(
            cypher,
            ids=input_ids,
            limit=limit,
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
        if self._driver is None:
            return []
        driver = self._driver
        res = driver.run_aoss_query('community_name', query, limit=limit)
        if not res or res.get('hits', {}).get('total', {}).get('value', 0) == 0:
            return []

        input_ids = []
        for r in res['hits']['hits']:
            input_ids.append({'id': r['_source']['uuid'], 'score': r['_score']})

        cypher = (
            """
            UNWIND $ids as i
            MATCH (n:Community)
            WHERE n.uuid=i.id
            RETURN
        """
            + COMMUNITY_NODE_RETURN_NEPTUNE
            + """
            ORDER BY i.score DESC
            LIMIT $limit
        """
        )

        records, _, _ = await executor.execute_query(
            cypher,
            ids=input_ids,
            limit=limit,
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
            group_filter_query += ' WHERE n.group_id IN $group_ids'
            query_params['group_ids'] = group_ids

        query = (
            'MATCH (n:Community)'
            + group_filter_query
            + """
            RETURN DISTINCT id(n) as id, n.name_embedding as embedding
            """
        )
        resp, _, _ = await executor.execute_query(
            query,
            **query_params,
        )

        if not resp:
            return []

        input_ids = []
        for r in resp:
            if r['embedding']:
                score = calculate_cosine_similarity(
                    search_vector, list(map(float, r['embedding'].split(',')))
                )
                if score > min_score:
                    input_ids.append({'id': r['id'], 'score': score})

        if not input_ids:
            return []

        cypher = (
            """
            UNWIND $ids as i
            MATCH (n:Community)
            WHERE id(n)=i.id
            RETURN
        """
            + COMMUNITY_NODE_RETURN_NEPTUNE
            + """
            ORDER BY i.score DESC
            LIMIT $limit
        """
        )

        records, _, _ = await executor.execute_query(
            cypher,
            ids=input_ids,
            limit=limit,
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

        cypher = """
        UNWIND $node_uuids AS node_uuid
        MATCH (center:Entity {uuid: $center_uuid})-[:RELATES_TO]-(n:Entity {uuid: node_uuid})
        RETURN 1 AS score, node_uuid AS uuid
        """

        results, _, _ = await executor.execute_query(
            cypher,
            node_uuids=filtered_uuids,
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

        get_query = """
            MATCH (n:Entity)
            WHERE n.uuid IN $uuids
            RETURN
            """ + get_entity_node_return_query(GraphProvider.NEPTUNE)

        records, _, _ = await executor.execute_query(get_query, uuids=reranked_uuids)

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
            """ + get_entity_node_return_query(GraphProvider.NEPTUNE)

        records, _, _ = await executor.execute_query(get_query, uuids=reranked_uuids)

        node_map = {r['uuid']: entity_node_from_record(r) for r in records}
        return [node_map[u] for u in reranked_uuids if u in node_map]

    # --- Filter builders ---

    def build_node_search_filters(self, search_filters: SearchFilters) -> Any:
        filter_queries, filter_params = node_search_filter_query_constructor(
            search_filters, GraphProvider.NEPTUNE
        )
        return {'filter_queries': filter_queries, 'filter_params': filter_params}

    def build_edge_search_filters(self, search_filters: SearchFilters) -> Any:
        filter_queries, filter_params = edge_search_filter_query_constructor(
            search_filters, GraphProvider.NEPTUNE
        )
        return {'filter_queries': filter_queries, 'filter_params': filter_params}

    # --- Fulltext query builder ---

    def build_fulltext_query(
        self,
        query: str,
        group_ids: list[str] | None = None,
        max_query_length: int = 8000,
    ) -> str:
        # Neptune uses AOSS for fulltext, so this is not used directly
        return query
