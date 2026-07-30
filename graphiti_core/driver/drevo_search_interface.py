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
from typing import Any

from graphiti_core.driver.search_interface.search_interface import SearchInterface
from graphiti_core.edges import get_entity_edge_from_record
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_return_query
from graphiti_core.models.nodes.node_db_queries import get_entity_node_return_query
from graphiti_core.nodes import get_entity_node_from_record
from graphiti_core.search.search_filters import (
    edge_search_filter_query_constructor,
    node_search_filter_query_constructor,
)

logger = logging.getLogger(__name__)


class DrevoSearchInterface(SearchInterface):
    """Search overlay for the drevo backend.

    Routed through ``driver.search_interface`` (honored by
    ``graphiti_core.search.search_utils``), so all search for a drevo driver is
    served here instead of by the inline provider-branched Cypher, which assumes
    Neo4j-style fulltext/vector index procedures that drevo does not implement.

    Vector similarity uses drevo's native ``cosine_similarity(a, b)`` Cypher
    scalar (drevo#202 request #1), so ranking happens server-side in a single
    query — the same shape as the Neo4j inline path, only the scalar differs.
    Fulltext, BFS, community search and the rerankers are added in subsequent
    slices; drevo has no Bolt/Cypher full-text search yet (tracked as drevo#208).
    """

    async def node_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[Any]:
        filter_queries, filter_params = node_search_filter_query_constructor(
            search_filter, driver.provider
        )
        if group_ids is not None:
            filter_queries.append('n.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids
        # cosine_similarity errors on a null/absent embedding, so exclude those rows.
        filter_queries.append('n.name_embedding IS NOT NULL')
        filter_query = ' WHERE ' + ' AND '.join(filter_queries)

        query = (
            f'MATCH (n:Entity){filter_query} '
            'WITH n, cosine_similarity(n.name_embedding, $search_vector) AS score '
            'WHERE score > $min_score '
            'RETURN ' + get_entity_node_return_query(driver.provider) + ' '
            'ORDER BY score DESC LIMIT $limit'
        )
        records, _, _ = await driver.execute_query(
            query,
            search_vector=search_vector,
            min_score=min_score,
            limit=limit,
            **filter_params,
        )
        return [get_entity_node_from_record(record, driver.provider) for record in records]

    async def edge_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        source_node_uuid: str | None,
        target_node_uuid: str | None,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[Any]:
        filter_queries, filter_params = edge_search_filter_query_constructor(
            search_filter, driver.provider
        )
        if group_ids is not None:
            filter_queries.append('e.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids
        if source_node_uuid is not None:
            filter_queries.append('n.uuid = $source_uuid')
            filter_params['source_uuid'] = source_node_uuid
        if target_node_uuid is not None:
            filter_queries.append('m.uuid = $target_uuid')
            filter_params['target_uuid'] = target_node_uuid
        filter_queries.append('e.fact_embedding IS NOT NULL')
        filter_query = ' WHERE ' + ' AND '.join(filter_queries)

        query = (
            f'MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity){filter_query} '
            'WITH e, n, m, cosine_similarity(e.fact_embedding, $search_vector) AS score '
            'WHERE score > $min_score '
            'RETURN ' + get_entity_edge_return_query(driver.provider) + ' '
            'ORDER BY score DESC LIMIT $limit'
        )
        records, _, _ = await driver.execute_query(
            query,
            search_vector=search_vector,
            min_score=min_score,
            limit=limit,
            **filter_params,
        )
        return [get_entity_edge_from_record(record, driver.provider) for record in records]
