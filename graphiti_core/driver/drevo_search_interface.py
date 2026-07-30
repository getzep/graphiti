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
from graphiti_core.search.search_utils import calculate_cosine_similarity

logger = logging.getLogger(__name__)


def rank_uuids_by_cosine(
    records: list[dict[str, Any]],
    search_vector: list[float],
    min_score: float,
    limit: int,
    *,
    uuid_key: str = 'uuid',
    embedding_key: str = 'embedding',
) -> list[tuple[str, float]]:
    """Rank candidate rows by cosine similarity to ``search_vector``, library-side.

    drevo's Cypher subset has no in-query cosine function (only the boolean
    ``similar()``), so — like the Neptune backend — the connector fetches candidate
    embeddings and ranks them in Python. Rows without an embedding are skipped;
    rows scoring at or below ``min_score`` are dropped; the rest are returned as
    ``(uuid, score)`` sorted by score descending and truncated to ``limit``.
    """
    scored: list[tuple[str, float]] = []
    for record in records:
        embedding = record.get(embedding_key)
        uuid = record.get(uuid_key)
        if embedding is None or uuid is None:
            continue
        score = calculate_cosine_similarity(search_vector, list(map(float, embedding)))
        if score > min_score:
            scored.append((uuid, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]


class DrevoSearchInterface(SearchInterface):
    """Search overlay for the drevo backend.

    Routed through ``driver.search_interface`` (honored by
    ``graphiti_core.search.search_utils``), so all search for a drevo driver is
    served here instead of by the inline provider-branched Cypher, which assumes
    Neo4j-style fulltext/vector index procedures that drevo does not implement.

    This first slice implements vector similarity (nodes and edges) using
    library-side cosine ranking. Native scored vector search over Cypher depends
    on drevo#202; until then ranking happens in Python. Fulltext, BFS, community
    search and the rerankers are added in subsequent slices.
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

        filter_query = (' WHERE ' + ' AND '.join(filter_queries)) if filter_queries else ''

        candidate_query = (
            f'MATCH (n:Entity){filter_query} RETURN n.uuid AS uuid, n.name_embedding AS embedding'
        )
        candidates, _, _ = await driver.execute_query(candidate_query, **filter_params)

        ranked = rank_uuids_by_cosine(candidates, search_vector, min_score, limit)
        if not ranked:
            return []

        ordered_uuids = [uuid for uuid, _ in ranked]
        fetch_query = (
            'UNWIND $uuids AS uuid MATCH (n:Entity {uuid: uuid}) RETURN '
            + get_entity_node_return_query(driver.provider)
        )
        records, _, _ = await driver.execute_query(fetch_query, uuids=ordered_uuids)

        nodes_by_uuid = {}
        for record in records:
            node = get_entity_node_from_record(record, driver.provider)
            nodes_by_uuid[node.uuid] = node

        return [nodes_by_uuid[uuid] for uuid in ordered_uuids if uuid in nodes_by_uuid]

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

        filter_query = (' WHERE ' + ' AND '.join(filter_queries)) if filter_queries else ''

        candidate_query = (
            f'MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity){filter_query} '
            'RETURN e.uuid AS uuid, e.fact_embedding AS embedding'
        )
        candidates, _, _ = await driver.execute_query(candidate_query, **filter_params)

        ranked = rank_uuids_by_cosine(candidates, search_vector, min_score, limit)
        if not ranked:
            return []

        ordered_uuids = [uuid for uuid, _ in ranked]
        fetch_query = (
            'UNWIND $uuids AS uuid MATCH (n:Entity)-[e:RELATES_TO {uuid: uuid}]->(m:Entity) RETURN '
            + get_entity_edge_return_query(driver.provider)
        )
        records, _, _ = await driver.execute_query(fetch_query, uuids=ordered_uuids)

        edges_by_uuid = {}
        for record in records:
            edge = get_entity_edge_from_record(record, driver.provider)
            edges_by_uuid[edge.uuid] = edge

        return [edges_by_uuid[uuid] for uuid in ordered_uuids if uuid in edges_by_uuid]
