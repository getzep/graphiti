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
from graphiti_core.models.nodes.node_db_queries import (
    COMMUNITY_NODE_RETURN,
    EPISODIC_NODE_RETURN,
    get_entity_node_return_query,
)
from graphiti_core.nodes import (
    get_community_node_from_record,
    get_entity_node_from_record,
    get_episodic_node_from_record,
)
from graphiti_core.search.search_filters import (
    edge_search_filter_query_constructor,
    node_search_filter_query_constructor,
)

logger = logging.getLogger(__name__)


def _tokenize(query: str) -> list[str]:
    """Split a fulltext query into lowercased terms."""
    return [term for term in query.lower().split() if term]


def _lexical_match_clause(fields: list[str], term_count: int) -> str:
    """Build a Cypher predicate matching any term against any field.

    Interim substitute for a real fulltext index: drevo exposes no BM25 search
    over Bolt/Cypher yet (drevo#208), so the connector filters candidates with
    ``CONTAINS`` and ranks them by term overlap in Python. Parameters are named
    ``term_0``..``term_{n-1}``.
    """
    clauses = [
        f'toLower({field}) CONTAINS $term_{i}' for i in range(term_count) for field in fields
    ]
    return '(' + ' OR '.join(clauses) + ')'


def _rank_by_term_overlap(
    records: list[dict[str, Any]],
    text_keys: list[str],
    terms: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Rank rows by how many distinct query terms appear in their text fields."""
    scored: list[tuple[dict[str, Any], int]] = []
    for record in records:
        haystack = ' '.join(str(record.get(key) or '') for key in text_keys).lower()
        overlap = sum(1 for term in terms if term in haystack)
        if overlap > 0:
            scored.append((record, overlap))

    scored.sort(key=lambda item: item[1], reverse=True)
    return [record for record, _ in scored[:limit]]


class DrevoSearchInterface(SearchInterface):
    """Search overlay for the drevo backend.

    Routed through ``driver.search_interface`` (honored by
    ``graphiti_core.search.search_utils``), so all search for a drevo driver is
    served here instead of by the inline provider-branched Cypher, which assumes
    Neo4j-style fulltext/vector index procedures that drevo does not implement.

    Vector similarity uses drevo's native ``cosine_similarity(a, b)`` Cypher
    scalar (drevo#202 request #1), so ranking happens server-side in a single
    query — the same shape as the Neo4j inline path, only the scalar differs.

    Fulltext (node/edge) is an interim lexical match: drevo has no BM25 search
    over Bolt/Cypher yet (drevo#208), so candidates are filtered with ``CONTAINS``
    and ranked by term overlap in Python; it is swapped for the native path once
    drevo#208 lands. Episode/community fulltext, BFS and the rerankers follow in
    subsequent slices.
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

    async def node_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        terms = _tokenize(query)
        if not terms:
            return []

        filter_queries, filter_params = node_search_filter_query_constructor(
            search_filter, driver.provider
        )
        if group_ids is not None:
            filter_queries.append('n.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids
        filter_queries.append(_lexical_match_clause(['n.name', 'n.summary'], len(terms)))
        filter_query = ' WHERE ' + ' AND '.join(filter_queries)

        term_params = {f'term_{i}': term for i, term in enumerate(terms)}
        cypher = (
            f'MATCH (n:Entity){filter_query} RETURN '
            + get_entity_node_return_query(driver.provider)
            + ', n.name AS fulltext_name, n.summary AS fulltext_summary'
        )
        records, _, _ = await driver.execute_query(cypher, **filter_params, **term_params)

        ranked = _rank_by_term_overlap(records, ['fulltext_name', 'fulltext_summary'], terms, limit)
        return [get_entity_node_from_record(record, driver.provider) for record in ranked]

    async def edge_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        terms = _tokenize(query)
        if not terms:
            return []

        filter_queries, filter_params = edge_search_filter_query_constructor(
            search_filter, driver.provider
        )
        if group_ids is not None:
            filter_queries.append('e.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids
        filter_queries.append(_lexical_match_clause(['e.name', 'e.fact'], len(terms)))
        filter_query = ' WHERE ' + ' AND '.join(filter_queries)

        term_params = {f'term_{i}': term for i, term in enumerate(terms)}
        cypher = (
            f'MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity){filter_query} RETURN '
            + get_entity_edge_return_query(driver.provider)
            + ', e.name AS fulltext_name, e.fact AS fulltext_fact'
        )
        records, _, _ = await driver.execute_query(cypher, **filter_params, **term_params)

        ranked = _rank_by_term_overlap(records, ['fulltext_name', 'fulltext_fact'], terms, limit)
        return [get_entity_edge_from_record(record, driver.provider) for record in ranked]

    async def episode_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        # Episodes carry no entity/edge-type filter (search_filter is unused here,
        # matching the inline path), only group scoping + the lexical match.
        terms = _tokenize(query)
        if not terms:
            return []

        filter_queries: list[str] = []
        filter_params: dict[str, Any] = {}
        if group_ids is not None:
            filter_queries.append('e.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids
        filter_queries.append(
            _lexical_match_clause(['e.content', 'e.source', 'e.source_description'], len(terms))
        )
        filter_query = ' WHERE ' + ' AND '.join(filter_queries)

        term_params = {f'term_{i}': term for i, term in enumerate(terms)}
        cypher = (
            f'MATCH (e:Episodic){filter_query} RETURN '
            + EPISODIC_NODE_RETURN
            + ', e.content AS fulltext_content, e.source AS fulltext_source, '
            'e.source_description AS fulltext_source_description'
        )
        records, _, _ = await driver.execute_query(cypher, **filter_params, **term_params)

        ranked = _rank_by_term_overlap(
            records,
            ['fulltext_content', 'fulltext_source', 'fulltext_source_description'],
            terms,
            limit,
        )
        return [get_episodic_node_from_record(record) for record in ranked]

    async def community_fulltext_search(
        self,
        driver: Any,
        query: str,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        terms = _tokenize(query)
        if not terms:
            return []

        filter_queries: list[str] = []
        filter_params: dict[str, Any] = {}
        if group_ids is not None:
            filter_queries.append('c.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids
        filter_queries.append(_lexical_match_clause(['c.name'], len(terms)))
        filter_query = ' WHERE ' + ' AND '.join(filter_queries)

        term_params = {f'term_{i}': term for i, term in enumerate(terms)}
        cypher = (
            f'MATCH (c:Community){filter_query} RETURN '
            + COMMUNITY_NODE_RETURN
            + ', c.name AS fulltext_name'
        )
        records, _, _ = await driver.execute_query(cypher, **filter_params, **term_params)

        ranked = _rank_by_term_overlap(records, ['fulltext_name'], terms, limit)
        return [get_community_node_from_record(record) for record in ranked]

    async def community_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.6,
    ) -> list[Any]:
        filter_queries: list[str] = []
        filter_params: dict[str, Any] = {}
        if group_ids is not None:
            filter_queries.append('c.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids
        filter_queries.append('c.name_embedding IS NOT NULL')
        filter_query = ' WHERE ' + ' AND '.join(filter_queries)

        cypher = (
            f'MATCH (c:Community){filter_query} '
            'WITH c, cosine_similarity(c.name_embedding, $search_vector) AS score '
            'WHERE score > $min_score '
            'RETURN ' + COMMUNITY_NODE_RETURN + ' '
            'ORDER BY score DESC LIMIT $limit'
        )
        records, _, _ = await driver.execute_query(
            cypher,
            search_vector=search_vector,
            min_score=min_score,
            limit=limit,
            **filter_params,
        )
        return [get_community_node_from_record(record) for record in records]
