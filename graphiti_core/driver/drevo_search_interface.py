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

from pydantic import PrivateAttr

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
from graphiti_core.search.search_utils import calculate_cosine_similarity

logger = logging.getLogger(__name__)


def _is_cosine_unsupported_error(exc: Exception) -> bool:
    """True if ``exc`` is drevo rejecting the ``cosine_similarity`` scalar.

    Older drevo builds (before the Phase-10 scalar, drevo#202) answer a
    ``cosine_similarity(...)`` call with an "unsupported Cypher feature" database
    error. We match on the message rather than an exception type to stay decoupled
    from the neo4j driver, and only this specific signature triggers the fallback —
    any other error propagates.
    """
    message = str(exc).lower()
    return 'unsupported' in message and 'cosine_similarity' in message


def _is_fts_unsupported_error(exc: Exception) -> bool:
    """True if ``exc`` is drevo rejecting the ``fts.search`` procedure.

    Older drevo builds (before the FTS-over-Cypher work, drevo#208/#227) answer a
    ``CALL fts.search(...)`` with an "unsupported"/"unknown procedure" database
    error. Matched on the message to stay decoupled from the neo4j driver; any
    other error propagates.
    """
    message = str(exc).lower()
    return 'fts.search' in message and (
        'unsupported' in message or 'unknown' in message or 'no such procedure' in message
    )


# fts.search returns the global BM25 top-k, which we then post-filter by label and
# group. Over-fetch so enough survive the filter to fill `limit`.
def _fts_k(limit: int) -> int:
    return max(limit * 10, 100)


def _rank_records_by_cosine(
    records: list[dict[str, Any]],
    search_vector: list[float],
    min_score: float,
    limit: int,
    embedding_key: str,
) -> list[dict[str, Any]]:
    """Rank rows by cosine similarity of ``embedding_key`` to ``search_vector``.

    The library-side fallback used when a drevo build lacks the native
    ``cosine_similarity`` scalar (drevo#202): like the Neptune backend, the
    connector fetches candidate embeddings (returned as plain lists over Bolt) and
    ranks them in Python. Rows without an embedding are skipped; rows scoring at or
    below ``min_score`` are dropped; the rest are sorted by score descending and
    truncated to ``limit``.
    """
    scored: list[tuple[dict[str, Any], float]] = []
    for record in records:
        embedding = record.get(embedding_key)
        if embedding is None:
            continue
        score = calculate_cosine_similarity(search_vector, [float(x) for x in embedding])
        if score > min_score:
            scored.append((record, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return [record for record, _ in scored[:limit]]


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

    Vector similarity prefers drevo's native ``cosine_similarity`` scalar (drevo#202)
    for a single server-side ranked query. Node/episode/community full-text prefers
    drevo's native ``CALL fts.search`` BM25 procedure (drevo#208/#227), post-filtered
    by label and group. Both fall back — to library-side cosine and lexical
    ``CONTAINS`` respectively — with one warning if the connected drevo is an older
    build that rejects the native feature. Edge full-text stays lexical because
    ``fts.search`` returns nodes only. BFS, rerankers and community embeddings are
    not overridden — they use the inline Neo4j-compatible Cypher drevo supports.
    """

    # None = not yet probed, True = native cosine_similarity works, False = fall back.
    _native_cosine: bool | None = PrivateAttr(default=None)
    # Same probe-once state for the native fts.search full-text procedure.
    _native_fts: bool | None = PrivateAttr(default=None)

    async def _ranked_similarity(
        self,
        driver: Any,
        *,
        native_cypher: str,
        native_params: dict[str, Any],
        fallback_cypher: str,
        fallback_params: dict[str, Any],
        search_vector: list[float],
        min_score: float,
        limit: int,
        embedding_key: str,
        parse: Any,
    ) -> list[Any]:
        """Run native cosine ranking, falling back to library-side on old drevo.

        Tries the native ``cosine_similarity`` query first (unless a previous call
        already found it unsupported). On the specific "unsupported cosine_similarity"
        database error it logs a single warning and switches to the library-side
        fallback for this and all subsequent calls; any other error propagates.
        """
        if self._native_cosine is not False:
            try:
                records, _, _ = await driver.execute_query(native_cypher, **native_params)
            except Exception as e:
                if not _is_cosine_unsupported_error(e):
                    raise
                if self._native_cosine is None:
                    logger.warning(
                        'drevo rejected the native cosine_similarity() scalar '
                        '(unsupported Cypher feature); falling back to slower '
                        'library-side vector ranking for this session. Upgrade drevo '
                        'to a build with the cosine_similarity scalar (drevo#202) for '
                        'server-side ranking.'
                    )
                self._native_cosine = False
            else:
                self._native_cosine = True
                return [parse(record) for record in records]

        records, _, _ = await driver.execute_query(fallback_cypher, **fallback_params)
        ranked = _rank_records_by_cosine(records, search_vector, min_score, limit, embedding_key)
        return [parse(record) for record in ranked]

    async def _fulltext(
        self,
        driver: Any,
        *,
        native_cypher: str,
        native_params: dict[str, Any],
        parse: Any,
        fallback: Any,
    ) -> list[Any]:
        """Run native ``fts.search`` BM25, falling back to lexical on old drevo.

        Tries the native procedure first (unless a previous call found it
        unsupported). On the specific "unsupported fts.search" error it logs a
        single warning and switches to the lexical ``CONTAINS`` fallback for this
        and all subsequent calls; any other error propagates.
        """
        if self._native_fts is not False:
            try:
                records, _, _ = await driver.execute_query(native_cypher, **native_params)
            except Exception as e:
                if not _is_fts_unsupported_error(e):
                    raise
                if self._native_fts is None:
                    logger.warning(
                        'drevo rejected the native fts.search procedure (unsupported); '
                        'falling back to lexical CONTAINS full-text search for this '
                        'session. Upgrade drevo to a build with fts.search over the '
                        'indexed node properties (drevo#208/#227) for BM25 ranking.'
                    )
                self._native_fts = False
            else:
                self._native_fts = True
                return [parse(record) for record in records]

        return await fallback()

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
        filter_queries.append('n.name_embedding IS NOT NULL')
        filter_query = ' WHERE ' + ' AND '.join(filter_queries)

        return_query = get_entity_node_return_query(driver.provider)
        native_cypher = (
            f'MATCH (n:Entity){filter_query} '
            'WITH n, cosine_similarity(n.name_embedding, $search_vector) AS score '
            'WHERE score > $min_score '
            'RETURN ' + return_query + ' '
            'ORDER BY score DESC LIMIT $limit'
        )
        fallback_cypher = (
            f'MATCH (n:Entity){filter_query} RETURN '
            + return_query
            + ', n.name_embedding AS _search_embedding'
        )
        return await self._ranked_similarity(
            driver,
            native_cypher=native_cypher,
            native_params={
                'search_vector': search_vector,
                'min_score': min_score,
                'limit': limit,
                **filter_params,
            },
            fallback_cypher=fallback_cypher,
            fallback_params=dict(filter_params),
            search_vector=search_vector,
            min_score=min_score,
            limit=limit,
            embedding_key='_search_embedding',
            parse=lambda record: get_entity_node_from_record(record, driver.provider),
        )

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

        return_query = get_entity_edge_return_query(driver.provider)
        match = f'MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity){filter_query}'
        native_cypher = (
            f'{match} '
            'WITH e, n, m, cosine_similarity(e.fact_embedding, $search_vector) AS score '
            'WHERE score > $min_score '
            'RETURN ' + return_query + ' '
            'ORDER BY score DESC LIMIT $limit'
        )
        fallback_cypher = (
            f'{match} RETURN ' + return_query + ', e.fact_embedding AS _search_embedding'
        )
        return await self._ranked_similarity(
            driver,
            native_cypher=native_cypher,
            native_params={
                'search_vector': search_vector,
                'min_score': min_score,
                'limit': limit,
                **filter_params,
            },
            fallback_cypher=fallback_cypher,
            fallback_params=dict(filter_params),
            search_vector=search_vector,
            min_score=min_score,
            limit=limit,
            embedding_key='_search_embedding',
            parse=lambda record: get_entity_edge_from_record(record, driver.provider),
        )

    async def node_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        if not query.strip():
            return []

        filter_queries, filter_params = node_search_filter_query_constructor(
            search_filter, driver.provider
        )
        if group_ids is not None:
            filter_queries.append('n.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids
        where = " WHERE 'Entity' IN labels(n)"
        if filter_queries:
            where += ' AND ' + ' AND '.join(filter_queries)

        native_cypher = (
            'CALL fts.search($fts_query, $fts_k) YIELD node, score '
            'WITH node AS n, score' + where + ' '
            'RETURN ' + get_entity_node_return_query(driver.provider) + ' '
            'ORDER BY score DESC LIMIT $limit'
        )
        native_params = {
            'fts_query': query,
            'fts_k': _fts_k(limit),
            'limit': limit,
            **filter_params,
        }
        return await self._fulltext(
            driver,
            native_cypher=native_cypher,
            native_params=native_params,
            parse=lambda record: get_entity_node_from_record(record, driver.provider),
            fallback=lambda: self._lexical_node_fulltext(
                driver, query, search_filter, group_ids, limit
            ),
        )

    async def _lexical_node_fulltext(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None,
        limit: int,
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
        # Episodes carry no entity/edge-type filter (search_filter unused, matching
        # the inline path), only group scoping.
        if not query.strip():
            return []

        where = " WHERE 'Episodic' IN labels(e)"
        native_params: dict[str, Any] = {
            'fts_query': query,
            'fts_k': _fts_k(limit),
            'limit': limit,
        }
        if group_ids is not None:
            where += ' AND e.group_id IN $group_ids'
            native_params['group_ids'] = group_ids

        native_cypher = (
            'CALL fts.search($fts_query, $fts_k) YIELD node, score '
            'WITH node AS e, score' + where + ' '
            'RETURN ' + EPISODIC_NODE_RETURN + ' '
            'ORDER BY score DESC LIMIT $limit'
        )
        return await self._fulltext(
            driver,
            native_cypher=native_cypher,
            native_params=native_params,
            parse=get_episodic_node_from_record,
            fallback=lambda: self._lexical_episode_fulltext(driver, query, group_ids, limit),
        )

    async def _lexical_episode_fulltext(
        self,
        driver: Any,
        query: str,
        group_ids: list[str] | None,
        limit: int,
    ) -> list[Any]:
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
        if not query.strip():
            return []

        where = " WHERE 'Community' IN labels(c)"
        native_params: dict[str, Any] = {
            'fts_query': query,
            'fts_k': _fts_k(limit),
            'limit': limit,
        }
        if group_ids is not None:
            where += ' AND c.group_id IN $group_ids'
            native_params['group_ids'] = group_ids

        native_cypher = (
            'CALL fts.search($fts_query, $fts_k) YIELD node, score '
            'WITH node AS c, score' + where + ' '
            'RETURN ' + COMMUNITY_NODE_RETURN + ' '
            'ORDER BY score DESC LIMIT $limit'
        )
        return await self._fulltext(
            driver,
            native_cypher=native_cypher,
            native_params=native_params,
            parse=get_community_node_from_record,
            fallback=lambda: self._lexical_community_fulltext(driver, query, group_ids, limit),
        )

    async def _lexical_community_fulltext(
        self,
        driver: Any,
        query: str,
        group_ids: list[str] | None,
        limit: int,
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

        # COMMUNITY_NODE_RETURN already projects `c.name_embedding AS name_embedding`,
        # which doubles as the fallback ranking key.
        native_cypher = (
            f'MATCH (c:Community){filter_query} '
            'WITH c, cosine_similarity(c.name_embedding, $search_vector) AS score '
            'WHERE score > $min_score '
            'RETURN ' + COMMUNITY_NODE_RETURN + ' '
            'ORDER BY score DESC LIMIT $limit'
        )
        fallback_cypher = f'MATCH (c:Community){filter_query} RETURN ' + COMMUNITY_NODE_RETURN
        return await self._ranked_similarity(
            driver,
            native_cypher=native_cypher,
            native_params={
                'search_vector': search_vector,
                'min_score': min_score,
                'limit': limit,
                **filter_params,
            },
            fallback_cypher=fallback_cypher,
            fallback_params=dict(filter_params),
            search_vector=search_vector,
            min_score=min_score,
            limit=limit,
            embedding_key='name_embedding',
            parse=get_community_node_from_record,
        )
