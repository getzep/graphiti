# Copyright 2026, Zep Software, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
from array import array
from collections import defaultdict
from typing import Any

from graphiti_core.driver.operations.search_ops import SearchOperations
from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.search.search_filters import SearchFilters

from ..codecs import (
    community_node_from_record,
    edge_from_record,
    encode_json,
    ensure_vector,
    entity_node_from_record,
    episode_node_from_record,
    vector_to_bind,
)
from ..errors import InvalidSearchFilter
from ..search.filters import OracleWhere, build_edge_where, build_node_where
from ..search.text import (
    build_edge_fulltext_sql,
    build_node_fulltext_sql,
    build_oracle_text_query,
)
from ..search.traversal import (
    build_bfs_sql,
    build_distance_neighbors_sql,
    build_mentions_reranker_sql,
    validate_traversal_depth,
)
from ..search.vector import (
    COMMUNITY_EMBEDDINGS_SQL,
    build_edge_vector_sql,
    build_node_vector_sql,
)
from ..sql.nodes import GET_ENTITIES_BY_UUIDS
from .base import resolve_group_ids, resolve_group_scope, run_read, validate_request_scope


def _validate_min_score(min_score: float) -> float:
    score = float(min_score)
    if not math.isfinite(score) or not -1.0 <= score <= 1.0:
        raise InvalidSearchFilter('min_score must be a finite value between -1 and 1')
    return score


def _vector(search_vector: list[float]) -> array:
    checked = vector_to_bind(search_vector)
    if checked is None:  # pragma: no cover - a list cannot normalize to None
        raise InvalidSearchFilter('search_vector is required')
    return checked


def _query_binds(
    where: OracleWhere,
    *,
    limit: int,
    **values: object,
) -> dict[str, object]:
    return {**dict(where.binds), **values, 'result_limit': limit}


def _graph_name(executor: QueryExecutor) -> str:
    config = getattr(executor, 'config', None)
    return str(getattr(config, 'graph_name', 'GRAPHITI_GRAPH'))


def _identifiers(values: list[str], *, field: str) -> list[str]:
    if len(values) > 1000:
        raise InvalidSearchFilter(f'{field} cannot contain more than 1000 values')
    normalized: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value or len(value) > 36:
            raise InvalidSearchFilter(f'{field} values must be non-empty strings up to 36 chars')
        normalized.add(value)
    return sorted(normalized)


def _rerank_min_score(min_score: float) -> float:
    score = float(min_score)
    if not math.isfinite(score) or score < 0:
        raise InvalidSearchFilter('reranker min_score must be finite and non-negative')
    return score


async def _scoped_entity_uuids(
    executor: QueryExecutor,
    uuids: list[str],
) -> list[str]:
    """Keep caller order while proving every reranker candidate belongs to the scope."""

    scope_group_id = resolve_group_scope(executor)
    if scope_group_id is None or not uuids:
        return uuids
    records = await run_read(
        executor,
        GET_ENTITIES_BY_UUIDS,
        uuids_json=encode_json(uuids),
        scope_group_id=scope_group_id,
    )
    existing = {str(record['uuid']) for record in records}
    return [uuid for uuid in uuids if uuid in existing]


class OracleSearchOperations(SearchOperations):
    async def node_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityNode]:
        group_ids = resolve_group_ids(executor, group_ids)
        if limit <= 0 or group_ids == []:
            return []
        where = build_node_where(search_filter, group_ids)
        text_query = build_oracle_text_query(query)
        records = await run_read(
            executor,
            build_node_fulltext_sql(kind='ENTITY', where=where),
            **_query_binds(where, limit=limit, text_query=text_query),
        )
        return [entity_node_from_record(record) for record in records]

    async def node_similarity_search(
        self,
        executor: QueryExecutor,
        search_vector: list[float],
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.6,
    ) -> list[EntityNode]:
        vector = _vector(search_vector)
        score = _validate_min_score(min_score)
        group_ids = resolve_group_ids(executor, group_ids)
        if limit <= 0 or group_ids == []:
            return []
        where = build_node_where(search_filter, group_ids)
        records = await run_read(
            executor,
            build_node_vector_sql(kind='ENTITY', where=where),
            **_query_binds(
                where,
                limit=limit,
                search_vector=vector,
                min_score=score,
            ),
        )
        return [entity_node_from_record(record) for record in records]

    async def node_bfs_search(
        self,
        executor: QueryExecutor,
        origin_uuids: list[str],
        search_filter: SearchFilters,
        max_depth: int,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityNode]:
        depth = validate_traversal_depth(max_depth)
        origins = _identifiers(origin_uuids, field='origin_uuids')
        group_ids = resolve_group_ids(executor, group_ids)
        if not origins or limit <= 0 or group_ids == []:
            return []
        where = build_node_where(search_filter, group_ids)
        sql, _ = build_bfs_sql(
            max_depth=depth,
            graph_name=_graph_name(executor),
            result='node',
            where_sql=where.sql,
        )
        records = await run_read(
            executor,
            sql,
            **_query_binds(
                where,
                limit=limit,
                origin_uuids_json=encode_json(origins),
            ),
        )
        return [entity_node_from_record(record) for record in records]

    async def edge_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityEdge]:
        group_ids = resolve_group_ids(executor, group_ids)
        if limit <= 0 or group_ids == []:
            return []
        where = build_edge_where(search_filter, group_ids)
        text_query = build_oracle_text_query(query)
        records = await run_read(
            executor,
            build_edge_fulltext_sql(where=where),
            **_query_binds(where, limit=limit, text_query=text_query),
        )
        return [edge_from_record(record) for record in records]  # type: ignore[misc]

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
        vector = _vector(search_vector)
        score = _validate_min_score(min_score)
        group_ids = resolve_group_ids(executor, group_ids)
        if limit <= 0 or group_ids == []:
            return []
        where = build_edge_where(search_filter, group_ids)
        endpoint_clauses: list[str] = []
        endpoint_binds: dict[str, object] = {}
        if source_node_uuid is not None:
            endpoint_clauses.append('e.source_uuid = :source_node_uuid')
            endpoint_binds['source_node_uuid'] = source_node_uuid
        if target_node_uuid is not None:
            endpoint_clauses.append('e.target_uuid = :target_node_uuid')
            endpoint_binds['target_node_uuid'] = target_node_uuid
        endpoint_sql = ' AND '.join(endpoint_clauses) or '1 = 1'
        records = await run_read(
            executor,
            build_edge_vector_sql(where=where, endpoint_sql=endpoint_sql),
            **_query_binds(
                where,
                limit=limit,
                search_vector=vector,
                min_score=score,
                **endpoint_binds,
            ),
        )
        return [edge_from_record(record) for record in records]  # type: ignore[misc]

    async def edge_bfs_search(
        self,
        executor: QueryExecutor,
        origin_uuids: list[str],
        max_depth: int,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityEdge]:
        depth = validate_traversal_depth(max_depth)
        origins = _identifiers(origin_uuids, field='origin_uuids')
        group_ids = resolve_group_ids(executor, group_ids)
        if not origins or limit <= 0 or group_ids == []:
            return []
        where = build_edge_where(search_filter, group_ids)
        sql, _ = build_bfs_sql(
            max_depth=depth,
            graph_name=_graph_name(executor),
            result='edge',
            where_sql=where.sql,
        )
        records = await run_read(
            executor,
            sql,
            **_query_binds(
                where,
                limit=limit,
                origin_uuids_json=encode_json(origins),
            ),
        )
        return [edge_from_record(record) for record in records]  # type: ignore[misc]

    async def episode_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EpisodicNode]:
        group_ids = resolve_group_ids(executor, group_ids)
        if limit <= 0 or group_ids == []:
            return []
        where = build_node_where(search_filter, group_ids)
        text_query = build_oracle_text_query(query)
        records = await run_read(
            executor,
            build_node_fulltext_sql(kind='EPISODIC', where=where),
            **_query_binds(where, limit=limit, text_query=text_query),
        )
        return [episode_node_from_record(record) for record in records]

    async def community_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[CommunityNode]:
        group_ids = resolve_group_ids(executor, group_ids)
        if limit <= 0 or group_ids == []:
            return []
        where = build_node_where(SearchFilters(), group_ids)
        text_query = build_oracle_text_query(query)
        records = await run_read(
            executor,
            build_node_fulltext_sql(kind='COMMUNITY', where=where),
            **_query_binds(where, limit=limit, text_query=text_query),
        )
        return [community_node_from_record(record) for record in records]

    async def community_similarity_search(
        self,
        executor: QueryExecutor,
        search_vector: list[float],
        group_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.6,
    ) -> list[CommunityNode]:
        vector = _vector(search_vector)
        score = _validate_min_score(min_score)
        group_ids = resolve_group_ids(executor, group_ids)
        if limit <= 0 or group_ids == []:
            return []
        where = build_node_where(SearchFilters(), group_ids)
        records = await run_read(
            executor,
            build_node_vector_sql(kind='COMMUNITY', where=where),
            **_query_binds(
                where,
                limit=limit,
                search_vector=vector,
                min_score=score,
            ),
        )
        return [community_node_from_record(record) for record in records]

    async def get_embeddings_for_communities(
        self,
        executor: QueryExecutor,
        communities: list[CommunityNode],
    ) -> dict[str, list[float]]:
        if not communities:
            return {}
        for community in communities:
            validate_request_scope(executor, kind='COMMUNITY', group_id=community.group_id)
        records = await run_read(
            executor,
            COMMUNITY_EMBEDDINGS_SQL,
            community_uuids_json=encode_json([community.uuid for community in communities]),
            scope_group_id=resolve_group_scope(executor),
        )
        embeddings: dict[str, list[float]] = {}
        for record in records:
            vector = ensure_vector(record.get('name_embedding'))
            if vector is not None:
                embeddings[str(record['uuid'])] = vector
        return embeddings

    async def node_distance_reranker(
        self,
        executor: QueryExecutor,
        node_uuids: list[str],
        center_node_uuid: str,
        min_score: float = 0,
    ) -> Any:
        threshold = _rerank_min_score(min_score)
        normalized = _identifiers(node_uuids, field='node_uuids')
        membership_candidates = list(normalized)
        if center_node_uuid not in membership_candidates:
            membership_candidates.append(center_node_uuid)
        scoped_members = set(await _scoped_entity_uuids(executor, membership_candidates))
        if resolve_group_scope(executor) is not None and center_node_uuid not in scoped_members:
            return [], []
        normalized = [uuid for uuid in normalized if uuid in scoped_members]
        if not normalized:
            return [], []

        includes_center = center_node_uuid in normalized
        candidates = [uuid for uuid in normalized if uuid != center_node_uuid]
        distances: dict[str, float] = {}
        if candidates:
            remaining = set(candidates)
            visited = {center_node_uuid}
            frontier = {center_node_uuid}
            for depth in range(1, 9):
                if not frontier or not remaining:
                    break
                records = await run_read(
                    executor,
                    build_distance_neighbors_sql(),
                    center_node_uuid=center_node_uuid,
                    frontier_uuids_json=encode_json(sorted(frontier)),
                    scope_group_id=resolve_group_scope(executor),
                )
                next_frontier = {
                    str(record['to_uuid'])
                    for record in records
                    if str(record['from_uuid']) in frontier
                    and str(record['to_uuid']) not in visited
                }
                for uuid in next_frontier & remaining:
                    distances[uuid] = float(depth)
                remaining.difference_update(next_frontier)
                visited.update(next_frontier)
                frontier = next_frontier

        ordered = sorted(candidates, key=lambda uuid: (distances.get(uuid, math.inf), uuid))
        scored = [
            (uuid, 0.0 if math.isinf(distances.get(uuid, math.inf)) else 1 / distances[uuid])
            for uuid in ordered
        ]
        if includes_center:
            scored.insert(0, (center_node_uuid, 0.1))
        filtered = [(uuid, score) for uuid, score in scored if score >= threshold]
        return [uuid for uuid, _ in filtered], [score for _, score in filtered]

    async def episode_mentions_reranker(
        self,
        executor: QueryExecutor,
        node_uuids: list[str] | list[list[str]],
        min_score: float = 0,
    ) -> Any:
        threshold = _rerank_min_score(min_score)
        if not node_uuids:
            return [], []

        if all(isinstance(value, str) for value in node_uuids):
            rankings = [node_uuids]
        elif all(isinstance(value, list) for value in node_uuids):
            rankings = node_uuids
        else:
            raise InvalidSearchFilter('node_uuids must be a list of strings or ranked lists')

        rrf_scores: defaultdict[str, float] = defaultdict(float)
        for ranking in rankings:
            seen: set[str] = set()
            for rank, uuid in enumerate(ranking):
                if not isinstance(uuid, str) or uuid in seen:
                    continue
                seen.add(uuid)
                rrf_scores[uuid] += 1 / (rank + 1)

        candidates = _identifiers(list(rrf_scores), field='node_uuids')
        candidates = await _scoped_entity_uuids(executor, candidates)
        if not candidates:
            return [], []
        records = await run_read(
            executor,
            build_mentions_reranker_sql(_graph_name(executor)),
            candidate_uuids_json=encode_json(candidates),
            scope_group_id=resolve_group_scope(executor),
        )
        mention_counts = {str(record['uuid']): float(record['mention_count']) for record in records}
        ordered = sorted(
            candidates,
            key=lambda uuid: (-mention_counts.get(uuid, 0.0), -rrf_scores[uuid], uuid),
        )
        filtered = [
            (uuid, mention_counts.get(uuid, 0.0))
            for uuid in ordered
            if mention_counts.get(uuid, 0.0) >= threshold
        ]
        return [uuid for uuid, _ in filtered], [score for _, score in filtered]

    def build_node_search_filters(self, search_filters: SearchFilters) -> OracleWhere:
        return build_node_where(search_filters)

    def build_edge_search_filters(self, search_filters: SearchFilters) -> OracleWhere:
        return build_edge_where(search_filters)

    def build_fulltext_query(
        self,
        query: str,
        group_ids: list[str] | None = None,
        max_query_length: int = 8000,
    ) -> str:
        del group_ids
        return build_oracle_text_query(query, max_query_length=max_query_length)
