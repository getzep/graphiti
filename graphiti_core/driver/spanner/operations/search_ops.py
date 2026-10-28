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

# Spanner search: full-text via SEARCH/SCORE, vectors via COSINE_DISTANCE (exact KNN), and
# traversals via GQL quantified paths. Semantics follow driver/neo4j/operations/search_ops.py;
# query text lives in search_common.py and SearchFilters compilation in ../filters.py.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from google.cloud.spanner_v1 import param_types

from graphiti_core.driver.operations.search_ops import SearchOperations
from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.driver.spanner.codecs import FLOAT32_ARRAY, STRING_ARRAY
from graphiti_core.driver.spanner.errors import SpannerDriverError
from graphiti_core.driver.spanner.filters import (
    SpannerSearchFilter,
    build_edge_filter,
    build_node_filter,
)
from graphiti_core.driver.spanner.fulltext import build_fulltext_query
from graphiti_core.driver.spanner.operations.base import spanner_driver
from graphiti_core.driver.spanner.operations.search_common import (
    COMMUNITY_ALIAS,
    COMMUNITY_COLUMNS,
    COMMUNITY_SOURCE,
    EDGE_ALIAS,
    EDGE_COLUMNS,
    ENTITIES_BY_UUIDS_SQL,
    ENTITY_ALIAS,
    ENTITY_COLUMNS,
    ENTITY_SOURCE,
    EPISODE_ALIAS,
    EPISODE_COLUMNS,
    EPISODE_SOURCE,
    MENTION_COUNTS_SQL,
    center_neighbors_gql,
    distance_rerank_order,
    edge_bfs_sql,
    edge_source,
    fulltext_sql,
    is_zero_vector,
    mentions_rerank_order,
    node_bfs_sql,
    similarity_sql,
    sort_by_score,
)
from graphiti_core.driver.spanner.rows import (
    community_node_from_row,
    entity_edge_from_row,
    entity_node_from_row,
    episodic_node_from_row,
)
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.search.search_filters import SearchFilters

if TYPE_CHECKING:
    # driver.py imports the ops modules; import the driver for annotations only.
    from graphiti_core.driver.spanner.driver import SpannerDriver


class _Query:
    """WHERE predicates plus typed parameters for one search statement."""

    def __init__(self, **params: tuple[Any, Any]):
        self.predicates: list[str] = []
        self.params: dict[str, Any] = {}
        self.types: dict[str, Any] = {}
        for name, (value, param_type) in params.items():
            self.bind(name, value, param_type)

    def bind(self, name: str, value: Any, param_type: Any) -> str:
        self.params[name] = value
        self.types[name] = param_type
        return f'@{name}'

    def where(self, predicate: str) -> None:
        self.predicates.append(predicate)

    def group_filter(self, column: str, group_ids: list[str] | None) -> None:
        """``column IN UNNEST(@group_ids)`` when ``group_ids`` is not None (None = all)."""
        if group_ids is not None:
            self.bind('group_ids', list(group_ids), STRING_ARRAY)
            self.where(f'{column} IN UNNEST(@group_ids)')

    def add_filter(self, compiled: SpannerSearchFilter, *, gql: bool) -> None:
        self.predicates.extend(compiled.gql if gql else compiled.sql)
        self.params.update(compiled.params)
        self.types.update(compiled.param_types)

    async def read(self, driver: SpannerDriver, sql: str) -> list[dict[str, Any]]:
        return await driver.read(sql, self.params, self.types)


def _check_vector(driver: SpannerDriver, search_vector: Sequence[float]) -> bool:
    """False when the vector cannot match (empty or all zeros). Raises on a dimension mismatch,
    which SAFE.COSINE_DISTANCE would otherwise turn into an empty result."""
    if is_zero_vector(search_vector):
        return False
    if driver.embedding_dim is not None and len(search_vector) != driver.embedding_dim:
        raise SpannerDriverError(
            f'Search vector has {len(search_vector)} dimensions but the Spanner schema uses '
            f'embedding_dim={driver.embedding_dim}'
        )
    return True


def _vector_query(search_vector: list[float], limit: int, min_score: float) -> _Query:
    return _Query(
        search_vector=([float(x) for x in search_vector], FLOAT32_ARRAY),
        limit=(limit, param_types.INT64),
        min_score=(float(min_score), param_types.FLOAT64),
    )


def _fulltext_query(rquery: str, limit: int) -> _Query:
    return _Query(query=(rquery, param_types.STRING), limit=(limit, param_types.INT64))


async def _entity_nodes_by_uuid(
    driver: SpannerDriver, uuids: Sequence[str]
) -> dict[str, EntityNode]:
    rows = await driver.read(
        ENTITIES_BY_UUIDS_SQL, {'uuids': list(dict.fromkeys(uuids))}, {'uuids': STRING_ARRAY}
    )
    return {row['uuid']: entity_node_from_row(row) for row in rows}


def _ordered_nodes(uuids: Sequence[str], nodes: Mapping[str, EntityNode]) -> list[EntityNode]:
    return [nodes[uuid] for uuid in uuids if uuid in nodes]


class SpannerSearchOperations(SearchOperations):
    """Spanner full-text (SEARCH/SCORE), vector (COSINE_DISTANCE) and GQL traversal search."""

    # --- Node search ---

    async def node_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityNode]:
        rquery = build_fulltext_query(query, group_ids)
        if not rquery:
            return []
        driver = spanner_driver(executor)
        q = _fulltext_query(rquery, limit)
        q.group_filter(f'{ENTITY_ALIAS}.group_id', group_ids)
        q.add_filter(build_node_filter(search_filter), gql=False)
        sql = fulltext_sql(ENTITY_COLUMNS, ENTITY_SOURCE, ENTITY_ALIAS, q.predicates)
        rows = sort_by_score(await q.read(driver, sql))
        return [entity_node_from_row(row) for row in rows]

    async def node_similarity_search(
        self,
        executor: QueryExecutor,
        search_vector: list[float],
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.6,
    ) -> list[EntityNode]:
        driver = spanner_driver(executor)
        if not _check_vector(driver, search_vector):
            return []
        q = _vector_query(search_vector, limit, min_score)
        q.group_filter(f'{ENTITY_ALIAS}.group_id', group_ids)
        q.add_filter(build_node_filter(search_filter), gql=False)
        sql = similarity_sql(
            ENTITY_COLUMNS, ENTITY_SOURCE, ENTITY_ALIAS, 'name_embedding', q.predicates
        )
        rows = sort_by_score(await q.read(driver, sql))
        return [entity_node_from_row(row) for row in rows]

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
        driver = spanner_driver(executor)
        q = _Query(
            origin_uuids=(list(origin_uuids), STRING_ARRAY),
            max_depth=(max_depth, param_types.INT64),
            limit=(limit, param_types.INT64),
        )
        # Same-group predicate from the Neo4j query: no cross-group hops even when
        # group_ids is None.
        q.where('n.group_id = origin.group_id')
        if group_ids is not None:
            q.group_filter('n.group_id', group_ids)
            q.where('origin.group_id IN UNNEST(@group_ids)')
        q.add_filter(build_node_filter(search_filter), gql=True)
        rows = await q.read(driver, node_bfs_sql(driver.graph_name, q.predicates))
        return [entity_node_from_row(row) for row in rows]

    # --- Edge search ---

    async def edge_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityEdge]:
        rquery = build_fulltext_query(query, group_ids)
        if not rquery:
            return []
        driver = spanner_driver(executor)
        compiled = build_edge_filter(search_filter)
        q = _fulltext_query(rquery, limit)
        q.group_filter(f'{EDGE_ALIAS}.group_id', group_ids)
        q.add_filter(compiled, gql=False)
        sql = fulltext_sql(
            EDGE_COLUMNS, edge_source(compiled.needs_endpoints), EDGE_ALIAS, q.predicates
        )
        rows = sort_by_score(await q.read(driver, sql))
        return [entity_edge_from_row(row) for row in rows]

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
        driver = spanner_driver(executor)
        if not _check_vector(driver, search_vector):
            return []
        compiled = build_edge_filter(search_filter)
        q = _vector_query(search_vector, limit, min_score)
        q.group_filter(f'{EDGE_ALIAS}.group_id', group_ids)
        # Applied whenever given (Neo4j applies them only when group_ids is set).
        if source_node_uuid is not None:
            q.bind('source_node_uuid', source_node_uuid, param_types.STRING)
            q.where(f'{EDGE_ALIAS}.uuid = @source_node_uuid')
        if target_node_uuid is not None:
            q.bind('target_node_uuid', target_node_uuid, param_types.STRING)
            q.where(f'{EDGE_ALIAS}.target_node_uuid = @target_node_uuid')
        q.add_filter(compiled, gql=False)
        sql = similarity_sql(
            EDGE_COLUMNS,
            edge_source(compiled.needs_endpoints),
            EDGE_ALIAS,
            'fact_embedding',
            q.predicates,
        )
        rows = sort_by_score(await q.read(driver, sql))
        return [entity_edge_from_row(row) for row in rows]

    async def edge_bfs_search(
        self,
        executor: QueryExecutor,
        origin_uuids: list[str],
        max_depth: int,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityEdge]:
        if not origin_uuids or max_depth < 1:
            return []
        driver = spanner_driver(executor)
        q = _Query(
            origin_uuids=(list(origin_uuids), STRING_ARRAY),
            limit=(limit, param_types.INT64),
        )
        multi_hop = max_depth > 1
        if multi_hop:
            q.bind('depth_minus_1', max_depth - 1, param_types.INT64)
        q.group_filter('e.group_id', group_ids)
        q.add_filter(build_edge_filter(search_filter), gql=True)
        rows = await q.read(driver, edge_bfs_sql(driver.graph_name, q.predicates, multi_hop))
        return [entity_edge_from_row(row) for row in rows]

    # --- Episode search ---

    async def episode_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,  # noqa: ARG002 - unused, as in Neo4j
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EpisodicNode]:
        rquery = build_fulltext_query(query, group_ids)
        if not rquery:
            return []
        driver = spanner_driver(executor)
        q = _fulltext_query(rquery, limit)
        q.group_filter(f'{EPISODE_ALIAS}.group_id', group_ids)
        sql = fulltext_sql(EPISODE_COLUMNS, EPISODE_SOURCE, EPISODE_ALIAS, q.predicates)
        rows = sort_by_score(await q.read(driver, sql))
        return [episodic_node_from_row(row) for row in rows]

    # --- Community search ---

    async def community_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[CommunityNode]:
        rquery = build_fulltext_query(query, group_ids)
        if not rquery:
            return []
        driver = spanner_driver(executor)
        q = _fulltext_query(rquery, limit)
        q.group_filter(f'{COMMUNITY_ALIAS}.group_id', group_ids)
        sql = fulltext_sql(COMMUNITY_COLUMNS, COMMUNITY_SOURCE, COMMUNITY_ALIAS, q.predicates)
        rows = sort_by_score(await q.read(driver, sql))
        return [community_node_from_row(row) for row in rows]

    async def community_similarity_search(
        self,
        executor: QueryExecutor,
        search_vector: list[float],
        group_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.6,
    ) -> list[CommunityNode]:
        driver = spanner_driver(executor)
        if not _check_vector(driver, search_vector):
            return []
        q = _vector_query(search_vector, limit, min_score)
        q.group_filter(f'{COMMUNITY_ALIAS}.group_id', group_ids)
        sql = similarity_sql(
            COMMUNITY_COLUMNS, COMMUNITY_SOURCE, COMMUNITY_ALIAS, 'name_embedding', q.predicates
        )
        rows = sort_by_score(await q.read(driver, sql))
        return [community_node_from_row(row) for row in rows]

    # --- Rerankers ---

    async def node_distance_reranker(
        self,
        executor: QueryExecutor,
        node_uuids: list[str],
        center_node_uuid: str,
        min_score: float = 0,
    ) -> list[EntityNode]:
        driver = spanner_driver(executor)
        candidates = list(dict.fromkeys(u for u in node_uuids if u != center_node_uuid))
        neighbors: list[str] = []
        if candidates:
            rows = await driver.read(
                center_neighbors_gql(driver.graph_name),
                {'center_uuid': center_node_uuid, 'node_uuids': candidates},
                {'center_uuid': param_types.STRING, 'node_uuids': STRING_ARRAY},
            )
            neighbors = [row['uuid'] for row in rows]
        reranked = distance_rerank_order(node_uuids, center_node_uuid, neighbors, min_score)
        if not reranked:
            return []
        return _ordered_nodes(reranked, await _entity_nodes_by_uuid(driver, reranked))

    async def episode_mentions_reranker(
        self,
        executor: QueryExecutor,
        node_uuids: list[str],
        min_score: float = 0,
    ) -> list[EntityNode]:
        if not node_uuids:
            return []
        driver = spanner_driver(executor)
        rows = await driver.read(
            MENTION_COUNTS_SQL,
            {'node_uuids': list(dict.fromkeys(node_uuids))},
            {'node_uuids': STRING_ARRAY},
        )
        counts = {row['uuid']: int(row['score']) for row in rows}
        reranked = mentions_rerank_order(node_uuids, counts, min_score)
        if not reranked:
            return []
        return _ordered_nodes(reranked, await _entity_nodes_by_uuid(driver, reranked))

    # --- Filter builders ---

    def build_node_search_filters(self, search_filters: SearchFilters) -> SpannerSearchFilter:
        return build_node_filter(search_filters)

    def build_edge_search_filters(self, search_filters: SearchFilters) -> SpannerSearchFilter:
        return build_edge_filter(search_filters)

    # --- Fulltext query builder ---

    def build_fulltext_query(
        self,
        query: str,
        group_ids: list[str] | None = None,
        max_query_length: int = 8000,
    ) -> str:
        return build_fulltext_query(query, group_ids, max_query_length)
