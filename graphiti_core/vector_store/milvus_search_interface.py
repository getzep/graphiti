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

from pydantic import PrivateAttr

from graphiti_core.driver.search_interface.search_interface import SearchInterface
from graphiti_core.vector_store.milvus_utils import (
    COLLECTION_COMMUNITY_NODES,
    COLLECTION_ENTITY_EDGES,
    COLLECTION_ENTITY_NODES,
    COLLECTION_EPISODIC_NODES,
    build_group_ids_filter,
    build_milvus_filter_from_search_filters,
    milvus_dict_to_community_node,
    milvus_dict_to_entity_edge,
    milvus_dict_to_entity_node,
    milvus_dict_to_episodic_node,
)

logger = logging.getLogger(__name__)

# Output field lists for each collection (excludes embedding vectors for efficiency)
_NODE_OUTPUT_FIELDS = [
    'uuid', 'group_id', 'name', 'summary', 'labels', 'created_at', 'attributes',
]
_EDGE_OUTPUT_FIELDS = [
    'uuid', 'group_id', 'source_node_uuid', 'target_node_uuid',
    'name', 'fact', 'episodes', 'created_at', 'expired_at',
    'valid_at', 'invalid_at', 'attributes',
]
_EPISODE_OUTPUT_FIELDS = [
    'uuid', 'group_id', 'name', 'content', 'source',
    'source_description', 'created_at', 'valid_at', 'entity_edges',
]
_COMMUNITY_OUTPUT_FIELDS = [
    'uuid', 'group_id', 'name', 'summary', 'created_at',
]


def _combine_filters(*parts: str) -> str:
    """Combine non-empty filter expression strings with AND."""
    non_empty = [p for p in parts if p]
    if not non_empty:
        return ''
    return ' and '.join(non_empty)


class MilvusSearchInterface(SearchInterface):
    """SearchInterface implementation backed by a VectorStoreClient.

    Delegates similarity search (HNSW/COSINE) and fulltext search (BM25) to the
    shared VectorStoreClient. BFS, distance reranker, and episode mentions reranker
    raise NotImplementedError to fall back to the graph database.
    """

    _vs_client: Any = PrivateAttr()

    def __init__(self, *, vs_client: Any, **data: Any) -> None:
        super().__init__(**data)
        object.__setattr__(self, '_vs_client', vs_client)

    # ---- Similarity Search (COSINE / HNSW) ----

    async def node_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[Any]:
        if not search_vector:
            return []
        filter_expr = _combine_filters(
            build_group_ids_filter(group_ids),
            build_milvus_filter_from_search_filters(search_filter, 'node'),
        )
        results = await self._vs_client.search(
            collection_name=self._vs_client.collection_name(COLLECTION_ENTITY_NODES),
            data=[search_vector],
            anns_field='name_embedding',
            search_params={'metric_type': 'COSINE', 'params': {'ef': 64}},
            filter_expr=filter_expr or '',
            output_fields=_NODE_OUTPUT_FIELDS,
            limit=limit,
        )
        nodes = []
        for hits in results:
            for hit in hits:
                if hit['distance'] >= min_score:
                    nodes.append(milvus_dict_to_entity_node(hit['entity']))
        return nodes

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
        if not search_vector:
            return []
        extra_filters: list[str] = []
        if source_node_uuid:
            extra_filters.append(f'source_node_uuid == "{source_node_uuid}"')
        if target_node_uuid:
            extra_filters.append(f'target_node_uuid == "{target_node_uuid}"')
        filter_expr = _combine_filters(
            build_group_ids_filter(group_ids),
            build_milvus_filter_from_search_filters(search_filter, 'edge'),
            *extra_filters,
        )
        results = await self._vs_client.search(
            collection_name=self._vs_client.collection_name(COLLECTION_ENTITY_EDGES),
            data=[search_vector],
            anns_field='fact_embedding',
            search_params={'metric_type': 'COSINE', 'params': {'ef': 64}},
            filter_expr=filter_expr or '',
            output_fields=_EDGE_OUTPUT_FIELDS,
            limit=limit,
        )
        edges = []
        for hits in results:
            for hit in hits:
                if hit['distance'] >= min_score:
                    edges.append(milvus_dict_to_entity_edge(hit['entity']))
        return edges

    async def community_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.6,
    ) -> list[Any]:
        if not search_vector:
            return []
        filter_expr = build_group_ids_filter(group_ids)
        results = await self._vs_client.search(
            collection_name=self._vs_client.collection_name(COLLECTION_COMMUNITY_NODES),
            data=[search_vector],
            anns_field='name_embedding',
            search_params={'metric_type': 'COSINE', 'params': {'ef': 64}},
            filter_expr=filter_expr or '',
            output_fields=_COMMUNITY_OUTPUT_FIELDS,
            limit=limit,
        )
        nodes = []
        for hits in results:
            for hit in hits:
                if hit['distance'] >= min_score:
                    nodes.append(milvus_dict_to_community_node(hit['entity']))
        return nodes

    # ---- Fulltext Search (BM25) ----

    async def edge_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        if not query:
            return []
        filter_expr = _combine_filters(
            build_group_ids_filter(group_ids),
            build_milvus_filter_from_search_filters(search_filter, 'edge'),
        )
        results = await self._vs_client.search(
            collection_name=self._vs_client.collection_name(COLLECTION_ENTITY_EDGES),
            data=[query],
            anns_field='fact_sparse',
            search_params={'metric_type': 'BM25'},
            filter_expr=filter_expr or '',
            output_fields=_EDGE_OUTPUT_FIELDS,
            limit=limit,
        )
        return [
            milvus_dict_to_entity_edge(hit['entity'])
            for hits in results
            for hit in hits
        ]

    async def node_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        if not query:
            return []
        filter_expr = _combine_filters(
            build_group_ids_filter(group_ids),
            build_milvus_filter_from_search_filters(search_filter, 'node'),
        )
        results = await self._vs_client.search(
            collection_name=self._vs_client.collection_name(COLLECTION_ENTITY_NODES),
            data=[query],
            anns_field='name_sparse',
            search_params={'metric_type': 'BM25'},
            filter_expr=filter_expr or '',
            output_fields=_NODE_OUTPUT_FIELDS,
            limit=limit,
        )
        return [
            milvus_dict_to_entity_node(hit['entity'])
            for hits in results
            for hit in hits
        ]

    async def episode_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        if not query:
            return []
        filter_expr = build_group_ids_filter(group_ids)
        results = await self._vs_client.search(
            collection_name=self._vs_client.collection_name(COLLECTION_EPISODIC_NODES),
            data=[query],
            anns_field='content_sparse',
            search_params={'metric_type': 'BM25'},
            filter_expr=filter_expr or '',
            output_fields=_EPISODE_OUTPUT_FIELDS,
            limit=limit,
        )
        return [
            milvus_dict_to_episodic_node(hit['entity'])
            for hits in results
            for hit in hits
        ]

    async def community_fulltext_search(
        self,
        driver: Any,
        query: str,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        if not query:
            return []
        filter_expr = build_group_ids_filter(group_ids)
        results = await self._vs_client.search(
            collection_name=self._vs_client.collection_name(COLLECTION_COMMUNITY_NODES),
            data=[query],
            anns_field='name_sparse',
            search_params={'metric_type': 'BM25'},
            filter_expr=filter_expr or '',
            output_fields=_COMMUNITY_OUTPUT_FIELDS,
            limit=limit,
        )
        return [
            milvus_dict_to_community_node(hit['entity'])
            for hits in results
            for hit in hits
        ]

    # ---- Embeddings ----

    async def get_embeddings_for_communities(
        self,
        driver: Any,
        communities: list[Any],
    ) -> dict[str, list[float]]:
        if not communities:
            return {}
        uuids = [c.uuid for c in communities]
        uuids_str = ', '.join(f'"{u}"' for u in uuids)
        results = await self._vs_client.query(
            collection_name=self._vs_client.collection_name(COLLECTION_COMMUNITY_NODES),
            filter_expr=f'uuid in [{uuids_str}]',
            output_fields=['uuid', 'name_embedding'],
        )
        return {r['uuid']: r['name_embedding'] for r in results if r.get('name_embedding')}

    # ---- Search Filters ----

    def build_node_search_filters(self, search_filters: Any) -> Any:
        return build_milvus_filter_from_search_filters(search_filters, 'node')

    def build_edge_search_filters(self, search_filters: Any) -> Any:
        return build_milvus_filter_from_search_filters(search_filters, 'edge')

    # ---- Not Implemented (fall back to graph DB) ----

    async def edge_bfs_search(
        self,
        driver: Any,
        bfs_origin_node_uuids: list[str] | None,
        bfs_max_depth: int,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        raise NotImplementedError

    async def node_bfs_search(
        self,
        driver: Any,
        bfs_origin_node_uuids: list[str] | None,
        search_filter: Any,
        bfs_max_depth: int,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        raise NotImplementedError

    async def node_distance_reranker(
        self,
        driver: Any,
        node_uuids: list[str],
        center_node_uuid: str,
        min_score: float = 0,
    ) -> tuple[list[str], list[float]]:
        raise NotImplementedError

    async def episode_mentions_reranker(
        self,
        driver: Any,
        node_uuids: list[list[str]],
        min_score: float = 0,
    ) -> tuple[list[str], list[float]]:
        raise NotImplementedError
