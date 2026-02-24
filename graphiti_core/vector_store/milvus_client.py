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

from graphiti_core.vector_store.client import VectorStoreClient, VectorStoreConfig
from graphiti_core.vector_store.milvus_utils import (
    COLLECTION_COMMUNITY_NODES,
    COLLECTION_ENTITY_EDGES,
    COLLECTION_ENTITY_NODES,
    COLLECTION_EPISODIC_NODES,
    community_node_to_milvus_dict,
    entity_edge_to_milvus_dict,
    entity_node_to_milvus_dict,
    episodic_node_to_milvus_dict,
    get_community_node_collection_schema,
    get_entity_edge_collection_schema,
    get_entity_node_collection_schema,
    get_episodic_node_collection_schema,
)

logger = logging.getLogger(__name__)


class MilvusVectorStoreConfig(VectorStoreConfig):
    """Configuration for MilvusVectorStoreClient."""

    uri: str = 'http://localhost:19530'
    token: str | None = None
    db_name: str = 'default'


class MilvusVectorStoreClient(VectorStoreClient):
    """VectorStoreClient backed by Milvus / Zilliz Cloud.

    Lazily creates an ``AsyncMilvusClient`` on first use and ensures all four
    managed collections exist.
    """

    def __init__(self, config: MilvusVectorStoreConfig) -> None:
        self._config = config
        self._client: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def ensure_ready(self) -> None:
        if self._client is not None:
            return
        from pymilvus import AsyncMilvusClient

        self._client = AsyncMilvusClient(
            uri=self._config.uri,
            token=self._config.token or '',
            db_name=self._config.db_name,
        )
        await self._ensure_collections()

    async def _ensure_collections(self) -> None:
        """Create all 4 collections if they don't already exist."""
        collections = {
            COLLECTION_ENTITY_NODES: get_entity_node_collection_schema(self._config.embedding_dim),
            COLLECTION_ENTITY_EDGES: get_entity_edge_collection_schema(self._config.embedding_dim),
            COLLECTION_EPISODIC_NODES: get_episodic_node_collection_schema(),
            COLLECTION_COMMUNITY_NODES: get_community_node_collection_schema(
                self._config.embedding_dim
            ),
        }
        for suffix, (schema, index_params) in collections.items():
            col_name = self.collection_name(suffix)
            has = await self._client.has_collection(col_name)
            if not has:
                await self._client.create_collection(
                    collection_name=col_name,
                    schema=schema,
                    index_params=index_params,
                )
                logger.info(f'Created Milvus collection: {col_name}')

    def collection_name(self, suffix: str) -> str:
        return f'{self._config.collection_prefix}_{suffix}'

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def upsert(self, collection_name: str, data: list[dict[str, Any]]) -> None:
        await self.ensure_ready()
        await self._client.upsert(collection_name=collection_name, data=data)

    async def delete(self, collection_name: str, filter_expr: str) -> None:
        await self.ensure_ready()
        await self._client.delete(collection_name=collection_name, filter=filter_expr)

    async def query(
        self,
        collection_name: str,
        filter_expr: str,
        output_fields: list[str],
    ) -> list[dict[str, Any]]:
        await self.ensure_ready()
        return await self._client.query(
            collection_name=collection_name,
            filter=filter_expr,
            output_fields=output_fields,
        )

    async def search(
        self,
        collection_name: str,
        data: list[Any],
        anns_field: str,
        search_params: dict[str, Any],
        filter_expr: str,
        output_fields: list[str],
        limit: int,
    ) -> list[list[dict[str, Any]]]:
        await self.ensure_ready()
        return await self._client.search(
            collection_name=collection_name,
            data=data,
            anns_field=anns_field,
            search_params=search_params,
            filter=filter_expr,
            output_fields=output_fields,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    async def has_collection(self, collection_name: str) -> bool:
        await self.ensure_ready()
        return await self._client.has_collection(collection_name)

    async def create_collection(
        self,
        collection_name: str,
        schema: Any,
        index_params: Any,
    ) -> None:
        await self.ensure_ready()
        await self._client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    async def drop_collection(self, collection_name: str) -> None:
        await self.ensure_ready()
        await self._client.drop_collection(collection_name)

    async def reset_collections(self) -> None:
        """Drop all 4 managed collections and recreate them."""
        await self.ensure_ready()
        all_suffixes = [
            COLLECTION_ENTITY_NODES,
            COLLECTION_ENTITY_EDGES,
            COLLECTION_EPISODIC_NODES,
            COLLECTION_COMMUNITY_NODES,
        ]
        for suffix in all_suffixes:
            col_name = self.collection_name(suffix)
            has = await self._client.has_collection(col_name)
            if has:
                await self._client.drop_collection(col_name)
        await self._ensure_collections()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Domain-aware save/delete
    # ------------------------------------------------------------------

    async def save_entity_nodes(self, nodes: list[Any]) -> None:
        if not nodes:
            return
        await self.ensure_ready()
        await self._client.upsert(
            collection_name=self.collection_name(COLLECTION_ENTITY_NODES),
            data=[entity_node_to_milvus_dict(n) for n in nodes],
        )

    async def save_entity_edges(self, edges: list[Any]) -> None:
        if not edges:
            return
        await self.ensure_ready()
        await self._client.upsert(
            collection_name=self.collection_name(COLLECTION_ENTITY_EDGES),
            data=[entity_edge_to_milvus_dict(e) for e in edges],
        )

    async def save_episodic_nodes(self, nodes: list[Any]) -> None:
        if not nodes:
            return
        await self.ensure_ready()
        await self._client.upsert(
            collection_name=self.collection_name(COLLECTION_EPISODIC_NODES),
            data=[episodic_node_to_milvus_dict(n) for n in nodes],
        )

    async def save_community_nodes(self, nodes: list[Any]) -> None:
        if not nodes:
            return
        await self.ensure_ready()
        await self._client.upsert(
            collection_name=self.collection_name(COLLECTION_COMMUNITY_NODES),
            data=[community_node_to_milvus_dict(n) for n in nodes],
        )

    async def delete_entity_nodes(self, uuids: list[str]) -> None:
        if not uuids:
            return
        await self.ensure_ready()
        uuid_list = ', '.join(f'"{u}"' for u in uuids)
        await self._client.delete(
            collection_name=self.collection_name(COLLECTION_ENTITY_NODES),
            filter=f'uuid in [{uuid_list}]',
        )

    async def delete_entity_edges(self, uuids: list[str]) -> None:
        if not uuids:
            return
        await self.ensure_ready()
        uuid_list = ', '.join(f'"{u}"' for u in uuids)
        await self._client.delete(
            collection_name=self.collection_name(COLLECTION_ENTITY_EDGES),
            filter=f'uuid in [{uuid_list}]',
        )

    async def delete_nodes_by_uuids(self, uuids: list[str]) -> None:
        if not uuids:
            return
        await self.ensure_ready()
        uuid_list = ', '.join(f'"{u}"' for u in uuids)
        filter_expr = f'uuid in [{uuid_list}]'
        for suffix in [
            COLLECTION_ENTITY_NODES,
            COLLECTION_EPISODIC_NODES,
            COLLECTION_COMMUNITY_NODES,
        ]:
            await self._client.delete(
                collection_name=self.collection_name(suffix),
                filter=filter_expr,
            )

    async def delete_community_nodes(self, uuids: list[str] | None = None) -> None:
        await self.ensure_ready()
        if uuids is None:
            await self._client.delete(
                collection_name=self.collection_name(COLLECTION_COMMUNITY_NODES),
                filter='uuid != ""',
            )
        elif uuids:
            uuid_list = ', '.join(f'"{u}"' for u in uuids)
            await self._client.delete(
                collection_name=self.collection_name(COLLECTION_COMMUNITY_NODES),
                filter=f'uuid in [{uuid_list}]',
            )

    async def delete_by_group_ids(self, group_ids: list[str]) -> None:
        if not group_ids:
            return
        await self.ensure_ready()
        gid_list = ', '.join(f'"{g}"' for g in group_ids)
        filter_expr = f'group_id in [{gid_list}]'
        for suffix in [
            COLLECTION_ENTITY_NODES,
            COLLECTION_EPISODIC_NODES,
            COLLECTION_COMMUNITY_NODES,
            COLLECTION_ENTITY_EDGES,
        ]:
            await self._client.delete(
                collection_name=self.collection_name(suffix),
                filter=filter_expr,
            )
