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

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class VectorStoreConfig(BaseModel):
    """Base configuration for vector store clients."""

    embedding_dim: int = 1024
    collection_prefix: str = 'graphiti'


class VectorStoreClient(ABC):
    """Abstract base class for vector store backends.

    Owns connection lifecycle and collection management. Concrete implementations
    (e.g. MilvusVectorStoreClient) wrap a specific vector DB SDK.
    """

    @abstractmethod
    async def ensure_ready(self) -> None:
        """Lazily initialize the connection and create collections. Idempotent."""
        ...

    @abstractmethod
    def collection_name(self, suffix: str) -> str:
        """Return the full collection name for a given suffix."""
        ...

    @abstractmethod
    async def upsert(self, collection_name: str, data: list[dict[str, Any]]) -> None:
        """Insert or update records."""
        ...

    @abstractmethod
    async def delete(self, collection_name: str, filter_expr: str) -> None:
        """Delete records matching the filter expression."""
        ...

    @abstractmethod
    async def query(
        self,
        collection_name: str,
        filter_expr: str,
        output_fields: list[str],
    ) -> list[dict[str, Any]]:
        """Query records by filter expression."""
        ...

    @abstractmethod
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
        """Vector or BM25 search. Returns list of hit-lists."""
        ...

    @abstractmethod
    async def has_collection(self, collection_name: str) -> bool:
        """Check whether a collection exists."""
        ...

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        schema: Any,
        index_params: Any,
    ) -> None:
        """Create a collection with the given schema and index parameters."""
        ...

    @abstractmethod
    async def drop_collection(self, collection_name: str) -> None:
        """Drop a collection."""
        ...

    @abstractmethod
    async def reset_collections(self) -> None:
        """Drop all managed collections and recreate them."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the underlying connection."""
        ...

    # ------------------------------------------------------------------
    # Domain-aware save/delete methods
    #
    # These methods accept Graphiti domain objects and handle serialization
    # internally. All type hints use ``Any`` to avoid circular imports.
    #
    # Type reference:
    #     - EntityNode, EpisodicNode, CommunityNode from graphiti_core.nodes
    #     - EntityEdge from graphiti_core.edges
    # ------------------------------------------------------------------

    async def save_entity_nodes(self, nodes: list[Any]) -> None:
        """Save entity nodes to the vector store.

        Args:
            nodes: list[EntityNode] — EntityNode objects from graphiti_core.nodes
        """
        raise NotImplementedError

    async def save_entity_edges(self, edges: list[Any]) -> None:
        """Save entity edges to the vector store.

        Args:
            edges: list[EntityEdge] — EntityEdge objects from graphiti_core.edges
        """
        raise NotImplementedError

    async def save_episodic_nodes(self, nodes: list[Any]) -> None:
        """Save episodic nodes to the vector store.

        Args:
            nodes: list[EpisodicNode] — EpisodicNode objects from graphiti_core.nodes
        """
        raise NotImplementedError

    async def save_community_nodes(self, nodes: list[Any]) -> None:
        """Save community nodes to the vector store.

        Args:
            nodes: list[CommunityNode] — CommunityNode objects from graphiti_core.nodes
        """
        raise NotImplementedError

    async def delete_entity_nodes(self, uuids: list[str]) -> None:
        """Delete entity nodes from the vector store by UUID.

        Args:
            uuids: List of entity node UUIDs to delete
        """
        raise NotImplementedError

    async def delete_entity_edges(self, uuids: list[str]) -> None:
        """Delete entity edges from the vector store by UUID.

        Args:
            uuids: List of entity edge UUIDs to delete
        """
        raise NotImplementedError

    async def delete_nodes_by_uuids(self, uuids: list[str]) -> None:
        """Delete nodes from entity, episodic, and community collections by UUID.

        Used by Node.delete() and Node.delete_by_uuids() which operate across
        all node types.

        Args:
            uuids: List of node UUIDs to delete
        """
        raise NotImplementedError

    async def delete_community_nodes(self, uuids: list[str] | None = None) -> None:
        """Delete community nodes from the vector store.

        Args:
            uuids: List of community node UUIDs to delete. If None, deletes all.
        """
        raise NotImplementedError

    async def delete_by_group_ids(self, group_ids: list[str]) -> None:
        """Delete all records for the given group IDs across all collections.

        Args:
            group_ids: List of group IDs whose records should be deleted
        """
        raise NotImplementedError

    async def clear_all(self) -> None:
        """Clear all data from all managed collections.

        Default implementation delegates to reset_collections().
        """
        await self.reset_collections()
