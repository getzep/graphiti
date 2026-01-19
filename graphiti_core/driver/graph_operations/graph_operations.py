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

from typing import Any

from pydantic import BaseModel


class GraphOperationsInterface(BaseModel):
    """
    Interface for updating graph mutation behavior.
    """

    # -----------------
    # Node: Save/Delete
    # -----------------

    async def node_save(self, node: Any, driver: Any) -> None:
        """Persist (create or update) a single node."""
        raise NotImplementedError

    async def node_delete(self, node: Any, driver: Any) -> None:
        raise NotImplementedError

    async def node_save_bulk(
        self,
        _cls: Any,  # kept for parity; callers won't pass it
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        """Persist (create or update) many nodes in batches."""
        raise NotImplementedError

    async def node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        raise NotImplementedError

    async def node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        raise NotImplementedError

    # -----------------
    # Node: Read
    # -----------------

    async def node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        """Retrieve a single node by UUID."""
        raise NotImplementedError

    async def node_get_by_uuids(self, _cls: Any, driver: Any, uuids: list[str]) -> list[Any]:
        """Retrieve multiple nodes by UUIDs."""
        raise NotImplementedError

    async def node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        """Retrieve nodes by group IDs with optional pagination."""
        raise NotImplementedError

    # --------------------------
    # Node: Embeddings (load)
    # --------------------------

    async def node_load_embeddings(self, node: Any, driver: Any) -> None:
        """
        Load embedding vectors for a single node into the instance (e.g., set node.embedding or similar).
        """
        raise NotImplementedError

    async def node_load_embeddings_bulk(
        self,
        driver: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> dict[str, list[float]]:
        """
        Load embedding vectors for many nodes in batches.
        """
        raise NotImplementedError

    # --------------------------
    # EpisodicNode: Save/Delete
    # --------------------------

    async def episodic_node_save(self, node: Any, driver: Any) -> None:
        """Persist (create or update) a single episodic node."""
        raise NotImplementedError

    async def episodic_node_delete(self, node: Any, driver: Any) -> None:
        raise NotImplementedError

    async def episodic_node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        """Persist (create or update) many episodic nodes in batches."""
        raise NotImplementedError

    async def episodic_edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        episodic_edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        """Persist (create or update) many episodic edges in batches."""
        raise NotImplementedError

    async def episodic_node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        raise NotImplementedError

    async def episodic_node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        raise NotImplementedError

    # -----------------------
    # EpisodicNode: Read
    # -----------------------

    async def episodic_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        """Retrieve a single episodic node by UUID."""
        raise NotImplementedError

    async def episodic_node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        """Retrieve multiple episodic nodes by UUIDs."""
        raise NotImplementedError

    async def episodic_node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        """Retrieve episodic nodes by group IDs with optional pagination."""
        raise NotImplementedError

    # -----------------------
    # CommunityNode: Save/Delete
    # -----------------------

    async def community_node_save(self, node: Any, driver: Any) -> None:
        """Persist (create or update) a single community node."""
        raise NotImplementedError

    async def community_node_delete(self, node: Any, driver: Any) -> None:
        raise NotImplementedError

    async def community_node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        """Persist (create or update) many community nodes in batches."""
        raise NotImplementedError

    async def community_node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        raise NotImplementedError

    async def community_node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        raise NotImplementedError

    # -----------------------
    # CommunityNode: Read
    # -----------------------

    async def community_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        """Retrieve a single community node by UUID."""
        raise NotImplementedError

    async def community_node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        """Retrieve multiple community nodes by UUIDs."""
        raise NotImplementedError

    async def community_node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        """Retrieve community nodes by group IDs with optional pagination."""
        raise NotImplementedError

    # -----------------------
    # SagaNode: Save/Delete
    # -----------------------

    async def saga_node_save(self, node: Any, driver: Any) -> None:
        """Persist (create or update) a single saga node."""
        raise NotImplementedError

    async def saga_node_delete(self, node: Any, driver: Any) -> None:
        raise NotImplementedError

    async def saga_node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        """Persist (create or update) many saga nodes in batches."""
        raise NotImplementedError

    async def saga_node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        raise NotImplementedError

    async def saga_node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        raise NotImplementedError

    # -----------------------
    # SagaNode: Read
    # -----------------------

    async def saga_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        """Retrieve a single saga node by UUID."""
        raise NotImplementedError

    async def saga_node_get_by_uuids(self, _cls: Any, driver: Any, uuids: list[str]) -> list[Any]:
        """Retrieve multiple saga nodes by UUIDs."""
        raise NotImplementedError

    async def saga_node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        """Retrieve saga nodes by group IDs with optional pagination."""
        raise NotImplementedError

    # -----------------
    # Edge: Save/Delete
    # -----------------

    async def edge_save(self, edge: Any, driver: Any) -> None:
        """Persist (create or update) a single edge."""
        raise NotImplementedError

    async def edge_delete(self, edge: Any, driver: Any) -> None:
        raise NotImplementedError

    async def edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        """Persist (create or update) many edges in batches."""
        raise NotImplementedError

    async def edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        raise NotImplementedError

    # -----------------
    # Edge: Read
    # -----------------

    async def edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        """Retrieve a single edge by UUID."""
        raise NotImplementedError

    async def edge_get_by_uuids(self, _cls: Any, driver: Any, uuids: list[str]) -> list[Any]:
        """Retrieve multiple edges by UUIDs."""
        raise NotImplementedError

    async def edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        """Retrieve edges by group IDs with optional pagination."""
        raise NotImplementedError

    # -----------------
    # Edge: Embeddings (load)
    # -----------------

    async def edge_load_embeddings(self, edge: Any, driver: Any) -> None:
        """
        Load embedding vectors for a single edge into the instance (e.g., set edge.embedding or similar).
        """
        raise NotImplementedError

    async def edge_load_embeddings_bulk(
        self,
        driver: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> dict[str, list[float]]:
        """
        Load embedding vectors for many edges in batches
        """
        raise NotImplementedError

    # ---------------------------
    # EpisodicEdge: Save/Delete
    # ---------------------------

    async def episodic_edge_save(self, edge: Any, driver: Any) -> None:
        """Persist (create or update) a single episodic edge (MENTIONS)."""
        raise NotImplementedError

    async def episodic_edge_delete(self, edge: Any, driver: Any) -> None:
        raise NotImplementedError

    async def episodic_edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        raise NotImplementedError

    # ---------------------------
    # EpisodicEdge: Read
    # ---------------------------

    async def episodic_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        """Retrieve a single episodic edge by UUID."""
        raise NotImplementedError

    async def episodic_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        """Retrieve multiple episodic edges by UUIDs."""
        raise NotImplementedError

    async def episodic_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        """Retrieve episodic edges by group IDs with optional pagination."""
        raise NotImplementedError

    # ---------------------------
    # CommunityEdge: Save/Delete
    # ---------------------------

    async def community_edge_save(self, edge: Any, driver: Any) -> None:
        """Persist (create or update) a single community edge (HAS_MEMBER)."""
        raise NotImplementedError

    async def community_edge_delete(self, edge: Any, driver: Any) -> None:
        raise NotImplementedError

    async def community_edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        raise NotImplementedError

    # ---------------------------
    # CommunityEdge: Read
    # ---------------------------

    async def community_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        """Retrieve a single community edge by UUID."""
        raise NotImplementedError

    async def community_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        """Retrieve multiple community edges by UUIDs."""
        raise NotImplementedError

    async def community_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        """Retrieve community edges by group IDs with optional pagination."""
        raise NotImplementedError

    # ---------------------------
    # HasEpisodeEdge: Save/Delete
    # ---------------------------

    async def has_episode_edge_save(self, edge: Any, driver: Any) -> None:
        """Persist (create or update) a single has_episode edge."""
        raise NotImplementedError

    async def has_episode_edge_delete(self, edge: Any, driver: Any) -> None:
        raise NotImplementedError

    async def has_episode_edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        """Persist (create or update) many has_episode edges in batches."""
        raise NotImplementedError

    async def has_episode_edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        raise NotImplementedError

    # ---------------------------
    # HasEpisodeEdge: Read
    # ---------------------------

    async def has_episode_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        """Retrieve a single has_episode edge by UUID."""
        raise NotImplementedError

    async def has_episode_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        """Retrieve multiple has_episode edges by UUIDs."""
        raise NotImplementedError

    async def has_episode_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        """Retrieve has_episode edges by group IDs with optional pagination."""
        raise NotImplementedError

    # ----------------------------
    # NextEpisodeEdge: Save/Delete
    # ----------------------------

    async def next_episode_edge_save(self, edge: Any, driver: Any) -> None:
        """Persist (create or update) a single next_episode edge."""
        raise NotImplementedError

    async def next_episode_edge_delete(self, edge: Any, driver: Any) -> None:
        raise NotImplementedError

    async def next_episode_edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        """Persist (create or update) many next_episode edges in batches."""
        raise NotImplementedError

    async def next_episode_edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        raise NotImplementedError

    # ----------------------------
    # NextEpisodeEdge: Read
    # ----------------------------

    async def next_episode_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        """Retrieve a single next_episode edge by UUID."""
        raise NotImplementedError

    async def next_episode_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        """Retrieve multiple next_episode edges by UUIDs."""
        raise NotImplementedError

    async def next_episode_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        """Retrieve next_episode edges by group IDs with optional pagination."""
        raise NotImplementedError
