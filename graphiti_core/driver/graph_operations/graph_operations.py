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

    All methods use `Any` type hints to avoid circular imports. See docstrings
    for expected concrete types.

    Type reference:
        - driver: GraphDriver
        - EntityNode, EpisodicNode, CommunityNode, SagaNode from graphiti_core.nodes
        - EntityEdge, EpisodicEdge, CommunityEdge from graphiti_core.edges
        - EpisodeType from graphiti_core.nodes
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

    async def retrieve_episodes(
        self,
        driver: Any,
        reference_time: Any,
        last_n: int = 3,
        group_ids: list[str] | None = None,
        source: Any | None = None,
        saga: str | None = None,
    ) -> list[Any]:
        """
        Retrieve the last n episodic nodes from the graph.

        Args:
            driver: GraphDriver instance
            reference_time: datetime object. Only episodes with valid_at <= reference_time
                are returned, allowing point-in-time queries.
            last_n: Number of most recent episodes to retrieve (default: 3)
            group_ids: Optional list of group IDs to filter by
            source: Optional EpisodeType to filter by source type
            saga: Optional saga name. If provided, only retrieves episodes
                belonging to that saga.

        Returns:
            list[EpisodicNode]: List of EpisodicNode objects in chronological order
                (oldest first)
        """
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

    # -----------------
    # Search
    # -----------------

    async def get_mentioned_nodes(
        self,
        driver: Any,
        episodes: list[Any],
    ) -> list[Any]:
        """
        Retrieve entity nodes mentioned by the given episodic nodes.

        Args:
            driver: GraphDriver instance
            episodes: List of EpisodicNode objects

        Returns:
            list[EntityNode]: List of EntityNode objects that are mentioned
                by the given episodes via MENTIONS relationships
        """
        raise NotImplementedError

    async def get_communities_by_nodes(
        self,
        driver: Any,
        nodes: list[Any],
    ) -> list[Any]:
        """
        Retrieve community nodes that contain the given entity nodes as members.

        Args:
            driver: GraphDriver instance
            nodes: List of EntityNode objects

        Returns:
            list[CommunityNode]: List of CommunityNode objects that have
                HAS_MEMBER relationships to the given entity nodes
        """
        raise NotImplementedError

    # -----------------
    # Maintenance
    # -----------------

    async def clear_data(
        self,
        driver: Any,
        group_ids: list[str] | None = None,
    ) -> None:
        """
        Clear all data or group-specific data from the graph.

        Args:
            driver: GraphDriver instance
            group_ids: If provided, only delete data in these groups.
                If None, deletes ALL data in the graph.
        """
        raise NotImplementedError

    async def get_community_clusters(
        self,
        driver: Any,
        group_ids: list[str] | None,
    ) -> list[list[Any]]:
        """
        Retrieve all entity node clusters for community detection.

        Uses label propagation algorithm internally to identify clusters
        of related entities based on their edge connections.

        Args:
            driver: GraphDriver instance
            group_ids: List of group IDs to process. If None, processes
                all groups found in the graph.

        Returns:
            list[list[EntityNode]]: List of clusters, where each cluster
                is a list of EntityNode objects that belong together
        """
        raise NotImplementedError

    async def remove_communities(
        self,
        driver: Any,
    ) -> None:
        """
        Delete all community nodes from the graph.

        This removes all Community-labeled nodes and their relationships.

        Args:
            driver: GraphDriver instance
        """
        raise NotImplementedError

    async def determine_entity_community(
        self,
        driver: Any,
        entity: Any,
    ) -> tuple[Any | None, bool]:
        """
        Determine which community an entity belongs to.

        First checks if the entity is already a member of a community.
        If not, finds the most common community among neighboring entities.

        Args:
            driver: GraphDriver instance
            entity: EntityNode object to find community for

        Returns:
            tuple[CommunityNode | None, bool]: Tuple of (community, is_new) where:
                - community: The CommunityNode the entity belongs to, or None
                - is_new: True if this is a new membership (entity wasn't already
                  in this community), False if entity was already a member
        """
        raise NotImplementedError

    # -----------------
    # Additional Node Operations
    # -----------------

    async def episodic_node_get_by_entity_node_uuid(
        self,
        _cls: Any,
        driver: Any,
        entity_node_uuid: str,
    ) -> list[Any]:
        """
        Retrieve all episodes mentioning a specific entity.

        Args:
            _cls: The EpisodicNode class (for interface consistency)
            driver: GraphDriver instance
            entity_node_uuid: UUID of the EntityNode to find episodes for

        Returns:
            list[EpisodicNode]: List of EpisodicNode objects that have
                MENTIONS relationships to the specified entity
        """
        raise NotImplementedError

    async def community_node_load_name_embedding(
        self,
        node: Any,
        driver: Any,
    ) -> None:
        """
        Load the name embedding for a community node.

        Populates the node.name_embedding field in-place.

        Args:
            node: CommunityNode object to load embedding for
            driver: GraphDriver instance
        """
        raise NotImplementedError

    # -----------------
    # Additional Edge Operations
    # -----------------

    async def edge_get_between_nodes(
        self,
        _cls: Any,
        driver: Any,
        source_node_uuid: str,
        target_node_uuid: str,
    ) -> list[Any]:
        """
        Get edges connecting two specific entity nodes.

        Args:
            _cls: The EntityEdge class (for interface consistency)
            driver: GraphDriver instance
            source_node_uuid: UUID of the source EntityNode
            target_node_uuid: UUID of the target EntityNode

        Returns:
            list[EntityEdge]: List of EntityEdge objects connecting the two nodes.
                Note: Only returns edges in the source->target direction.
        """
        raise NotImplementedError

    async def edge_get_by_node_uuid(
        self,
        _cls: Any,
        driver: Any,
        node_uuid: str,
    ) -> list[Any]:
        """
        Get all edges connected to a specific node.

        Args:
            _cls: The EntityEdge class (for interface consistency)
            driver: GraphDriver instance
            node_uuid: UUID of the EntityNode to find edges for

        Returns:
            list[EntityEdge]: List of EntityEdge objects where the node
                is either the source or target
        """
        raise NotImplementedError
