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

from datetime import datetime
from typing import Any

from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface


class FalkorGraphOperationsInterface(GraphOperationsInterface):
    """
    FalkorDB implementation of GraphOperationsInterface.

    Delegates all operations to the FalkorDB-specific operations classes,
    which handle multi-tenant routing by cloning the driver to the appropriate
    database for each group_id.

    This facade makes the per-group routing logic in FalkorDB operations reachable
    via the classmethod dispatch path (e.g., EpisodicNode.get_by_group_ids).
    """

    def __init__(
        self,
        entity_node_ops: Any,
        episode_node_ops: Any,
        community_node_ops: Any,
        saga_node_ops: Any,
        entity_edge_ops: Any,
        episodic_edge_ops: Any,
        community_edge_ops: Any,
        has_episode_edge_ops: Any,
        next_episode_edge_ops: Any,
        search_ops: Any,
        graph_ops: Any,
    ):
        super().__init__()
        self._entity_node_ops = entity_node_ops
        self._episode_node_ops = episode_node_ops
        self._community_node_ops = community_node_ops
        self._saga_node_ops = saga_node_ops
        self._entity_edge_ops = entity_edge_ops
        self._episodic_edge_ops = episodic_edge_ops
        self._community_edge_ops = community_edge_ops
        self._has_episode_edge_ops = has_episode_edge_ops
        self._next_episode_edge_ops = next_episode_edge_ops
        self._search_ops = search_ops
        self._graph_ops = graph_ops

    # -----------------
    # Node: Save/Delete
    # -----------------

    async def node_save(self, node: Any, driver: Any) -> None:
        await self._entity_node_ops.save(driver, node)

    async def node_delete(self, node: Any, driver: Any) -> None:
        await self._entity_node_ops.delete(driver, node)

    async def node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        await self._entity_node_ops.save_bulk(driver, nodes, tx=transaction, batch_size=batch_size)

    async def node_update_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        await self._entity_node_ops.update_bulk(
            driver, nodes, tx=transaction, batch_size=batch_size
        )

    async def node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        await self._entity_node_ops.delete_by_group_id(driver, group_id, batch_size=batch_size)

    async def node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        await self._entity_node_ops.delete_by_uuids(driver, uuids, batch_size=batch_size)

    # -----------------
    # Node: Read
    # -----------------

    async def node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await self._entity_node_ops.get_by_uuid(driver, uuid)

    async def node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str], group_id: str | None = None
    ) -> list[Any]:
        return await self._entity_node_ops.get_by_uuids(driver, uuids)

    async def node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await self._entity_node_ops.get_by_group_ids(driver, group_ids, limit, uuid_cursor)

    async def node_get_by_names(
        self,
        _cls: Any,
        driver: Any,
        names: list[str],
        group_ids: list[str],
    ) -> list[Any]:
        return await self._entity_node_ops.get_by_names(driver, names, group_ids)

    async def node_get_by_attribute_values(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        attributes: dict[str, Any],
        node_labels: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        return await self._entity_node_ops.get_by_attribute_values(
            driver, group_id, attributes, node_labels, limit
        )

    # --------------------------
    # Node: Embeddings (load)
    # --------------------------

    async def node_load_embeddings(self, node: Any, driver: Any) -> None:
        await self._entity_node_ops.load_embeddings(driver, node)

    async def node_load_embeddings_bulk(
        self,
        driver: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> dict[str, list[float]]:
        return await self._entity_node_ops.load_embeddings_bulk(driver, nodes, batch_size)

    # --------------------------
    # EpisodicNode: Save/Delete
    # --------------------------

    async def episodic_node_save(self, node: Any, driver: Any) -> None:
        await self._episode_node_ops.save(driver, node)

    async def episodic_node_delete(self, node: Any, driver: Any) -> None:
        await self._episode_node_ops.delete(driver, node)

    async def episodic_node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        await self._episode_node_ops.save_bulk(driver, nodes, tx=transaction, batch_size=batch_size)

    async def episodic_edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        episodic_edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        await self._episodic_edge_ops.save_bulk(
            driver, episodic_edges, tx=transaction, batch_size=batch_size
        )

    async def episodic_node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        await self._episode_node_ops.delete_by_group_id(driver, group_id, batch_size=batch_size)

    async def episodic_node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        await self._episode_node_ops.delete_by_uuids(driver, uuids, batch_size=batch_size)

    # -----------------------
    # EpisodicNode: Read
    # -----------------------

    async def episodic_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await self._episode_node_ops.get_by_uuid(driver, uuid)

    async def episodic_node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        return await self._episode_node_ops.get_by_uuids(driver, uuids)

    async def episodic_node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await self._episode_node_ops.get_by_group_ids(driver, group_ids, limit, uuid_cursor)

    async def retrieve_episodes(
        self,
        driver: Any,
        reference_time: datetime,
        last_n: int = 3,
        group_ids: list[str] | None = None,
        source: Any | None = None,
        saga: str | None = None,
    ) -> list[Any]:
        source_str = source.value if source is not None else None
        return await self._episode_node_ops.retrieve_episodes(
            driver, reference_time, last_n, group_ids, source_str, saga
        )

    # -----------------------
    # CommunityNode: Save/Delete
    # -----------------------

    async def community_node_save(self, node: Any, driver: Any) -> None:
        await self._community_node_ops.save(driver, node)

    async def community_node_delete(self, node: Any, driver: Any) -> None:
        await self._community_node_ops.delete(driver, node)

    async def community_node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        await self._community_node_ops.save_bulk(
            driver, nodes, tx=transaction, batch_size=batch_size
        )

    async def community_node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        await self._community_node_ops.delete_by_group_id(driver, group_id, batch_size=batch_size)

    async def community_node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        await self._community_node_ops.delete_by_uuids(driver, uuids, batch_size=batch_size)

    # -----------------------
    # CommunityNode: Read
    # -----------------------

    async def community_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await self._community_node_ops.get_by_uuid(driver, uuid)

    async def community_node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        return await self._community_node_ops.get_by_uuids(driver, uuids)

    async def community_node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await self._community_node_ops.get_by_group_ids(
            driver, group_ids, limit, uuid_cursor
        )

    # -----------------------
    # SagaNode: Save/Delete
    # -----------------------

    async def saga_node_save(self, node: Any, driver: Any) -> None:
        await self._saga_node_ops.save(driver, node)

    async def saga_node_delete(self, node: Any, driver: Any) -> None:
        await self._saga_node_ops.delete(driver, node)

    async def saga_node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        await self._saga_node_ops.save_bulk(driver, nodes, tx=transaction, batch_size=batch_size)

    async def saga_node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        await self._saga_node_ops.delete_by_group_id(driver, group_id, batch_size=batch_size)

    async def saga_node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        await self._saga_node_ops.delete_by_uuids(driver, uuids, batch_size=batch_size)

    # -----------------------
    # SagaNode: Read
    # -----------------------

    async def saga_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await self._saga_node_ops.get_by_uuid(driver, uuid)

    async def saga_node_get_by_uuids(self, _cls: Any, driver: Any, uuids: list[str]) -> list[Any]:
        return await self._saga_node_ops.get_by_uuids(driver, uuids)

    async def saga_node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await self._saga_node_ops.get_by_group_ids(driver, group_ids, limit, uuid_cursor)

    # -----------------------
    # Saga: Queries
    # -----------------------

    async def saga_get_previous_episode_uuid(
        self,
        driver: Any,
        saga_uuid: str,
        current_episode_uuid: str,
    ) -> str | None:
        return await self._saga_node_ops.get_previous_episode_uuid(
            driver, saga_uuid, current_episode_uuid
        )

    async def saga_get_episode_contents(
        self,
        driver: Any,
        saga_uuid: str,
        since: datetime | None = None,
        limit: int = 200,
    ) -> list[tuple[str, datetime | None, datetime]]:
        return await self._saga_node_ops.get_episode_contents(driver, saga_uuid, since, limit)

    # -----------------
    # Edge: Save/Delete
    # -----------------

    async def edge_save(self, edge: Any, driver: Any) -> None:
        await self._entity_edge_ops.save(driver, edge)

    async def edge_delete(self, edge: Any, driver: Any) -> None:
        await self._entity_edge_ops.delete(driver, edge)

    async def edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        await self._entity_edge_ops.save_bulk(driver, edges, tx=transaction, batch_size=batch_size)

    async def edge_update_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        await self._entity_edge_ops.update_bulk(
            driver, edges, tx=transaction, batch_size=batch_size
        )

    async def edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        await self._entity_edge_ops.delete_by_uuids(driver, uuids)

    # -----------------
    # Edge: Read
    # -----------------

    async def edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await self._entity_edge_ops.get_by_uuid(driver, uuid)

    async def edge_get_by_uuids(self, _cls: Any, driver: Any, uuids: list[str]) -> list[Any]:
        return await self._entity_edge_ops.get_by_uuids(driver, uuids)

    async def edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await self._entity_edge_ops.get_by_group_ids(driver, group_ids, limit, uuid_cursor)

    # -----------------
    # Edge: Embeddings (load)
    # -----------------

    async def edge_load_embeddings(self, edge: Any, driver: Any) -> None:
        await self._entity_edge_ops.load_embeddings(driver, edge)

    async def edge_load_embeddings_bulk(
        self,
        driver: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> dict[str, list[float]]:
        return await self._entity_edge_ops.load_embeddings_bulk(driver, edges, batch_size)

    # ---------------------------
    # EpisodicEdge: Save/Delete
    # ---------------------------

    async def episodic_edge_save(self, edge: Any, driver: Any) -> None:
        await self._episodic_edge_ops.save(driver, edge)

    async def episodic_edge_delete(self, edge: Any, driver: Any) -> None:
        await self._episodic_edge_ops.delete(driver, edge)

    async def episodic_edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        await self._episodic_edge_ops.delete_by_uuids(driver, uuids)

    # ---------------------------
    # EpisodicEdge: Read
    # ---------------------------

    async def episodic_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await self._episodic_edge_ops.get_by_uuid(driver, uuid)

    async def episodic_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        return await self._episodic_edge_ops.get_by_uuids(driver, uuids)

    async def episodic_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await self._episodic_edge_ops.get_by_group_ids(driver, group_ids, limit, uuid_cursor)

    # ---------------------------
    # CommunityEdge: Save/Delete
    # ---------------------------

    async def community_edge_save(self, edge: Any, driver: Any) -> None:
        await self._community_edge_ops.save(driver, edge)

    async def community_edge_delete(self, edge: Any, driver: Any) -> None:
        await self._community_edge_ops.delete(driver, edge)

    async def community_edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        await self._community_edge_ops.delete_by_uuids(driver, uuids)

    # ---------------------------
    # CommunityEdge: Read
    # ---------------------------

    async def community_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await self._community_edge_ops.get_by_uuid(driver, uuid)

    async def community_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        return await self._community_edge_ops.get_by_uuids(driver, uuids)

    async def community_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await self._community_edge_ops.get_by_group_ids(
            driver, group_ids, limit, uuid_cursor
        )

    # ---------------------------
    # HasEpisodeEdge: Save/Delete
    # ---------------------------

    async def has_episode_edge_save(self, edge: Any, driver: Any) -> None:
        await self._has_episode_edge_ops.save(driver, edge)

    async def has_episode_edge_delete(self, edge: Any, driver: Any) -> None:
        await self._has_episode_edge_ops.delete(driver, edge)

    async def has_episode_edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        await self._has_episode_edge_ops.save_bulk(
            driver, edges, tx=transaction, batch_size=batch_size
        )

    async def has_episode_edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        await self._has_episode_edge_ops.delete_by_uuids(driver, uuids)

    # ---------------------------
    # HasEpisodeEdge: Read
    # ---------------------------

    async def has_episode_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await self._has_episode_edge_ops.get_by_uuid(driver, uuid)

    async def has_episode_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        return await self._has_episode_edge_ops.get_by_uuids(driver, uuids)

    async def has_episode_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await self._has_episode_edge_ops.get_by_group_ids(
            driver, group_ids, limit, uuid_cursor
        )

    # ----------------------------
    # NextEpisodeEdge: Save/Delete
    # ----------------------------

    async def next_episode_edge_save(self, edge: Any, driver: Any) -> None:
        await self._next_episode_edge_ops.save(driver, edge)

    async def next_episode_edge_delete(self, edge: Any, driver: Any) -> None:
        await self._next_episode_edge_ops.delete(driver, edge)

    async def next_episode_edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        await self._next_episode_edge_ops.save_bulk(
            driver, edges, tx=transaction, batch_size=batch_size
        )

    async def next_episode_edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        await self._next_episode_edge_ops.delete_by_uuids(driver, uuids)

    # ----------------------------
    # NextEpisodeEdge: Read
    # ----------------------------

    async def next_episode_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await self._next_episode_edge_ops.get_by_uuid(driver, uuid)

    async def next_episode_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        return await self._next_episode_edge_ops.get_by_uuids(driver, uuids)

    async def next_episode_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await self._next_episode_edge_ops.get_by_group_ids(
            driver, group_ids, limit, uuid_cursor
        )

    # -----------------
    # Search
    # -----------------

    async def get_mentioned_nodes(
        self,
        driver: Any,
        episodes: list[Any],
    ) -> list[Any]:
        return await self._search_ops.get_mentioned_nodes(driver, episodes)

    async def get_communities_by_nodes(
        self,
        driver: Any,
        nodes: list[Any],
    ) -> list[Any]:
        return await self._search_ops.get_communities_by_nodes(driver, nodes)

    # -----------------
    # Maintenance
    # -----------------

    async def clear_data(
        self,
        driver: Any,
        group_ids: list[str] | None = None,
    ) -> None:
        await self._graph_ops.clear_data(driver, group_ids)

    async def get_community_clusters(
        self,
        driver: Any,
        group_ids: list[str] | None,
    ) -> list[list[Any]]:
        return await self._graph_ops.get_community_clusters(driver, group_ids)

    async def remove_communities(
        self,
        driver: Any,
        group_ids: list[str] | None = None,
    ) -> None:
        await self._graph_ops.remove_communities(driver, group_ids)

    async def determine_entity_community(
        self,
        driver: Any,
        entity: Any,
    ) -> tuple[Any | None, bool]:
        return await self._graph_ops.determine_entity_community(driver, entity)

    # -----------------
    # Additional Node Operations
    # -----------------

    async def episodic_node_get_by_entity_node_uuid(
        self,
        _cls: Any,
        driver: Any,
        entity_node_uuid: str,
    ) -> list[Any]:
        return await self._episode_node_ops.get_by_entity_node_uuid(driver, entity_node_uuid)

    async def community_node_load_name_embedding(
        self,
        node: Any,
        driver: Any,
    ) -> None:
        await self._community_node_ops.load_name_embedding(driver, node)

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
        return await self._entity_edge_ops.get_between_nodes(
            driver, source_node_uuid, target_node_uuid
        )

    async def edge_get_by_node_uuid(
        self,
        _cls: Any,
        driver: Any,
        node_uuid: str,
    ) -> list[Any]:
        return await self._entity_edge_ops.get_by_node_uuid(driver, node_uuid)
