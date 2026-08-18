from __future__ import annotations

from typing import Any

from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface


class PostGraphOperationsInterface(GraphOperationsInterface):
    """Bridges the Graphiti model-layer API to PostGraph's SQL operations."""

    def __init__(self, driver: Any):
        super().__init__()
        self._driver = driver

    # --- Entity Node ---

    async def node_save(self, node: Any, driver: Any) -> None:
        await driver.entity_node_ops.save(driver, node)

    async def node_delete(self, node: Any, driver: Any) -> None:
        await driver.entity_node_ops.delete(driver, node)

    async def node_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, nodes: list[Any], batch_size: int = 100,
    ) -> None:
        await driver.entity_node_ops.save_bulk(driver, nodes, tx=transaction, batch_size=batch_size)

    async def node_delete_by_group_id(
        self, _cls: Any, driver: Any, group_id: str, batch_size: int = 100,
    ) -> None:
        await driver.entity_node_ops.delete_by_group_id(driver, group_id, batch_size=batch_size)

    async def node_delete_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str], group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        await driver.entity_node_ops.delete_by_uuids(driver, uuids, batch_size=batch_size)

    async def node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await driver.entity_node_ops.get_by_uuid(driver, uuid)

    async def node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str], group_id: str | None = None,
    ) -> list[Any]:
        return await driver.entity_node_ops.get_by_uuids(driver, uuids)

    async def node_get_by_group_ids(
        self, _cls: Any, driver: Any, group_ids: list[str],
        limit: int | None = None, uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await driver.entity_node_ops.get_by_group_ids(
            driver, group_ids, limit=limit, uuid_cursor=uuid_cursor,
        )

    async def node_load_embeddings(self, node: Any, driver: Any) -> None:
        await driver.entity_node_ops.load_embeddings(driver, node)

    async def node_load_embeddings_bulk(
        self, driver: Any, nodes: list[Any], batch_size: int = 100,
    ) -> dict[str, list[float]]:
        await driver.entity_node_ops.load_embeddings_bulk(driver, nodes, batch_size=batch_size)
        return {n.uuid: n.name_embedding for n in nodes if n.name_embedding is not None}

    # --- Episodic Node ---

    async def episodic_node_save(self, node: Any, driver: Any) -> None:
        await driver.episode_node_ops.save(driver, node)

    async def episodic_node_delete(self, node: Any, driver: Any) -> None:
        await driver.episode_node_ops.delete(driver, node)

    async def episodic_node_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, nodes: list[Any], batch_size: int = 100,
    ) -> None:
        await driver.episode_node_ops.save_bulk(driver, nodes, tx=transaction, batch_size=batch_size)

    async def episodic_node_delete_by_group_id(
        self, _cls: Any, driver: Any, group_id: str, batch_size: int = 100,
    ) -> None:
        await driver.episode_node_ops.delete_by_group_id(driver, group_id, batch_size=batch_size)

    async def episodic_node_delete_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str], group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        await driver.episode_node_ops.delete_by_uuids(driver, uuids, batch_size=batch_size)

    async def episodic_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await driver.episode_node_ops.get_by_uuid(driver, uuid)

    async def episodic_node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str],
    ) -> list[Any]:
        return await driver.episode_node_ops.get_by_uuids(driver, uuids)

    async def episodic_node_get_by_group_ids(
        self, _cls: Any, driver: Any, group_ids: list[str],
        limit: int | None = None, uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await driver.episode_node_ops.get_by_group_ids(
            driver, group_ids, limit=limit, uuid_cursor=uuid_cursor,
        )

    async def retrieve_episodes(
        self, driver: Any, reference_time: Any, last_n: int = 3,
        group_ids: list[str] | None = None, source: Any | None = None, saga: str | None = None,
    ) -> list[Any]:
        return await driver.episode_node_ops.retrieve_episodes(
            driver, reference_time, last_n=last_n, group_ids=group_ids,
            source=source, saga=saga,
        )

    async def episodic_node_get_by_entity_node_uuid(
        self, _cls: Any, driver: Any, entity_node_uuid: str,
    ) -> list[Any]:
        return await driver.episode_node_ops.get_by_entity_node_uuid(driver, entity_node_uuid)

    # --- Community Node ---

    async def community_node_save(self, node: Any, driver: Any) -> None:
        await driver.community_node_ops.save(driver, node)

    async def community_node_delete(self, node: Any, driver: Any) -> None:
        await driver.community_node_ops.delete(driver, node)

    async def community_node_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, nodes: list[Any], batch_size: int = 100,
    ) -> None:
        await driver.community_node_ops.save_bulk(
            driver, nodes, tx=transaction, batch_size=batch_size,
        )

    async def community_node_delete_by_group_id(
        self, _cls: Any, driver: Any, group_id: str, batch_size: int = 100,
    ) -> None:
        await driver.community_node_ops.delete_by_group_id(driver, group_id, batch_size=batch_size)

    async def community_node_delete_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str], group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        await driver.community_node_ops.delete_by_uuids(driver, uuids, batch_size=batch_size)

    async def community_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await driver.community_node_ops.get_by_uuid(driver, uuid)

    async def community_node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str],
    ) -> list[Any]:
        return await driver.community_node_ops.get_by_uuids(driver, uuids)

    async def community_node_get_by_group_ids(
        self, _cls: Any, driver: Any, group_ids: list[str],
        limit: int | None = None, uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await driver.community_node_ops.get_by_group_ids(
            driver, group_ids, limit=limit, uuid_cursor=uuid_cursor,
        )

    async def community_node_load_name_embedding(self, node: Any, driver: Any) -> None:
        await driver.community_node_ops.load_name_embedding(driver, node)

    # --- Saga Node ---

    async def saga_node_save(self, node: Any, driver: Any) -> None:
        await driver.saga_node_ops.save(driver, node)

    async def saga_node_delete(self, node: Any, driver: Any) -> None:
        await driver.saga_node_ops.delete(driver, node)

    async def saga_node_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, nodes: list[Any], batch_size: int = 100,
    ) -> None:
        await driver.saga_node_ops.save_bulk(
            driver, nodes, tx=transaction, batch_size=batch_size,
        )

    async def saga_node_delete_by_group_id(
        self, _cls: Any, driver: Any, group_id: str, batch_size: int = 100,
    ) -> None:
        await driver.saga_node_ops.delete_by_group_id(driver, group_id, batch_size=batch_size)

    async def saga_node_delete_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str], group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        await driver.saga_node_ops.delete_by_uuids(driver, uuids, batch_size=batch_size)

    async def saga_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await driver.saga_node_ops.get_by_uuid(driver, uuid)

    async def saga_node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str],
    ) -> list[Any]:
        return await driver.saga_node_ops.get_by_uuids(driver, uuids)

    async def saga_node_get_by_group_ids(
        self, _cls: Any, driver: Any, group_ids: list[str],
        limit: int | None = None, uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await driver.saga_node_ops.get_by_group_ids(
            driver, group_ids, limit=limit, uuid_cursor=uuid_cursor,
        )

    # --- Entity Edge ---

    async def edge_save(self, edge: Any, driver: Any) -> None:
        await driver.entity_edge_ops.save(driver, edge)

    async def edge_delete(self, edge: Any, driver: Any) -> None:
        await driver.entity_edge_ops.delete(driver, edge)

    async def edge_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, edges: list[Any], batch_size: int = 100,
    ) -> None:
        await driver.entity_edge_ops.save_bulk(
            driver, edges, tx=transaction, batch_size=batch_size,
        )

    async def edge_delete_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str], group_id: str | None = None,
    ) -> None:
        await driver.entity_edge_ops.delete_by_uuids(driver, uuids)

    async def edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await driver.entity_edge_ops.get_by_uuid(driver, uuid)

    async def edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str],
    ) -> list[Any]:
        return await driver.entity_edge_ops.get_by_uuids(driver, uuids)

    async def edge_get_by_group_ids(
        self, _cls: Any, driver: Any, group_ids: list[str],
        limit: int | None = None, uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await driver.entity_edge_ops.get_by_group_ids(
            driver, group_ids, limit=limit, uuid_cursor=uuid_cursor,
        )

    async def edge_load_embeddings(self, edge: Any, driver: Any) -> None:
        await driver.entity_edge_ops.load_embeddings(driver, edge)

    async def edge_load_embeddings_bulk(
        self, driver: Any, edges: list[Any], batch_size: int = 100,
    ) -> dict[str, list[float]]:
        await driver.entity_edge_ops.load_embeddings_bulk(driver, edges, batch_size=batch_size)
        return {e.uuid: e.fact_embedding for e in edges if e.fact_embedding is not None}

    async def edge_get_between_nodes(
        self, _cls: Any, driver: Any, source_node_uuid: str, target_node_uuid: str,
    ) -> list[Any]:
        return await driver.entity_edge_ops.get_between_nodes(
            driver, source_node_uuid, target_node_uuid,
        )

    async def edge_get_by_node_uuid(
        self, _cls: Any, driver: Any, node_uuid: str,
    ) -> list[Any]:
        return await driver.entity_edge_ops.get_by_node_uuid(driver, node_uuid)

    # --- Episodic Edge ---

    async def episodic_edge_save(self, edge: Any, driver: Any) -> None:
        await driver.episodic_edge_ops.save(driver, edge)

    async def episodic_edge_delete(self, edge: Any, driver: Any) -> None:
        await driver.episodic_edge_ops.delete(driver, edge)

    async def episodic_edge_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any,
        episodic_edges: list[Any], batch_size: int = 100,
    ) -> None:
        await driver.episodic_edge_ops.save_bulk(
            driver, episodic_edges, tx=transaction, batch_size=batch_size,
        )

    async def episodic_edge_delete_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str], group_id: str | None = None,
    ) -> None:
        await driver.episodic_edge_ops.delete_by_uuids(driver, uuids)

    async def episodic_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await driver.episodic_edge_ops.get_by_uuid(driver, uuid)

    async def episodic_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str],
    ) -> list[Any]:
        return await driver.episodic_edge_ops.get_by_uuids(driver, uuids)

    async def episodic_edge_get_by_group_ids(
        self, _cls: Any, driver: Any, group_ids: list[str],
        limit: int | None = None, uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await driver.episodic_edge_ops.get_by_group_ids(
            driver, group_ids, limit=limit, uuid_cursor=uuid_cursor,
        )

    # --- Community Edge ---

    async def community_edge_save(self, edge: Any, driver: Any) -> None:
        await driver.community_edge_ops.save(driver, edge)

    async def community_edge_delete(self, edge: Any, driver: Any) -> None:
        await driver.community_edge_ops.delete(driver, edge)

    async def community_edge_delete_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str], group_id: str | None = None,
    ) -> None:
        await driver.community_edge_ops.delete_by_uuids(driver, uuids)

    async def community_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await driver.community_edge_ops.get_by_uuid(driver, uuid)

    async def community_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str],
    ) -> list[Any]:
        return await driver.community_edge_ops.get_by_uuids(driver, uuids)

    async def community_edge_get_by_group_ids(
        self, _cls: Any, driver: Any, group_ids: list[str],
        limit: int | None = None, uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await driver.community_edge_ops.get_by_group_ids(
            driver, group_ids, limit=limit, uuid_cursor=uuid_cursor,
        )

    # --- Has Episode Edge ---

    async def has_episode_edge_save(self, edge: Any, driver: Any) -> None:
        await driver.has_episode_edge_ops.save(driver, edge)

    async def has_episode_edge_delete(self, edge: Any, driver: Any) -> None:
        await driver.has_episode_edge_ops.delete(driver, edge)

    async def has_episode_edge_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, edges: list[Any], batch_size: int = 100,
    ) -> None:
        await driver.has_episode_edge_ops.save_bulk(
            driver, edges, tx=transaction, batch_size=batch_size,
        )

    async def has_episode_edge_delete_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str], group_id: str | None = None,
    ) -> None:
        await driver.has_episode_edge_ops.delete_by_uuids(driver, uuids)

    async def has_episode_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await driver.has_episode_edge_ops.get_by_uuid(driver, uuid)

    async def has_episode_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str],
    ) -> list[Any]:
        return await driver.has_episode_edge_ops.get_by_uuids(driver, uuids)

    async def has_episode_edge_get_by_group_ids(
        self, _cls: Any, driver: Any, group_ids: list[str],
        limit: int | None = None, uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await driver.has_episode_edge_ops.get_by_group_ids(
            driver, group_ids, limit=limit, uuid_cursor=uuid_cursor,
        )

    # --- Next Episode Edge ---

    async def next_episode_edge_save(self, edge: Any, driver: Any) -> None:
        await driver.next_episode_edge_ops.save(driver, edge)

    async def next_episode_edge_delete(self, edge: Any, driver: Any) -> None:
        await driver.next_episode_edge_ops.delete(driver, edge)

    async def next_episode_edge_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, edges: list[Any], batch_size: int = 100,
    ) -> None:
        await driver.next_episode_edge_ops.save_bulk(
            driver, edges, tx=transaction, batch_size=batch_size,
        )

    async def next_episode_edge_delete_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str], group_id: str | None = None,
    ) -> None:
        await driver.next_episode_edge_ops.delete_by_uuids(driver, uuids)

    async def next_episode_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await driver.next_episode_edge_ops.get_by_uuid(driver, uuid)

    async def next_episode_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str],
    ) -> list[Any]:
        return await driver.next_episode_edge_ops.get_by_uuids(driver, uuids)

    async def next_episode_edge_get_by_group_ids(
        self, _cls: Any, driver: Any, group_ids: list[str],
        limit: int | None = None, uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await driver.next_episode_edge_ops.get_by_group_ids(
            driver, group_ids, limit=limit, uuid_cursor=uuid_cursor,
        )

    # --- Maintenance ---

    async def clear_data(self, driver: Any, group_ids: list[str] | None = None) -> None:
        await driver.graph_ops.clear_data(driver, group_ids=group_ids)

    async def get_community_clusters(
        self, driver: Any, group_ids: list[str] | None,
    ) -> list[list[Any]]:
        return await driver.graph_ops.get_community_clusters(driver, group_ids=group_ids)

    async def remove_communities(
        self, driver: Any, group_ids: list[str] | None = None,
    ) -> None:
        await driver.graph_ops.remove_communities(driver, group_ids=group_ids)

    async def determine_entity_community(
        self, driver: Any, entity: Any,
    ) -> tuple[Any | None, bool]:
        return await driver.graph_ops.determine_entity_community(driver, entity)

    async def get_mentioned_nodes(self, driver: Any, episodes: list[Any]) -> list[Any]:
        return await driver.graph_ops.get_mentioned_nodes(driver, episodes)

    async def get_communities_by_nodes(self, driver: Any, nodes: list[Any]) -> list[Any]:
        return await driver.graph_ops.get_communities_by_nodes(driver, nodes)
