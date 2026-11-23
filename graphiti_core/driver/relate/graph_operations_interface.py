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
from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    HasEpisodeEdge,
    NextEpisodeEdge,
)
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode, SagaNode

_NODE_OPS = (
    (EpisodicNode, 'episode_node_ops'),
    (CommunityNode, 'community_node_ops'),
    (SagaNode, 'saga_node_ops'),
    (EntityNode, 'entity_node_ops'),
)

_EDGE_OPS = (
    (EpisodicEdge, 'episodic_edge_ops'),
    (CommunityEdge, 'community_edge_ops'),
    (HasEpisodeEdge, 'has_episode_edge_ops'),
    (NextEpisodeEdge, 'next_episode_edge_ops'),
    (EntityEdge, 'entity_edge_ops'),
)


def _ops_of_instance(driver: Any, item: Any, table: tuple[tuple[type, str], ...]) -> Any:
    for model, attr in table:
        if isinstance(item, model):
            return getattr(driver, attr)
    raise NotImplementedError(f'no Relate operations for {type(item).__name__}')


def _ops_of_class(driver: Any, cls: Any, table: tuple[tuple[type, str], ...]) -> list[Any]:
    """The operations of the model `cls`, or of every model when `cls` is the base class."""
    if isinstance(cls, type):
        for model, attr in table:
            if issubclass(cls, model):
                return [getattr(driver, attr)]
    return [getattr(driver, attr) for _, attr in table if attr != 'saga_node_ops']


class RelateGraphOperationsInterface(GraphOperationsInterface):
    """Legacy `GraphOperationsInterface` of the Relate driver.

    Every method delegates to the operations of the driver (`driver.entity_node_ops` and
    the others), which hold the Cypher. Where a method receives query records instead of
    model objects, it calls the `*_records` method of the operations.

    Methods that this class does not override raise `NotImplementedError`, and graphiti
    core runs its generic Cypher instead, which works unchanged on Relate.
    """

    # --- Entity nodes ---

    async def node_save(self, node: Any, driver: Any) -> None:
        await _ops_of_instance(driver, node, _NODE_OPS).save(driver, node)

    async def node_delete(self, node: Any, driver: Any) -> None:
        await _ops_of_instance(driver, node, _NODE_OPS).delete(driver, node)

    async def node_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, nodes: list[Any], batch_size: int = 100
    ) -> None:
        await driver.entity_node_ops.save_bulk_records(driver, nodes, transaction)

    async def node_update_bulk(
        self, _cls: Any, driver: Any, transaction: Any, nodes: list[Any], batch_size: int = 100
    ) -> None:
        await driver.entity_node_ops.update_bulk_records(driver, nodes, transaction)

    async def node_delete_by_group_id(
        self, _cls: Any, driver: Any, group_id: str, batch_size: int = 100
    ) -> None:
        for ops in _ops_of_class(driver, _cls, _NODE_OPS):
            await ops.delete_by_group_id(driver, group_id, batch_size=batch_size)

    async def node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        for ops in _ops_of_class(driver, _cls, _NODE_OPS):
            await ops.delete_by_uuids(driver, uuids, batch_size=batch_size)

    async def node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await driver.entity_node_ops.get_by_uuid(driver, uuid)

    async def node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str], group_id: str | None = None
    ) -> list[Any]:
        nodes = await driver.entity_node_ops.get_by_uuids(driver, uuids)
        if group_id is not None:
            nodes = [node for node in nodes if node.group_id == group_id]
        return nodes

    async def node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await driver.entity_node_ops.get_by_group_ids(driver, group_ids, limit, uuid_cursor)

    async def node_get_by_attribute_values(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        attributes: dict[str, Any],
        node_labels: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        return await driver.entity_node_ops.get_by_attribute_values(
            driver, group_id, attributes, node_labels, limit
        )

    async def node_load_embeddings(self, node: Any, driver: Any) -> None:
        await driver.entity_node_ops.load_embeddings(driver, node)

    async def node_load_embeddings_bulk(
        self, driver: Any, nodes: list[Any], batch_size: int = 100
    ) -> dict[str, list[float]]:
        await driver.entity_node_ops.load_embeddings_bulk(driver, nodes, batch_size)
        return {node.uuid: node.name_embedding for node in nodes if node.name_embedding}

    # --- Episodes ---

    async def episodic_node_save(self, node: Any, driver: Any) -> None:
        await driver.episode_node_ops.save(driver, node)

    async def episodic_node_delete(self, node: Any, driver: Any) -> None:
        await driver.episode_node_ops.delete(driver, node)

    async def episodic_node_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, nodes: list[Any], batch_size: int = 100
    ) -> None:
        await driver.episode_node_ops.save_bulk_records(driver, nodes, transaction)

    async def episodic_edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        episodic_edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        await driver.episodic_edge_ops.save_bulk_records(driver, episodic_edges, transaction)

    async def episodic_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await driver.episode_node_ops.get_by_uuid(driver, uuid)

    async def episodic_node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        return await driver.episode_node_ops.get_by_uuids(driver, uuids)

    async def episodic_node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await driver.episode_node_ops.get_by_group_ids(driver, group_ids, limit, uuid_cursor)

    async def episodic_node_get_by_entity_node_uuid(
        self, _cls: Any, driver: Any, entity_node_uuid: str
    ) -> list[Any]:
        return await driver.episode_node_ops.get_by_entity_node_uuid(driver, entity_node_uuid)

    async def retrieve_episodes(
        self,
        driver: Any,
        reference_time: datetime,
        last_n: int = 3,
        group_ids: list[str] | None = None,
        source: Any | None = None,
        saga: str | None = None,
    ) -> list[Any]:
        return await driver.episode_node_ops.retrieve_episodes(
            driver,
            reference_time,
            last_n,
            group_ids,
            getattr(source, 'value', source),
            saga,
        )

    # --- Communities ---

    async def community_node_save(self, node: Any, driver: Any) -> None:
        await driver.community_node_ops.save(driver, node)

    async def community_node_delete(self, node: Any, driver: Any) -> None:
        await driver.community_node_ops.delete(driver, node)

    async def community_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await driver.community_node_ops.get_by_uuid(driver, uuid)

    async def community_node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        return await driver.community_node_ops.get_by_uuids(driver, uuids)

    async def community_node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await driver.community_node_ops.get_by_group_ids(
            driver, group_ids, limit, uuid_cursor
        )

    async def community_node_load_name_embedding(self, node: Any, driver: Any) -> None:
        await driver.community_node_ops.load_name_embedding(driver, node)

    # --- Sagas ---

    async def saga_node_save(self, node: Any, driver: Any) -> None:
        await driver.saga_node_ops.save(driver, node)

    async def saga_node_delete(self, node: Any, driver: Any) -> None:
        await driver.saga_node_ops.delete(driver, node)

    async def saga_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await driver.saga_node_ops.get_by_uuid(driver, uuid)

    async def saga_node_get_by_uuids(self, _cls: Any, driver: Any, uuids: list[str]) -> list[Any]:
        return await driver.saga_node_ops.get_by_uuids(driver, uuids)

    async def saga_node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await driver.saga_node_ops.get_by_group_ids(driver, group_ids, limit, uuid_cursor)

    # --- Entity edges ---

    async def edge_save(self, edge: Any, driver: Any) -> None:
        await driver.entity_edge_ops.save(driver, edge)

    async def edge_delete(self, edge: Any, driver: Any) -> None:
        await _ops_of_instance(driver, edge, _EDGE_OPS).delete(driver, edge)

    async def edge_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, edges: list[Any], batch_size: int = 100
    ) -> None:
        await driver.entity_edge_ops.save_bulk_records(driver, edges, transaction)

    async def edge_update_bulk(
        self, _cls: Any, driver: Any, transaction: Any, edges: list[Any], batch_size: int = 100
    ) -> None:
        await driver.entity_edge_ops.update_bulk_records(driver, edges, transaction)

    async def edge_delete_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str], group_id: str | None = None
    ) -> None:
        for ops in _ops_of_class(driver, _cls, _EDGE_OPS):
            await ops.delete_by_uuids(driver, uuids)

    async def edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await driver.entity_edge_ops.get_by_uuid(driver, uuid)

    async def edge_get_by_uuids(self, _cls: Any, driver: Any, uuids: list[str]) -> list[Any]:
        return await driver.entity_edge_ops.get_by_uuids(driver, uuids)

    async def edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await driver.entity_edge_ops.get_by_group_ids(driver, group_ids, limit, uuid_cursor)

    async def edge_get_between_nodes(
        self, _cls: Any, driver: Any, source_node_uuid: str, target_node_uuid: str
    ) -> list[Any]:
        return await driver.entity_edge_ops.get_between_nodes(
            driver, source_node_uuid, target_node_uuid
        )

    async def edge_get_by_node_uuid(self, _cls: Any, driver: Any, node_uuid: str) -> list[Any]:
        return await driver.entity_edge_ops.get_by_node_uuid(driver, node_uuid)

    async def edge_load_embeddings(self, edge: Any, driver: Any) -> None:
        await driver.entity_edge_ops.load_embeddings(driver, edge)

    async def edge_load_embeddings_bulk(
        self, driver: Any, edges: list[Any], batch_size: int = 100
    ) -> dict[str, list[float]]:
        await driver.entity_edge_ops.load_embeddings_bulk(driver, edges, batch_size)
        return {edge.uuid: edge.fact_embedding for edge in edges if edge.fact_embedding}

    # --- Episodic edges ---

    async def episodic_edge_save(self, edge: Any, driver: Any) -> None:
        await driver.episodic_edge_ops.save(driver, edge)

    async def episodic_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await driver.episodic_edge_ops.get_by_uuid(driver, uuid)

    async def episodic_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        return await driver.episodic_edge_ops.get_by_uuids(driver, uuids)

    async def episodic_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await driver.episodic_edge_ops.get_by_group_ids(
            driver, group_ids, limit, uuid_cursor
        )

    # --- Community edges ---

    async def community_edge_save(self, edge: Any, driver: Any) -> None:
        await driver.community_edge_ops.save(driver, edge)

    async def community_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await driver.community_edge_ops.get_by_uuid(driver, uuid)

    async def community_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        return await driver.community_edge_ops.get_by_uuids(driver, uuids)

    async def community_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await driver.community_edge_ops.get_by_group_ids(
            driver, group_ids, limit, uuid_cursor
        )

    # --- Has-episode edges ---

    async def has_episode_edge_save(self, edge: Any, driver: Any) -> None:
        await driver.has_episode_edge_ops.save(driver, edge)

    async def has_episode_edge_delete(self, edge: Any, driver: Any) -> None:
        await driver.has_episode_edge_ops.delete(driver, edge)

    async def has_episode_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await driver.has_episode_edge_ops.get_by_uuid(driver, uuid)

    async def has_episode_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        return await driver.has_episode_edge_ops.get_by_uuids(driver, uuids)

    async def has_episode_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await driver.has_episode_edge_ops.get_by_group_ids(
            driver, group_ids, limit, uuid_cursor
        )

    # --- Next-episode edges ---

    async def next_episode_edge_save(self, edge: Any, driver: Any) -> None:
        await driver.next_episode_edge_ops.save(driver, edge)

    async def next_episode_edge_delete(self, edge: Any, driver: Any) -> None:
        await driver.next_episode_edge_ops.delete(driver, edge)

    async def next_episode_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await driver.next_episode_edge_ops.get_by_uuid(driver, uuid)

    async def next_episode_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        return await driver.next_episode_edge_ops.get_by_uuids(driver, uuids)

    async def next_episode_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await driver.next_episode_edge_ops.get_by_group_ids(
            driver, group_ids, limit, uuid_cursor
        )

    # --- Graph maintenance ---

    async def get_mentioned_nodes(self, driver: Any, episodes: list[Any]) -> list[Any]:
        return await driver.graph_ops.get_mentioned_nodes(driver, episodes)

    async def get_communities_by_nodes(self, driver: Any, nodes: list[Any]) -> list[Any]:
        return await driver.graph_ops.get_communities_by_nodes(driver, nodes)

    async def clear_data(self, driver: Any, group_ids: list[str] | None = None) -> None:
        await driver.graph_ops.clear_data(driver, group_ids)

    async def get_community_clusters(
        self, driver: Any, group_ids: list[str] | None
    ) -> list[list[Any]]:
        return await driver.graph_ops.get_community_clusters(driver, group_ids)

    async def remove_communities(self, driver: Any, group_ids: list[str] | None = None) -> None:
        await driver.graph_ops.remove_communities(driver, group_ids)
