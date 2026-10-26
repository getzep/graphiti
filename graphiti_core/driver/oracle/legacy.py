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

"""Compatibility bridge from Graphiti's legacy model API to Oracle operations.

Graphiti 0.30 still routes several public node and edge helpers through
``graph_operations_interface``.  Keeping the bridge explicit makes the fork
small and, importantly, guarantees that those helpers reach Oracle SQL rather
than the built-in Cypher fallback.
"""

from __future__ import annotations

from typing import Any

from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface
from graphiti_core.driver.search_interface.search_interface import SearchInterface
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.nodes import EntityNode, EpisodicNode

from .operations.search import OracleSearchOperations


def _typed_model(value: Any, model: type[Any], *, preserve_extra_attributes: bool = False) -> Any:
    if isinstance(value, model):
        return value
    if not isinstance(value, dict):
        raise TypeError('Graphiti bulk value must be a model or mapping')
    fields = dict(value)
    if model is EpisodicNode:
        fields.setdefault('labels', [])
    if preserve_extra_attributes:
        attributes = dict(fields.pop('attributes', {}) or {})
        for key in list(fields):
            if key not in model.model_fields:
                attributes[key] = fields.pop(key)
        fields['attributes'] = attributes
    return model.model_validate(fields)


class OracleLegacyGraphOperations(GraphOperationsInterface):
    """Delegate every legacy Graphiti operation to the typed Oracle families."""

    async def node_save(self, node: Any, driver: Any) -> None:
        await driver.entity_node_ops.save(driver, node)

    async def node_delete(self, node: Any, driver: Any) -> None:
        await driver.entity_node_ops.delete(driver, node)

    async def node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        typed_nodes = [
            _typed_model(node, EntityNode, preserve_extra_attributes=True) for node in nodes
        ]
        await driver.entity_node_ops.save_bulk(
            driver, typed_nodes, tx=transaction, batch_size=batch_size
        )

    async def node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        await driver.entity_node_ops.delete_by_group_id(driver, group_id, batch_size=batch_size)

    async def node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        await driver.entity_node_ops.delete_by_uuids(
            driver,
            uuids,
            batch_size=batch_size,
            group_id=group_id,
        )

    async def node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await driver.entity_node_ops.get_by_uuid(driver, uuid)

    async def node_get_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> list[Any]:
        return await driver.entity_node_ops.get_by_uuids(driver, uuids, group_id=group_id)

    async def node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await driver.entity_node_ops.get_by_group_ids(
            driver, group_ids, limit=limit, uuid_cursor=uuid_cursor
        )

    async def node_get_by_names(
        self,
        _cls: Any,
        driver: Any,
        names: list[str],
        group_ids: list[str],
    ) -> list[Any]:
        return await driver.entity_node_ops.get_by_names(driver, names, group_ids)

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
            driver,
            group_id,
            attributes,
            node_labels=node_labels,
            limit=limit,
        )

    async def node_load_embeddings(self, node: Any, driver: Any) -> None:
        await driver.entity_node_ops.load_embeddings(driver, node)

    async def node_load_embeddings_bulk(
        self,
        driver: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> dict[str, list[float]]:
        await driver.entity_node_ops.load_embeddings_bulk(driver, nodes, batch_size=batch_size)
        return {node.uuid: node.name_embedding for node in nodes if node.name_embedding is not None}

    async def episodic_node_save(self, node: Any, driver: Any) -> None:
        await driver.episode_node_ops.save(driver, node)

    async def episodic_node_delete(self, node: Any, driver: Any) -> None:
        await driver.episode_node_ops.delete(driver, node)

    async def episodic_node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        typed_nodes = [_typed_model(node, EpisodicNode) for node in nodes]
        await driver.episode_node_ops.save_bulk(
            driver, typed_nodes, tx=transaction, batch_size=batch_size
        )

    async def episodic_edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        episodic_edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        typed_edges = [_typed_model(edge, EpisodicEdge) for edge in episodic_edges]
        await driver.episodic_edge_ops.save_bulk(
            driver,
            typed_edges,
            tx=transaction,
            batch_size=batch_size,
        )

    async def episodic_node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        await driver.episode_node_ops.delete_by_group_id(driver, group_id, batch_size=batch_size)

    async def episodic_node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        await driver.episode_node_ops.delete_by_uuids(
            driver,
            uuids,
            batch_size=batch_size,
            group_id=group_id,
        )

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
        return await driver.episode_node_ops.get_by_group_ids(
            driver, group_ids, limit=limit, uuid_cursor=uuid_cursor
        )

    async def retrieve_episodes(
        self,
        driver: Any,
        reference_time: Any,
        last_n: int = 3,
        group_ids: list[str] | None = None,
        source: Any | None = None,
        saga: str | None = None,
    ) -> list[Any]:
        source_value = getattr(source, 'value', source)
        return await driver.episode_node_ops.retrieve_episodes(
            driver,
            reference_time,
            last_n=last_n,
            group_ids=group_ids,
            source=source_value,
            saga=saga,
        )

    async def community_node_save(self, node: Any, driver: Any) -> None:
        await driver.community_node_ops.save(driver, node)

    async def community_node_delete(self, node: Any, driver: Any) -> None:
        await driver.community_node_ops.delete(driver, node)

    async def community_node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        await driver.community_node_ops.save_bulk(
            driver, nodes, tx=transaction, batch_size=batch_size
        )

    async def community_node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        await driver.community_node_ops.delete_by_group_id(driver, group_id, batch_size=batch_size)

    async def community_node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        await driver.community_node_ops.delete_by_uuids(
            driver,
            uuids,
            batch_size=batch_size,
            group_id=group_id,
        )

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
            driver, group_ids, limit=limit, uuid_cursor=uuid_cursor
        )

    async def saga_node_save(self, node: Any, driver: Any) -> None:
        await driver.saga_node_ops.save(driver, node)

    async def saga_node_delete(self, node: Any, driver: Any) -> None:
        await driver.saga_node_ops.delete(driver, node)

    async def saga_node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        await driver.saga_node_ops.save_bulk(driver, nodes, tx=transaction, batch_size=batch_size)

    async def saga_node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        await driver.saga_node_ops.delete_by_group_id(driver, group_id, batch_size=batch_size)

    async def saga_node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        await driver.saga_node_ops.delete_by_uuids(
            driver,
            uuids,
            batch_size=batch_size,
            group_id=group_id,
        )

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
        return await driver.saga_node_ops.get_by_group_ids(
            driver, group_ids, limit=limit, uuid_cursor=uuid_cursor
        )

    async def saga_get_previous_episode_uuid(
        self,
        driver: Any,
        saga_uuid: str,
        current_episode_uuid: str,
    ) -> str | None:
        return await driver.saga_node_ops.get_previous_episode_uuid(
            driver, saga_uuid, current_episode_uuid
        )

    async def saga_get_episode_contents(
        self,
        driver: Any,
        saga_uuid: str,
        since: Any | None = None,
        limit: int = 200,
        *,
        after_uuid: str | None = None,
    ) -> list[tuple[str, Any, Any, str]]:
        return await driver.saga_node_ops.get_episode_contents(
            driver,
            saga_uuid,
            since=since,
            after_uuid=after_uuid,
            limit=limit,
        )

    async def edge_save(self, edge: Any, driver: Any) -> None:
        await driver.entity_edge_ops.save(driver, edge)

    async def edge_delete(self, edge: Any, driver: Any) -> None:
        await driver.entity_edge_ops.delete(driver, edge)

    async def edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        typed_edges = [
            _typed_model(edge, EntityEdge, preserve_extra_attributes=True) for edge in edges
        ]
        await driver.entity_edge_ops.save_bulk(
            driver, typed_edges, tx=transaction, batch_size=batch_size
        )

    async def edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        await driver.entity_edge_ops.delete_by_uuids(driver, uuids, group_id=group_id)

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
        return await driver.entity_edge_ops.get_by_group_ids(
            driver, group_ids, limit=limit, uuid_cursor=uuid_cursor
        )

    async def edge_load_embeddings(self, edge: Any, driver: Any) -> None:
        await driver.entity_edge_ops.load_embeddings(driver, edge)

    async def edge_load_embeddings_bulk(
        self,
        driver: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> dict[str, list[float]]:
        await driver.entity_edge_ops.load_embeddings_bulk(driver, edges, batch_size=batch_size)
        return {edge.uuid: edge.fact_embedding for edge in edges if edge.fact_embedding is not None}

    async def episodic_edge_save(self, edge: Any, driver: Any) -> None:
        await driver.episodic_edge_ops.save(driver, edge)

    async def episodic_edge_delete(self, edge: Any, driver: Any) -> None:
        await driver.episodic_edge_ops.delete(driver, edge)

    async def episodic_edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        await driver.episodic_edge_ops.delete_by_uuids(driver, uuids, group_id=group_id)

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
            driver, group_ids, limit=limit, uuid_cursor=uuid_cursor
        )

    async def community_edge_save(self, edge: Any, driver: Any) -> None:
        await driver.community_edge_ops.save(driver, edge)

    async def community_edge_delete(self, edge: Any, driver: Any) -> None:
        await driver.community_edge_ops.delete(driver, edge)

    async def community_edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        await driver.community_edge_ops.delete_by_uuids(driver, uuids, group_id=group_id)

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
            driver, group_ids, limit=limit, uuid_cursor=uuid_cursor
        )

    async def has_episode_edge_save(self, edge: Any, driver: Any) -> None:
        await driver.has_episode_edge_ops.save(driver, edge)

    async def has_episode_edge_delete(self, edge: Any, driver: Any) -> None:
        await driver.has_episode_edge_ops.delete(driver, edge)

    async def has_episode_edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        await driver.has_episode_edge_ops.save_bulk(
            driver, edges, tx=transaction, batch_size=batch_size
        )

    async def has_episode_edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        await driver.has_episode_edge_ops.delete_by_uuids(driver, uuids, group_id=group_id)

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
            driver, group_ids, limit=limit, uuid_cursor=uuid_cursor
        )

    async def next_episode_edge_save(self, edge: Any, driver: Any) -> None:
        await driver.next_episode_edge_ops.save(driver, edge)

    async def next_episode_edge_delete(self, edge: Any, driver: Any) -> None:
        await driver.next_episode_edge_ops.delete(driver, edge)

    async def next_episode_edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        await driver.next_episode_edge_ops.save_bulk(
            driver, edges, tx=transaction, batch_size=batch_size
        )

    async def next_episode_edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        await driver.next_episode_edge_ops.delete_by_uuids(driver, uuids, group_id=group_id)

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
            driver, group_ids, limit=limit, uuid_cursor=uuid_cursor
        )

    async def get_mentioned_nodes(
        self,
        driver: Any,
        episodes: list[Any],
    ) -> list[Any]:
        return await driver.graph_ops.get_mentioned_nodes(driver, episodes)

    async def get_communities_by_nodes(
        self,
        driver: Any,
        nodes: list[Any],
    ) -> list[Any]:
        return await driver.graph_ops.get_communities_by_nodes(driver, nodes)

    async def clear_data(
        self,
        driver: Any,
        group_ids: list[str] | None = None,
    ) -> None:
        await driver.graph_ops.clear_data(driver, group_ids)

    async def get_community_clusters(
        self,
        driver: Any,
        group_ids: list[str] | None,
    ) -> list[list[Any]]:
        return await driver.graph_ops.get_community_clusters(driver, group_ids)

    async def remove_communities(
        self,
        driver: Any,
        group_ids: list[str] | None = None,
    ) -> None:
        await driver.graph_ops.remove_communities(driver, group_ids)

    async def determine_entity_community(
        self,
        driver: Any,
        entity: Any,
    ) -> tuple[Any | None, bool]:
        return await driver.graph_ops.determine_entity_community_result(driver, entity)

    async def episodic_node_get_by_entity_node_uuid(
        self,
        _cls: Any,
        driver: Any,
        entity_node_uuid: str,
    ) -> list[Any]:
        return await driver.episode_node_ops.get_by_entity_node_uuid(driver, entity_node_uuid)

    async def community_node_load_name_embedding(
        self,
        node: Any,
        driver: Any,
    ) -> None:
        await driver.community_node_ops.load_name_embedding(driver, node)

    async def edge_get_between_nodes(
        self,
        _cls: Any,
        driver: Any,
        source_node_uuid: str,
        target_node_uuid: str,
    ) -> list[Any]:
        return await driver.entity_edge_ops.get_between_nodes(
            driver, source_node_uuid, target_node_uuid
        )

    async def edge_get_by_node_uuid(
        self,
        _cls: Any,
        driver: Any,
        node_uuid: str,
    ) -> list[Any]:
        return await driver.entity_edge_ops.get_by_node_uuid(driver, node_uuid)


class OracleLegacySearchInterface(SearchInterface):
    """Route legacy search helpers to the Oracle search operation family."""

    async def edge_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        return await driver.search_ops.edge_fulltext_search(
            driver,
            query,
            search_filter,
            group_ids=group_ids,
            limit=limit,
        )

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
        return await driver.search_ops.edge_similarity_search(
            driver,
            search_vector,
            source_node_uuid,
            target_node_uuid,
            search_filter,
            group_ids=group_ids,
            limit=limit,
            min_score=min_score,
        )

    async def node_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        return await driver.search_ops.node_fulltext_search(
            driver,
            query,
            search_filter,
            group_ids=group_ids,
            limit=limit,
        )

    async def node_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[Any]:
        return await driver.search_ops.node_similarity_search(
            driver,
            search_vector,
            search_filter,
            group_ids=group_ids,
            limit=limit,
            min_score=min_score,
        )

    async def episode_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        return await driver.search_ops.episode_fulltext_search(
            driver,
            query,
            search_filter,
            group_ids=group_ids,
            limit=limit,
        )

    async def edge_bfs_search(
        self,
        driver: Any,
        bfs_origin_node_uuids: list[str] | None,
        bfs_max_depth: int,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        return await driver.search_ops.edge_bfs_search(
            driver,
            bfs_origin_node_uuids or [],
            bfs_max_depth,
            search_filter,
            group_ids=group_ids,
            limit=limit,
        )

    async def node_bfs_search(
        self,
        driver: Any,
        bfs_origin_node_uuids: list[str] | None,
        search_filter: Any,
        bfs_max_depth: int,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        return await driver.search_ops.node_bfs_search(
            driver,
            bfs_origin_node_uuids or [],
            search_filter,
            bfs_max_depth,
            group_ids=group_ids,
            limit=limit,
        )

    async def community_fulltext_search(
        self,
        driver: Any,
        query: str,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        return await driver.search_ops.community_fulltext_search(
            driver,
            query,
            group_ids=group_ids,
            limit=limit,
        )

    async def community_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.6,
    ) -> list[Any]:
        return await driver.search_ops.community_similarity_search(
            driver,
            search_vector,
            group_ids=group_ids,
            limit=limit,
            min_score=min_score,
        )

    async def get_embeddings_for_communities(
        self,
        driver: Any,
        communities: list[Any],
    ) -> dict[str, list[float]]:
        return await driver.search_ops.get_embeddings_for_communities(driver, communities)

    async def node_distance_reranker(
        self,
        driver: Any,
        node_uuids: list[str],
        center_node_uuid: str,
        min_score: float = 0,
    ) -> tuple[list[str], list[float]]:
        return await driver.search_ops.node_distance_reranker(
            driver,
            node_uuids,
            center_node_uuid,
            min_score=min_score,
        )

    async def episode_mentions_reranker(
        self,
        driver: Any,
        node_uuids: list[list[str]],
        min_score: float = 0,
    ) -> tuple[list[str], list[float]]:
        return await driver.search_ops.episode_mentions_reranker(
            driver,
            node_uuids,
            min_score=min_score,
        )

    def build_node_search_filters(self, search_filters: Any) -> Any:
        return OracleSearchOperations().build_node_search_filters(search_filters)

    def build_edge_search_filters(self, search_filters: Any) -> Any:
        return OracleSearchOperations().build_edge_search_filters(search_filters)
