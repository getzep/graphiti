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

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.driver.operations.community_node_ops import CommunityNodeOperations
from graphiti_core.driver.operations.entity_node_ops import EntityNodeOperations
from graphiti_core.driver.operations.episode_node_ops import EpisodeNodeOperations
from graphiti_core.driver.operations.saga_node_ops import SagaNodeOperations
from graphiti_core.driver.query_executor import Transaction
from graphiti_core.embedder import EmbedderClient
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode, SagaNode


class EntityNodeNamespace:
    """Namespace for entity node operations. Accessed as ``graphiti.nodes.entity``."""

    def __init__(
        self,
        driver: GraphDriver,
        ops: EntityNodeOperations,
        embedder: EmbedderClient,
    ):
        self._driver = driver
        self._ops = ops
        self._embedder = embedder

    async def save(
        self,
        node: EntityNode,
        tx: Transaction | None = None,
    ) -> EntityNode:
        await node.generate_name_embedding(self._embedder)
        await self._ops.save(self._driver, node, tx=tx)
        return node

    async def save_bulk(
        self,
        nodes: list[EntityNode],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        await self._ops.save_bulk(self._driver, nodes, tx=tx, batch_size=batch_size)

    async def delete(
        self,
        node: EntityNode,
        tx: Transaction | None = None,
    ) -> None:
        await self._ops.delete(self._driver, node, tx=tx)

    async def delete_by_group_id(
        self,
        group_id: str,
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        await self._ops.delete_by_group_id(self._driver, group_id, tx=tx, batch_size=batch_size)

    async def delete_by_uuids(
        self,
        uuids: list[str],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        await self._ops.delete_by_uuids(self._driver, uuids, tx=tx, batch_size=batch_size)

    async def get_by_uuid(self, uuid: str) -> EntityNode:
        return await self._ops.get_by_uuid(self._driver, uuid)

    async def get_by_uuids(self, uuids: list[str]) -> list[EntityNode]:
        return await self._ops.get_by_uuids(self._driver, uuids)

    async def get_by_group_ids(
        self,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EntityNode]:
        return await self._ops.get_by_group_ids(self._driver, group_ids, limit, uuid_cursor)

    async def load_embeddings(self, node: EntityNode) -> None:
        await self._ops.load_embeddings(self._driver, node)

    async def load_embeddings_bulk(
        self,
        nodes: list[EntityNode],
        batch_size: int = 100,
    ) -> None:
        await self._ops.load_embeddings_bulk(self._driver, nodes, batch_size)


class EpisodeNodeNamespace:
    """Namespace for episode node operations. Accessed as ``graphiti.nodes.episode``."""

    def __init__(self, driver: GraphDriver, ops: EpisodeNodeOperations):
        self._driver = driver
        self._ops = ops

    async def save(
        self,
        node: EpisodicNode,
        tx: Transaction | None = None,
    ) -> EpisodicNode:
        await self._ops.save(self._driver, node, tx=tx)
        return node

    async def save_bulk(
        self,
        nodes: list[EpisodicNode],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        await self._ops.save_bulk(self._driver, nodes, tx=tx, batch_size=batch_size)

    async def delete(
        self,
        node: EpisodicNode,
        tx: Transaction | None = None,
    ) -> None:
        await self._ops.delete(self._driver, node, tx=tx)

    async def delete_by_group_id(
        self,
        group_id: str,
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        await self._ops.delete_by_group_id(self._driver, group_id, tx=tx, batch_size=batch_size)

    async def delete_by_uuids(
        self,
        uuids: list[str],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        await self._ops.delete_by_uuids(self._driver, uuids, tx=tx, batch_size=batch_size)

    async def get_by_uuid(self, uuid: str) -> EpisodicNode:
        return await self._ops.get_by_uuid(self._driver, uuid)

    async def get_by_uuids(self, uuids: list[str]) -> list[EpisodicNode]:
        return await self._ops.get_by_uuids(self._driver, uuids)

    async def get_by_group_ids(
        self,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EpisodicNode]:
        return await self._ops.get_by_group_ids(self._driver, group_ids, limit, uuid_cursor)

    async def get_by_entity_node_uuid(
        self,
        entity_node_uuid: str,
    ) -> list[EpisodicNode]:
        return await self._ops.get_by_entity_node_uuid(self._driver, entity_node_uuid)

    async def retrieve_episodes(
        self,
        reference_time: datetime,
        last_n: int = 3,
        group_ids: list[str] | None = None,
        source: str | None = None,
        saga: str | None = None,
    ) -> list[EpisodicNode]:
        return await self._ops.retrieve_episodes(
            self._driver, reference_time, last_n, group_ids, source, saga
        )


class CommunityNodeNamespace:
    """Namespace for community node operations. Accessed as ``graphiti.nodes.community``."""

    def __init__(
        self,
        driver: GraphDriver,
        ops: CommunityNodeOperations,
        embedder: EmbedderClient,
    ):
        self._driver = driver
        self._ops = ops
        self._embedder = embedder

    async def save(
        self,
        node: CommunityNode,
        tx: Transaction | None = None,
    ) -> CommunityNode:
        await node.generate_name_embedding(self._embedder)
        await self._ops.save(self._driver, node, tx=tx)
        return node

    async def save_bulk(
        self,
        nodes: list[CommunityNode],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        await self._ops.save_bulk(self._driver, nodes, tx=tx, batch_size=batch_size)

    async def delete(
        self,
        node: CommunityNode,
        tx: Transaction | None = None,
    ) -> None:
        await self._ops.delete(self._driver, node, tx=tx)

    async def delete_by_group_id(
        self,
        group_id: str,
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        await self._ops.delete_by_group_id(self._driver, group_id, tx=tx, batch_size=batch_size)

    async def delete_by_uuids(
        self,
        uuids: list[str],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        await self._ops.delete_by_uuids(self._driver, uuids, tx=tx, batch_size=batch_size)

    async def get_by_uuid(self, uuid: str) -> CommunityNode:
        return await self._ops.get_by_uuid(self._driver, uuid)

    async def get_by_uuids(self, uuids: list[str]) -> list[CommunityNode]:
        return await self._ops.get_by_uuids(self._driver, uuids)

    async def get_by_group_ids(
        self,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[CommunityNode]:
        return await self._ops.get_by_group_ids(self._driver, group_ids, limit, uuid_cursor)

    async def load_name_embedding(self, node: CommunityNode) -> None:
        await self._ops.load_name_embedding(self._driver, node)


class SagaNodeNamespace:
    """Namespace for saga node operations. Accessed as ``graphiti.nodes.saga``."""

    def __init__(self, driver: GraphDriver, ops: SagaNodeOperations):
        self._driver = driver
        self._ops = ops

    async def save(
        self,
        node: SagaNode,
        tx: Transaction | None = None,
    ) -> SagaNode:
        await self._ops.save(self._driver, node, tx=tx)
        return node

    async def save_bulk(
        self,
        nodes: list[SagaNode],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        await self._ops.save_bulk(self._driver, nodes, tx=tx, batch_size=batch_size)

    async def delete(
        self,
        node: SagaNode,
        tx: Transaction | None = None,
    ) -> None:
        await self._ops.delete(self._driver, node, tx=tx)

    async def delete_by_group_id(
        self,
        group_id: str,
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        await self._ops.delete_by_group_id(self._driver, group_id, tx=tx, batch_size=batch_size)

    async def delete_by_uuids(
        self,
        uuids: list[str],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        await self._ops.delete_by_uuids(self._driver, uuids, tx=tx, batch_size=batch_size)

    async def get_by_uuid(self, uuid: str) -> SagaNode:
        return await self._ops.get_by_uuid(self._driver, uuid)

    async def get_by_uuids(self, uuids: list[str]) -> list[SagaNode]:
        return await self._ops.get_by_uuids(self._driver, uuids)

    async def get_by_group_ids(
        self,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[SagaNode]:
        return await self._ops.get_by_group_ids(self._driver, group_ids, limit, uuid_cursor)


class NodeNamespace:
    """Namespace for all node operations. Accessed as ``graphiti.nodes``.

    Sub-namespaces are set only when the driver provides the corresponding
    operations implementation.  Accessing an unset attribute raises
    ``NotImplementedError`` with a clear message.
    """

    entity: EntityNodeNamespace
    episode: EpisodeNodeNamespace
    community: CommunityNodeNamespace
    saga: SagaNodeNamespace

    _driver_name: str

    def __init__(self, driver: GraphDriver, embedder: EmbedderClient):
        self._driver_name = type(driver).__name__

        entity_node_ops = driver.entity_node_ops
        if entity_node_ops is not None:
            self.entity = EntityNodeNamespace(driver, entity_node_ops, embedder)

        episode_node_ops = driver.episode_node_ops
        if episode_node_ops is not None:
            self.episode = EpisodeNodeNamespace(driver, episode_node_ops)

        community_node_ops = driver.community_node_ops
        if community_node_ops is not None:
            self.community = CommunityNodeNamespace(driver, community_node_ops, embedder)

        saga_node_ops = driver.saga_node_ops
        if saga_node_ops is not None:
            self.saga = SagaNodeNamespace(driver, saga_node_ops)

    def __getattr__(self, name: str) -> object:
        if name in ('entity', 'episode', 'community', 'saga'):
            raise NotImplementedError(f'{self._driver_name} does not implement {name}_node_ops')
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
