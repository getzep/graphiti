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

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.driver.operations.community_edge_ops import CommunityEdgeOperations
from graphiti_core.driver.operations.entity_edge_ops import EntityEdgeOperations
from graphiti_core.driver.operations.episodic_edge_ops import EpisodicEdgeOperations
from graphiti_core.driver.operations.has_episode_edge_ops import HasEpisodeEdgeOperations
from graphiti_core.driver.operations.next_episode_edge_ops import NextEpisodeEdgeOperations
from graphiti_core.driver.query_executor import Transaction
from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    HasEpisodeEdge,
    NextEpisodeEdge,
)
from graphiti_core.embedder import EmbedderClient


class EntityEdgeNamespace:
    """Namespace for entity edge operations. Accessed as ``graphiti.edges.entity``."""

    def __init__(
        self,
        driver: GraphDriver,
        ops: EntityEdgeOperations,
        embedder: EmbedderClient,
    ):
        self._driver = driver
        self._ops = ops
        self._embedder = embedder

    async def save(
        self,
        edge: EntityEdge,
        tx: Transaction | None = None,
    ) -> EntityEdge:
        await edge.generate_embedding(self._embedder)
        await self._ops.save(self._driver, edge, tx=tx)
        return edge

    async def save_bulk(
        self,
        edges: list[EntityEdge],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        await self._ops.save_bulk(self._driver, edges, tx=tx, batch_size=batch_size)

    async def delete(
        self,
        edge: EntityEdge,
        tx: Transaction | None = None,
    ) -> None:
        await self._ops.delete(self._driver, edge, tx=tx)

    async def delete_by_uuids(
        self,
        uuids: list[str],
        tx: Transaction | None = None,
    ) -> None:
        await self._ops.delete_by_uuids(self._driver, uuids, tx=tx)

    async def get_by_uuid(self, uuid: str) -> EntityEdge:
        return await self._ops.get_by_uuid(self._driver, uuid)

    async def get_by_uuids(self, uuids: list[str]) -> list[EntityEdge]:
        return await self._ops.get_by_uuids(self._driver, uuids)

    async def get_by_group_ids(
        self,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EntityEdge]:
        return await self._ops.get_by_group_ids(self._driver, group_ids, limit, uuid_cursor)

    async def get_between_nodes(
        self,
        source_node_uuid: str,
        target_node_uuid: str,
    ) -> list[EntityEdge]:
        return await self._ops.get_between_nodes(self._driver, source_node_uuid, target_node_uuid)

    async def get_by_node_uuid(self, node_uuid: str) -> list[EntityEdge]:
        return await self._ops.get_by_node_uuid(self._driver, node_uuid)

    async def load_embeddings(self, edge: EntityEdge) -> None:
        await self._ops.load_embeddings(self._driver, edge)

    async def load_embeddings_bulk(
        self,
        edges: list[EntityEdge],
        batch_size: int = 100,
    ) -> None:
        await self._ops.load_embeddings_bulk(self._driver, edges, batch_size)


class EpisodicEdgeNamespace:
    """Namespace for episodic edge operations. Accessed as ``graphiti.edges.episodic``."""

    def __init__(self, driver: GraphDriver, ops: EpisodicEdgeOperations):
        self._driver = driver
        self._ops = ops

    async def save(
        self,
        edge: EpisodicEdge,
        tx: Transaction | None = None,
    ) -> EpisodicEdge:
        await self._ops.save(self._driver, edge, tx=tx)
        return edge

    async def save_bulk(
        self,
        edges: list[EpisodicEdge],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        await self._ops.save_bulk(self._driver, edges, tx=tx, batch_size=batch_size)

    async def delete(
        self,
        edge: EpisodicEdge,
        tx: Transaction | None = None,
    ) -> None:
        await self._ops.delete(self._driver, edge, tx=tx)

    async def delete_by_uuids(
        self,
        uuids: list[str],
        tx: Transaction | None = None,
    ) -> None:
        await self._ops.delete_by_uuids(self._driver, uuids, tx=tx)

    async def get_by_uuid(self, uuid: str) -> EpisodicEdge:
        return await self._ops.get_by_uuid(self._driver, uuid)

    async def get_by_uuids(self, uuids: list[str]) -> list[EpisodicEdge]:
        return await self._ops.get_by_uuids(self._driver, uuids)

    async def get_by_group_ids(
        self,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EpisodicEdge]:
        return await self._ops.get_by_group_ids(self._driver, group_ids, limit, uuid_cursor)


class CommunityEdgeNamespace:
    """Namespace for community edge operations. Accessed as ``graphiti.edges.community``."""

    def __init__(self, driver: GraphDriver, ops: CommunityEdgeOperations):
        self._driver = driver
        self._ops = ops

    async def save(
        self,
        edge: CommunityEdge,
        tx: Transaction | None = None,
    ) -> CommunityEdge:
        await self._ops.save(self._driver, edge, tx=tx)
        return edge

    async def delete(
        self,
        edge: CommunityEdge,
        tx: Transaction | None = None,
    ) -> None:
        await self._ops.delete(self._driver, edge, tx=tx)

    async def delete_by_uuids(
        self,
        uuids: list[str],
        tx: Transaction | None = None,
    ) -> None:
        await self._ops.delete_by_uuids(self._driver, uuids, tx=tx)

    async def get_by_uuid(self, uuid: str) -> CommunityEdge:
        return await self._ops.get_by_uuid(self._driver, uuid)

    async def get_by_uuids(self, uuids: list[str]) -> list[CommunityEdge]:
        return await self._ops.get_by_uuids(self._driver, uuids)

    async def get_by_group_ids(
        self,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[CommunityEdge]:
        return await self._ops.get_by_group_ids(self._driver, group_ids, limit, uuid_cursor)


class HasEpisodeEdgeNamespace:
    """Namespace for has_episode edge operations. Accessed as ``graphiti.edges.has_episode``."""

    def __init__(self, driver: GraphDriver, ops: HasEpisodeEdgeOperations):
        self._driver = driver
        self._ops = ops

    async def save(
        self,
        edge: HasEpisodeEdge,
        tx: Transaction | None = None,
    ) -> HasEpisodeEdge:
        await self._ops.save(self._driver, edge, tx=tx)
        return edge

    async def save_bulk(
        self,
        edges: list[HasEpisodeEdge],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        await self._ops.save_bulk(self._driver, edges, tx=tx, batch_size=batch_size)

    async def delete(
        self,
        edge: HasEpisodeEdge,
        tx: Transaction | None = None,
    ) -> None:
        await self._ops.delete(self._driver, edge, tx=tx)

    async def delete_by_uuids(
        self,
        uuids: list[str],
        tx: Transaction | None = None,
    ) -> None:
        await self._ops.delete_by_uuids(self._driver, uuids, tx=tx)

    async def get_by_uuid(self, uuid: str) -> HasEpisodeEdge:
        return await self._ops.get_by_uuid(self._driver, uuid)

    async def get_by_uuids(self, uuids: list[str]) -> list[HasEpisodeEdge]:
        return await self._ops.get_by_uuids(self._driver, uuids)

    async def get_by_group_ids(
        self,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[HasEpisodeEdge]:
        return await self._ops.get_by_group_ids(self._driver, group_ids, limit, uuid_cursor)


class NextEpisodeEdgeNamespace:
    """Namespace for next_episode edge operations. Accessed as ``graphiti.edges.next_episode``."""

    def __init__(self, driver: GraphDriver, ops: NextEpisodeEdgeOperations):
        self._driver = driver
        self._ops = ops

    async def save(
        self,
        edge: NextEpisodeEdge,
        tx: Transaction | None = None,
    ) -> NextEpisodeEdge:
        await self._ops.save(self._driver, edge, tx=tx)
        return edge

    async def save_bulk(
        self,
        edges: list[NextEpisodeEdge],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        await self._ops.save_bulk(self._driver, edges, tx=tx, batch_size=batch_size)

    async def delete(
        self,
        edge: NextEpisodeEdge,
        tx: Transaction | None = None,
    ) -> None:
        await self._ops.delete(self._driver, edge, tx=tx)

    async def delete_by_uuids(
        self,
        uuids: list[str],
        tx: Transaction | None = None,
    ) -> None:
        await self._ops.delete_by_uuids(self._driver, uuids, tx=tx)

    async def get_by_uuid(self, uuid: str) -> NextEpisodeEdge:
        return await self._ops.get_by_uuid(self._driver, uuid)

    async def get_by_uuids(self, uuids: list[str]) -> list[NextEpisodeEdge]:
        return await self._ops.get_by_uuids(self._driver, uuids)

    async def get_by_group_ids(
        self,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[NextEpisodeEdge]:
        return await self._ops.get_by_group_ids(self._driver, group_ids, limit, uuid_cursor)


class EdgeNamespace:
    """Namespace for all edge operations. Accessed as ``graphiti.edges``.

    Sub-namespaces are set only when the driver provides the corresponding
    operations implementation.  Accessing an unset attribute raises
    ``NotImplementedError`` with a clear message.
    """

    entity: EntityEdgeNamespace
    episodic: EpisodicEdgeNamespace
    community: CommunityEdgeNamespace
    has_episode: HasEpisodeEdgeNamespace
    next_episode: NextEpisodeEdgeNamespace

    _driver_name: str

    def __init__(self, driver: GraphDriver, embedder: EmbedderClient):
        self._driver_name = type(driver).__name__

        entity_edge_ops = driver.entity_edge_ops
        if entity_edge_ops is not None:
            self.entity = EntityEdgeNamespace(driver, entity_edge_ops, embedder)

        episodic_edge_ops = driver.episodic_edge_ops
        if episodic_edge_ops is not None:
            self.episodic = EpisodicEdgeNamespace(driver, episodic_edge_ops)

        community_edge_ops = driver.community_edge_ops
        if community_edge_ops is not None:
            self.community = CommunityEdgeNamespace(driver, community_edge_ops)

        has_episode_edge_ops = driver.has_episode_edge_ops
        if has_episode_edge_ops is not None:
            self.has_episode = HasEpisodeEdgeNamespace(driver, has_episode_edge_ops)

        next_episode_edge_ops = driver.next_episode_edge_ops
        if next_episode_edge_ops is not None:
            self.next_episode = NextEpisodeEdgeNamespace(driver, next_episode_edge_ops)

    def __getattr__(self, name: str) -> object:
        if name in ('entity', 'episodic', 'community', 'has_episode', 'next_episode'):
            raise NotImplementedError(f'{self._driver_name} does not implement {name}_edge_ops')
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
