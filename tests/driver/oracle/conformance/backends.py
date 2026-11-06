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

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.driver.oracle_driver import Oracle26aiConfig, Oracle26aiDriver
from graphiti_core.driver.query_executor import Transaction
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.namespaces.edges import EdgeNamespace
from graphiti_core.namespaces.nodes import NodeNamespace
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_filters import SearchFilters


class DeterministicEmbedder(EmbedderClient):
    async def create(
        self,
        input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]],
    ) -> list[float]:
        encoded = str(input_data).encode('utf-8')
        vector = [0.0] * 1024
        vector[sum(encoded) % len(vector)] = 1.0
        return vector

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return [await self.create(value) for value in input_data_list]


@dataclass(slots=True)
class _State:
    nodes: dict[str, EntityNode]
    edges: dict[str, EntityEdge]


class _MemoryTransaction(Transaction):
    def __init__(self, state: _State) -> None:
        self.state = _State(nodes=deepcopy(state.nodes), edges=deepcopy(state.edges))

    async def run(self, query: str, **kwargs: Any) -> Any:
        raise AssertionError(f'memory transaction received SQL unexpectedly: {query}')


def _driver(executor: Any) -> MemoryConformanceDriver:
    if not isinstance(executor, MemoryConformanceDriver):
        raise TypeError('memory operations require MemoryConformanceDriver')
    return executor


def _state(executor: Any, tx: Transaction | None = None) -> _State:
    if tx is None:
        return _driver(executor)._state
    if not isinstance(tx, _MemoryTransaction):
        raise TypeError('memory operations require _MemoryTransaction')
    return tx.state


class _EntityNodeOps:
    async def save(
        self,
        executor: Any,
        node: EntityNode,
        tx: Transaction | None = None,
    ) -> None:
        _state(executor, tx).nodes[node.uuid] = deepcopy(node)

    async def save_bulk(
        self,
        executor: Any,
        nodes: list[EntityNode],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        del batch_size
        for node in nodes:
            await self.save(executor, node, tx)

    async def delete(self, executor: Any, node: EntityNode, tx: Transaction | None = None) -> None:
        state = _state(executor, tx)
        state.nodes.pop(node.uuid, None)
        state.edges = {
            uuid: edge
            for uuid, edge in state.edges.items()
            if node.uuid not in {edge.source_node_uuid, edge.target_node_uuid}
        }

    async def delete_by_group_id(
        self,
        executor: Any,
        group_id: str,
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        del batch_size
        state = _state(executor, tx)
        state.nodes = {
            uuid: node for uuid, node in state.nodes.items() if node.group_id != group_id
        }
        state.edges = {
            uuid: edge for uuid, edge in state.edges.items() if edge.group_id != group_id
        }

    async def delete_by_uuids(
        self,
        executor: Any,
        uuids: list[str],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        del batch_size
        for uuid in uuids:
            node = _state(executor, tx).nodes.get(uuid)
            if node:
                await self.delete(executor, node, tx)

    async def get_by_uuid(self, executor: Any, uuid: str) -> EntityNode:
        return deepcopy(_driver(executor)._state.nodes[uuid])

    async def get_by_uuids(self, executor: Any, uuids: list[str]) -> list[EntityNode]:
        state = _driver(executor)._state
        return [deepcopy(state.nodes[uuid]) for uuid in uuids if uuid in state.nodes]

    async def get_by_group_ids(
        self,
        executor: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EntityNode]:
        groups = set(group_ids)
        nodes = [
            deepcopy(node)
            for node in _driver(executor)._state.nodes.values()
            if node.group_id in groups and (uuid_cursor is None or node.uuid < uuid_cursor)
        ]
        nodes.sort(key=lambda item: item.uuid, reverse=True)
        return nodes[:limit] if limit is not None else nodes

    async def load_embeddings(self, executor: Any, node: EntityNode) -> None:
        node.name_embedding = deepcopy(_driver(executor)._state.nodes[node.uuid].name_embedding)

    async def load_embeddings_bulk(
        self, executor: Any, nodes: list[EntityNode], batch_size: int = 100
    ) -> None:
        del batch_size
        for node in nodes:
            await self.load_embeddings(executor, node)


class _EntityEdgeOps:
    async def save(
        self,
        executor: Any,
        edge: EntityEdge,
        tx: Transaction | None = None,
    ) -> None:
        state = _state(executor, tx)
        source = state.nodes.get(edge.source_node_uuid)
        target = state.nodes.get(edge.target_node_uuid)
        if source is None or target is None:
            raise ValueError('missing edge endpoint')
        if {source.group_id, target.group_id, edge.group_id} != {edge.group_id}:
            raise ValueError('cross-group edge')
        state.edges[edge.uuid] = deepcopy(edge)

    async def save_bulk(
        self,
        executor: Any,
        edges: list[EntityEdge],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        del batch_size
        for edge in edges:
            await self.save(executor, edge, tx)

    async def delete(self, executor: Any, edge: EntityEdge, tx: Transaction | None = None) -> None:
        _state(executor, tx).edges.pop(edge.uuid, None)

    async def delete_by_uuids(
        self, executor: Any, uuids: list[str], tx: Transaction | None = None
    ) -> None:
        for uuid in uuids:
            _state(executor, tx).edges.pop(uuid, None)

    async def get_by_uuid(self, executor: Any, uuid: str) -> EntityEdge:
        return deepcopy(_driver(executor)._state.edges[uuid])

    async def get_by_uuids(self, executor: Any, uuids: list[str]) -> list[EntityEdge]:
        state = _driver(executor)._state
        return [deepcopy(state.edges[uuid]) for uuid in uuids if uuid in state.edges]

    async def get_by_group_ids(
        self,
        executor: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EntityEdge]:
        groups = set(group_ids)
        edges = [
            deepcopy(edge)
            for edge in _driver(executor)._state.edges.values()
            if edge.group_id in groups and (uuid_cursor is None or edge.uuid < uuid_cursor)
        ]
        edges.sort(key=lambda item: item.uuid, reverse=True)
        return edges[:limit] if limit is not None else edges

    async def get_between_nodes(
        self, executor: Any, source_node_uuid: str, target_node_uuid: str
    ) -> list[EntityEdge]:
        return [
            deepcopy(edge)
            for edge in _driver(executor)._state.edges.values()
            if edge.source_node_uuid == source_node_uuid
            and edge.target_node_uuid == target_node_uuid
        ]

    async def get_by_node_uuid(self, executor: Any, node_uuid: str) -> list[EntityEdge]:
        return [
            deepcopy(edge)
            for edge in _driver(executor)._state.edges.values()
            if node_uuid in {edge.source_node_uuid, edge.target_node_uuid}
        ]

    async def load_embeddings(self, executor: Any, edge: EntityEdge) -> None:
        edge.fact_embedding = deepcopy(_driver(executor)._state.edges[edge.uuid].fact_embedding)

    async def load_embeddings_bulk(
        self, executor: Any, edges: list[EntityEdge], batch_size: int = 100
    ) -> None:
        del batch_size
        for edge in edges:
            await self.load_embeddings(executor, edge)


class _GraphOps:
    async def clear_data(self, executor: Any, group_ids: list[str] | None = None) -> None:
        state = _driver(executor)._state
        if group_ids is None:
            state.nodes.clear()
            state.edges.clear()
            return
        groups = set(group_ids)
        state.edges = {
            uuid: edge for uuid, edge in state.edges.items() if edge.group_id not in groups
        }
        state.nodes = {
            uuid: node for uuid, node in state.nodes.items() if node.group_id not in groups
        }


class _SearchOps:
    async def node_bfs_search(
        self,
        executor: Any,
        origin_uuids: list[str],
        search_filter: SearchFilters,
        max_depth: int,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityNode]:
        del search_filter
        state = _driver(executor)._state
        allowed_groups = set(group_ids) if group_ids is not None else None
        frontier = set(origin_uuids)
        visited = set(origin_uuids)
        reached: set[str] = set()
        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for edge in state.edges.values():
                if edge.source_node_uuid not in frontier:
                    continue
                if allowed_groups is not None and edge.group_id not in allowed_groups:
                    continue
                target = state.nodes.get(edge.target_node_uuid)
                if target is None or target.uuid in visited:
                    continue
                visited.add(target.uuid)
                reached.add(target.uuid)
                next_frontier.add(target.uuid)
            frontier = next_frontier
            if not frontier:
                break
        return [deepcopy(state.nodes[uuid]) for uuid in sorted(reached)[:limit]]


class MemoryConformanceDriver(GraphDriver):
    provider = GraphProvider.ORACLE

    def __init__(self) -> None:
        self._state = _State(nodes={}, edges={})
        self._entity_node_ops = _EntityNodeOps()
        self._entity_edge_ops = _EntityEdgeOps()
        self._graph_ops = _GraphOps()
        self._search_ops = _SearchOps()

    async def execute_query(self, cypher_query_: str, **kwargs: Any) -> Any:
        raise AssertionError(f'memory driver received SQL unexpectedly: {cypher_query_}')

    def session(self, database: str | None = None) -> GraphDriverSession:
        raise NotImplementedError

    async def close(self) -> None:
        return None

    async def delete_all_indexes(self) -> None:
        return None

    async def build_indices_and_constraints(self, delete_existing: bool = False) -> None:
        return None

    @property
    def entity_node_ops(self) -> Any:
        return self._entity_node_ops

    @property
    def entity_edge_ops(self) -> Any:
        return self._entity_edge_ops

    @property
    def graph_ops(self) -> Any:
        return self._graph_ops

    @property
    def search_ops(self) -> Any:
        return self._search_ops

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Transaction]:
        transaction = _MemoryTransaction(self._state)
        try:
            yield transaction
        except BaseException:
            raise
        else:
            self._state = transaction.state


@dataclass(slots=True)
class ConformanceSurface:
    driver: GraphDriver
    nodes: NodeNamespace
    edges: EdgeNamespace


def surface_for_driver(driver: GraphDriver) -> ConformanceSurface:
    embedder = DeterministicEmbedder()
    return ConformanceSurface(
        driver=driver,
        nodes=NodeNamespace(driver, embedder),
        edges=EdgeNamespace(driver, embedder),
    )


def reference_surface() -> ConformanceSurface:
    return surface_for_driver(MemoryConformanceDriver())


async def oracle_surface_from_env() -> ConformanceSurface:
    password = os.environ.get('ORACLE_PASSWORD')
    if not password:
        raise RuntimeError('ORACLE_PASSWORD is required for live Oracle conformance')
    root = Path(__file__).parents[4]
    config_dir = os.environ.get('ORACLE_CONFIG_DIR')
    wallet_dir = os.environ.get('ORACLE_WALLET_DIR')
    config = Oracle26aiConfig.model_validate(
        {
            'dsn': os.environ.get('ORACLE_DSN', 'myatp_low'),
            'user': os.environ.get('ORACLE_USER', 'admin'),
            'password': password,
            'config_dir': str(root / config_dir) if config_dir else None,
            'wallet_location': str(root / wallet_dir) if wallet_dir else None,
            'wallet_password': os.environ.get('ORACLE_WALLET_PASSWORD') or None,
        }
    )
    driver = Oracle26aiDriver(config)
    await driver.build_indices_and_constraints()
    return surface_for_driver(driver)


async def falkor_surface_from_env() -> ConformanceSurface:
    try:
        from graphiti_core.driver.falkordb_driver import FalkorDriver
    except ImportError as error:
        raise RuntimeError(
            'install graphiti-core[falkordb] for live FalkorDB conformance'
        ) from error
    driver = FalkorDriver(
        host=os.environ.get('FALKORDB_HOST', '127.0.0.1'),
        port=int(os.environ.get('FALKORDB_PORT', '6379')),
        username=os.environ.get('FALKORDB_USERNAME') or None,
        password=os.environ.get('FALKORDB_PASSWORD') or None,
        database=os.environ.get('FALKORDB_DATABASE', 'graphiti_conformance'),
    )
    await driver.build_indices_and_constraints()
    return surface_for_driver(driver)


async def close_surface(surface: ConformanceSurface) -> None:
    await surface.driver.close()
