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

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from os import PathLike
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import relate as _relate  # noqa: F401  # pyright: ignore[reportMissingImports]
else:
    try:
        import relate as _relate  # noqa: F401
    except ImportError:
        raise ImportError(
            'relate-db is required for RelateDriver. '
            'Install it with: pip install graphiti-core[relate]'
        ) from None

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.driver.operations.community_edge_ops import CommunityEdgeOperations
from graphiti_core.driver.operations.community_node_ops import CommunityNodeOperations
from graphiti_core.driver.operations.entity_edge_ops import EntityEdgeOperations
from graphiti_core.driver.operations.entity_node_ops import EntityNodeOperations
from graphiti_core.driver.operations.episode_node_ops import EpisodeNodeOperations
from graphiti_core.driver.operations.episodic_edge_ops import EpisodicEdgeOperations
from graphiti_core.driver.operations.graph_ops import GraphMaintenanceOperations
from graphiti_core.driver.operations.has_episode_edge_ops import HasEpisodeEdgeOperations
from graphiti_core.driver.operations.next_episode_edge_ops import NextEpisodeEdgeOperations
from graphiti_core.driver.operations.saga_node_ops import SagaNodeOperations
from graphiti_core.driver.operations.search_ops import SearchOperations
from graphiti_core.driver.query_executor import Transaction
from graphiti_core.driver.relate import (
    Node,
    Path,
    QueryResult,
    RelateStore,
    RelateTransaction,
    Relationship,
    StoreClosedError,
    Temporal,
    Vector,
)
from graphiti_core.driver.relate.graph_operations_interface import (
    RelateGraphOperationsInterface,
)
from graphiti_core.driver.relate.operations import (
    RelateCommunityEdgeOperations,
    RelateCommunityNodeOperations,
    RelateEntityEdgeOperations,
    RelateEntityNodeOperations,
    RelateEpisodeNodeOperations,
    RelateEpisodicEdgeOperations,
    RelateGraphMaintenanceOperations,
    RelateHasEpisodeEdgeOperations,
    RelateNextEpisodeEdgeOperations,
    RelateSagaNodeOperations,
    RelateSearchOperations,
)
from graphiti_core.driver.relate.search_interface import RelateSearchInterface
from graphiti_core.driver.relate.translation import translate_relate_query
from graphiti_core.utils.datetime_utils import ensure_utc

logger = logging.getLogger(__name__)


def _prepare_param_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, Mapping):
        return {key: _prepare_param_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_prepare_param_value(item) for item in value]
    return value


def _extract_params(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(kwargs)
    raw.pop('routing_', None)
    raw.pop('database_', None)
    nested_params = raw.pop('params', None)

    merged: dict[str, Any] = {}
    if isinstance(nested_params, Mapping):
        merged.update(nested_params)
    merged.update(raw)

    return {key: _prepare_param_value(value) for key, value in merged.items()}


def _convert_result_value(value: Any) -> Any:
    if isinstance(value, Vector):
        return list(value.values)
    if isinstance(value, Temporal):
        return value.text
    if isinstance(value, Node):
        return {
            'id': value.id,
            'labels': list(value.labels),
            'properties': _convert_result_value(value.properties),
        }
    if isinstance(value, Relationship):
        return {
            'id': value.id,
            'type': value.type,
            'start_node_id': value.start_node_id,
            'end_node_id': value.end_node_id,
            'properties': _convert_result_value(value.properties),
        }
    if isinstance(value, Path):
        return {
            'nodes': [_convert_result_value(node) for node in value.nodes],
            'relationships': [_convert_result_value(rel) for rel in value.relationships],
        }
    if isinstance(value, (tuple, list)):
        return [_convert_result_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _convert_result_value(item) for key, item in value.items()}
    return value


def _format_query_result(result: QueryResult) -> tuple[list[dict[str, Any]], list[str], None]:
    header = [col.name for col in result.columns]
    records = [{col: _convert_result_value(row[col]) for col in header} for row in result.rows]
    return records, header, None


class RelateDriverSession(GraphDriverSession):
    provider = GraphProvider.RELATE

    def __init__(self, driver: RelateDriver):
        self._driver = driver
        self._active_tx: RelateTransaction | None = None
        self.query_timeout_ms: int | None = None

    async def __aenter__(self) -> RelateDriverSession:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        # The underlying RelateStore lifecycle is owned by RelateDriver.
        pass

    async def execute_write(self, func, *args, **kwargs):
        store = await self._driver._ensure_store()
        async with store.transaction() as relate_tx:
            self._active_tx = relate_tx
            try:
                return await func(self, *args, **kwargs)
            finally:
                self._active_tx = None

    async def run(self, query: str | list[Any], **kwargs: Any) -> Any:
        if isinstance(query, list):
            last_result: Any = ([], [], None)
            for item in query:
                if isinstance(item, tuple):
                    cypher, params = item
                else:
                    cypher, params = str(item), kwargs
                if self._active_tx is not None:
                    last_result = await self._driver._execute_in_tx(
                        self._active_tx, str(cypher), params
                    )
                else:
                    last_result = await self._driver.execute_query(str(cypher), **params)
            return last_result

        if self._active_tx is not None:
            return await self._driver._execute_in_tx(self._active_tx, str(query), kwargs)
        return await self._driver.execute_query(str(query), **kwargs)


class _RelateTransaction(Transaction):
    """Wraps a RelateTransaction for the Graphiti Transaction ABC."""

    def __init__(self, tx: RelateTransaction, driver: RelateDriver):
        self._tx = tx
        self._driver = driver

    async def run(self, query: str, **kwargs: Any) -> Any:
        return await self._driver._execute_in_tx(self._tx, query, kwargs)


class RelateDriver(GraphDriver):
    provider = GraphProvider.RELATE
    default_group_id: str = ''
    fulltext_syntax: str = ''
    aoss_client: None = None

    def __init__(
        self,
        database: str | PathLike[str] = ':memory:',
        *,
        namespace: str = 'neo4j',
        store: RelateStore | None = None,
    ):
        super().__init__()
        self._database = os.fsdecode(os.fspath(database)) if store is None else store.database
        self._namespace = namespace if store is None else store.namespace
        self._store: RelateStore | None = store
        self._open_lock: asyncio.Lock | None = None
        self._closed = False
        self._init_task: asyncio.Task[Any] | None = None

        self._entity_node_ops = RelateEntityNodeOperations()
        self._episode_node_ops = RelateEpisodeNodeOperations()
        self._community_node_ops = RelateCommunityNodeOperations()
        self._saga_node_ops = RelateSagaNodeOperations()
        self._entity_edge_ops = RelateEntityEdgeOperations()
        self._episodic_edge_ops = RelateEpisodicEdgeOperations()
        self._community_edge_ops = RelateCommunityEdgeOperations()
        self._has_episode_edge_ops = RelateHasEpisodeEdgeOperations()
        self._next_episode_edge_ops = RelateNextEpisodeEdgeOperations()
        self._search_ops = RelateSearchOperations()
        self._graph_ops = RelateGraphMaintenanceOperations()

        # The legacy interfaces delegate to the operations above.
        self.search_interface = RelateSearchInterface()
        self.graph_operations_interface = RelateGraphOperationsInterface()

    @classmethod
    async def open(
        cls,
        database: str | PathLike[str] = ':memory:',
        *,
        namespace: str = 'neo4j',
    ) -> RelateDriver:
        store = await RelateStore.open(database, namespace=namespace)
        return cls(database=database, namespace=namespace, store=store)

    def _get_open_lock(self) -> asyncio.Lock:
        if self._open_lock is None:
            self._open_lock = asyncio.Lock()
        return self._open_lock

    async def _ensure_store(self) -> RelateStore:
        if self._closed:
            raise StoreClosedError('store does not accept new operations')
        if self._store is not None:
            return self._store
        async with self._get_open_lock():
            if self._closed:
                raise StoreClosedError('store does not accept new operations')
            if self._store is None:
                opened: RelateStore = await RelateStore.open(
                    self._database, namespace=self._namespace
                )
                self._store = opened
                return opened
            return self._store

    # --- Operations properties ---

    @property
    def entity_node_ops(self) -> EntityNodeOperations:
        return self._entity_node_ops

    @property
    def episode_node_ops(self) -> EpisodeNodeOperations:
        return self._episode_node_ops

    @property
    def community_node_ops(self) -> CommunityNodeOperations:
        return self._community_node_ops

    @property
    def saga_node_ops(self) -> SagaNodeOperations:
        return self._saga_node_ops

    @property
    def entity_edge_ops(self) -> EntityEdgeOperations:
        return self._entity_edge_ops

    @property
    def episodic_edge_ops(self) -> EpisodicEdgeOperations:
        return self._episodic_edge_ops

    @property
    def community_edge_ops(self) -> CommunityEdgeOperations:
        return self._community_edge_ops

    @property
    def has_episode_edge_ops(self) -> HasEpisodeEdgeOperations:
        return self._has_episode_edge_ops

    @property
    def next_episode_edge_ops(self) -> NextEpisodeEdgeOperations:
        return self._next_episode_edge_ops

    @property
    def search_ops(self) -> SearchOperations:
        return self._search_ops

    @property
    def graph_ops(self) -> GraphMaintenanceOperations:
        return self._graph_ops

    async def _execute_in_tx(
        self,
        tx: RelateTransaction,
        cypher_query_: str,
        kwargs: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], list[str], None]:
        translated = translate_relate_query(cypher_query_)
        params = _extract_params(kwargs)
        try:
            result = await tx.execute(translated, params)
        except Exception as exc:
            logger.error(
                'Error executing Relate transaction query (%s); parameter keys: %s',
                type(exc).__name__,
                sorted(params),
            )
            raise
        return _format_query_result(result)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Transaction]:
        store = await self._ensure_store()
        async with store.transaction() as tx:
            yield _RelateTransaction(tx, self)

    async def execute_query(
        self,
        cypher_query_: str | list[Any],
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], list[str], None]:
        if isinstance(cypher_query_, list):
            last_result: tuple[list[dict[str, Any]], list[str], None] = ([], [], None)
            for item in cypher_query_:
                if isinstance(item, tuple):
                    cypher, params = item
                else:
                    cypher, params = str(item), kwargs
                last_result = await self.execute_query(str(cypher), **params)
            return last_result

        store = await self._ensure_store()
        translated = translate_relate_query(cypher_query_)
        params = _extract_params(kwargs)
        try:
            result = await store.execute(translated, params)
        except Exception as exc:
            logger.error(
                'Error executing Relate query (%s); parameter keys: %s',
                type(exc).__name__,
                sorted(params),
            )
            raise
        return _format_query_result(result)

    def session(self, database: str | None = None) -> GraphDriverSession:  # noqa: ARG002
        return RelateDriverSession(self)

    async def close(self) -> None:
        if self._init_task is not None:
            if not self._init_task.done():
                self._init_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._init_task
            elif not self._init_task.cancelled():
                self._init_task.exception()

        self._closed = True
        async with self._get_open_lock():
            store = self._store
        if store is not None:
            await store.close()

    async def delete_all_indexes(self) -> None:
        await self._graph_ops.delete_all_indexes(self)

    async def build_indices_and_constraints(self, delete_existing: bool = False) -> None:
        await self._graph_ops.build_indices_and_constraints(self, delete_existing=delete_existing)

    async def health_check(self) -> None:
        await self.execute_query('RETURN 1 AS ok')

    def build_fulltext_query(
        self,
        query: str,
        group_ids: list[str] | None = None,
        max_query_length: int = 128,
    ) -> str:
        return self._search_ops.build_fulltext_query(query, group_ids, max_query_length)
