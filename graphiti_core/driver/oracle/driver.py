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

import copy
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import as_file, files
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import oracledb
else:
    try:
        import oracledb
    except ImportError:
        raise ImportError(
            'oracledb is required for Oracle26aiDriver. '
            'Install it with: pip install graphiti-core[oracle]'
        ) from None
from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.driver.query_executor import Transaction

from .config import Oracle26aiConfig
from .errors import GroupScopeConflict
from .legacy import OracleLegacyGraphOperations, OracleLegacySearchInterface
from .migrations import MigrationRunner, load_migrations
from .operations import (
    OracleCommunityEdgeOperations,
    OracleCommunityNodeOperations,
    OracleEntityEdgeOperations,
    OracleEntityNodeOperations,
    OracleEpisodeNodeOperations,
    OracleEpisodicEdgeOperations,
    OracleGraphMaintenanceOperations,
    OracleHasEpisodeEdgeOperations,
    OracleNextEpisodeEdgeOperations,
    OracleSagaNodeOperations,
    OracleSearchOperations,
)
from .protocols import AsyncPoolProtocol
from .schema import SchemaInspector
from .session import Oracle26aiSession
from .sql.maintenance import SYNC_FULLTEXT_INDEX
from .transaction import Oracle26aiTransaction

_TEXT_INDEXES = (
    'GRAPHITI_NODES_TEXT_IDX',
    'GRAPHITI_EDGES_TEXT_IDX',
)
_SEARCH_INDEXES = (
    *_TEXT_INDEXES,
    'GRAPHITI_NODES_VECTOR_IDX',
    'GRAPHITI_EDGES_VECTOR_IDX',
)


class Oracle26aiDriver(GraphDriver):
    provider = GraphProvider.ORACLE
    default_group_id = ''
    fulltext_syntax = ''
    aoss_client = None

    def __init__(
        self,
        config: Oracle26aiConfig,
        *,
        pool: AsyncPoolProtocol | None = None,
    ) -> None:
        self.config = config
        self._database = self.default_group_id
        self._scope_group_id: str | None = None
        self._owns_pool = True
        self._entity_node_ops = OracleEntityNodeOperations()
        self._episode_node_ops = OracleEpisodeNodeOperations()
        self._community_node_ops = OracleCommunityNodeOperations()
        self._saga_node_ops = OracleSagaNodeOperations()
        self._entity_edge_ops = OracleEntityEdgeOperations()
        self._episodic_edge_ops = OracleEpisodicEdgeOperations()
        self._community_edge_ops = OracleCommunityEdgeOperations()
        self._has_episode_edge_ops = OracleHasEpisodeEdgeOperations()
        self._next_episode_edge_ops = OracleNextEpisodeEdgeOperations()
        self._graph_ops = OracleGraphMaintenanceOperations()
        self._search_ops = OracleSearchOperations()
        self.graph_operations_interface = OracleLegacyGraphOperations()
        self.search_interface = OracleLegacySearchInterface()
        self._pool = cast(
            AsyncPoolProtocol,
            pool
            or oracledb.create_pool_async(
                user=config.user,
                password=config.password.get_secret_value(),
                dsn=config.dsn,
                config_dir=config.config_dir,
                wallet_location=config.wallet_location,
                wallet_password=(
                    config.wallet_password.get_secret_value() if config.wallet_password else None
                ),
                min=config.pool_min,
                max=config.pool_max,
                increment=1,
            ),
        )

    async def execute_query(
        self,
        cypher_query_: str,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], list[str], None]:
        return await self.session().run(cypher_query_, **kwargs)

    def session(self, database: str | None = None) -> GraphDriverSession:
        return Oracle26aiSession(self._pool)

    async def close(self) -> None:
        if self._owns_pool:
            await self._pool.close()

    async def sync_fulltext_indexes(self) -> tuple[str, str]:
        """Synchronize driver-owned Oracle Text indexes after a committed write batch."""

        self._require_unscoped_admin('sync_fulltext_indexes')
        for index_name in _TEXT_INDEXES:
            await self.execute_query(SYNC_FULLTEXT_INDEX, index_name=index_name)
        return _TEXT_INDEXES

    async def delete_all_indexes(self) -> None:
        self._require_unscoped_admin('delete_all_indexes')
        records, _, _ = await self.execute_query(
            'SELECT index_name FROM user_indexes '
            'WHERE index_name IN ('
            "'GRAPHITI_NODES_TEXT_IDX', 'GRAPHITI_EDGES_TEXT_IDX', "
            "'GRAPHITI_NODES_VECTOR_IDX', 'GRAPHITI_EDGES_VECTOR_IDX'"
            ') ORDER BY index_name'
        )
        for record in records:
            name = str(record['index_name']).upper()
            if name not in _SEARCH_INDEXES:
                raise RuntimeError(f'refusing to drop unmanaged index {name}')
            await self.execute_query(f'DROP INDEX {name}')

    async def build_indices_and_constraints(self, delete_existing: bool = False) -> None:
        self._require_unscoped_admin('build_indices_and_constraints')
        migration_resources = files('graphiti_core.driver.oracle.sql.migrations')
        with as_file(migration_resources) as migration_directory:
            dropped_existing = False
            if delete_existing:
                before = await self._managed_index_count()
                await self.delete_all_indexes()
                dropped_existing = before > 0

            await MigrationRunner(self, migration_directory).apply()
            if dropped_existing:
                search_migration = next(
                    migration
                    for migration in load_migrations(migration_directory)
                    if migration.version == '003'
                )
                for statement in search_migration.statements:
                    await self.execute_query(statement)

    async def _managed_index_count(self) -> int:
        records, _, _ = await self.execute_query(
            'SELECT COUNT(*) AS object_count FROM user_indexes '
            'WHERE index_name IN ('
            "'GRAPHITI_NODES_TEXT_IDX', 'GRAPHITI_EDGES_TEXT_IDX', "
            "'GRAPHITI_NODES_VECTOR_IDX', 'GRAPHITI_EDGES_VECTOR_IDX'"
            ')'
        )
        return int(records[0]['object_count'])

    def _require_unscoped_admin(self, operation: str) -> None:
        if self._scope_group_id is not None:
            raise GroupScopeConflict(
                f'{operation} is unavailable inside request scope {self._scope_group_id}'
            )

    async def schema_readiness(self):
        return await SchemaInspector(self).readiness()

    def clone(self, database: str) -> Oracle26aiDriver:
        current_scope = self._scope_group_id
        if not isinstance(database, str) or not database:
            suffix = f' {current_scope}' if current_scope is not None else ''
            raise GroupScopeConflict(f'cannot remove or create an empty request scope{suffix}')
        if current_scope is not None and database != current_scope:
            raise GroupScopeConflict(f'cannot change request scope {current_scope} to {database}')
        scoped = copy.copy(self)
        scoped._database = database
        scoped._scope_group_id = database
        scoped._owns_pool = False
        return scoped

    def with_database(self, database: str) -> Oracle26aiDriver:
        return self.clone(database)

    @property
    def entity_node_ops(self) -> OracleEntityNodeOperations:
        return self._entity_node_ops

    @property
    def episode_node_ops(self) -> OracleEpisodeNodeOperations:
        return self._episode_node_ops

    @property
    def community_node_ops(self) -> OracleCommunityNodeOperations:
        return self._community_node_ops

    @property
    def saga_node_ops(self) -> OracleSagaNodeOperations:
        return self._saga_node_ops

    @property
    def entity_edge_ops(self) -> OracleEntityEdgeOperations:
        return self._entity_edge_ops

    @property
    def episodic_edge_ops(self) -> OracleEpisodicEdgeOperations:
        return self._episodic_edge_ops

    @property
    def community_edge_ops(self) -> OracleCommunityEdgeOperations:
        return self._community_edge_ops

    @property
    def has_episode_edge_ops(self) -> OracleHasEpisodeEdgeOperations:
        return self._has_episode_edge_ops

    @property
    def next_episode_edge_ops(self) -> OracleNextEpisodeEdgeOperations:
        return self._next_episode_edge_ops

    @property
    def graph_ops(self) -> OracleGraphMaintenanceOperations:
        return self._graph_ops

    @property
    def search_ops(self) -> OracleSearchOperations:
        return self._search_ops

    def build_fulltext_query(
        self,
        query: str,
        group_ids: list[str] | None = None,
        max_query_length: int = 8000,
    ) -> str:
        return self._search_ops.build_fulltext_query(query, group_ids, max_query_length)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Transaction]:
        connection = await self._pool.acquire()
        transaction = Oracle26aiTransaction(connection)
        try:
            yield transaction
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise
        finally:
            await self._pool.release(connection)
