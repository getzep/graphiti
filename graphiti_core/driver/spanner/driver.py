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
import copy
import functools
import logging
import os
import re
import threading
from collections.abc import AsyncIterator, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from typing import Any, Literal, TypeVar

from google.api_core.exceptions import AlreadyExists, GoogleAPICallError
from google.auth.credentials import AnonymousCredentials
from google.cloud import spanner
from google.cloud.spanner_v1 import KeySet

from graphiti_core.driver.driver import GraphDriver, GraphProvider
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
from graphiti_core.driver.spanner.codecs import encode_params, infer_param_types, rows_to_dicts
from graphiti_core.driver.spanner.config import SpannerConfig
from graphiti_core.driver.spanner.errors import (
    SpannerCypherNotSupportedError,
    SpannerDriverError,
    SpannerSchemaError,
    query_snippet,
)
from graphiti_core.driver.spanner.fulltext import build_fulltext_query
from graphiti_core.driver.spanner.legacy import (
    SpannerLegacyGraphOperations,
    SpannerLegacySearchInterface,
)
from graphiti_core.driver.spanner.mutations import Mutation, chunk_mutations
from graphiti_core.driver.spanner.operations import (
    SpannerCommunityEdgeOperations,
    SpannerCommunityNodeOperations,
    SpannerEntityEdgeOperations,
    SpannerEntityNodeOperations,
    SpannerEpisodeNodeOperations,
    SpannerEpisodicEdgeOperations,
    SpannerGraphMaintenanceOperations,
    SpannerHasEpisodeEdgeOperations,
    SpannerNextEpisodeEdgeOperations,
    SpannerSagaNodeOperations,
    SpannerSearchOperations,
)
from graphiti_core.driver.spanner.schema import (
    EDGE_TABLES,
    INDEXES,
    KIND_INDEX,
    KIND_PROPERTY_GRAPH,
    KIND_SEARCH_INDEX,
    KIND_TABLE,
    KIND_VECTOR_INDEX,
    NODE_TABLES,
    SEARCH_INDEXES,
    build_ddl,
    build_drop_ddl,
    ddl_object_name,
)
from graphiti_core.driver.spanner.session import SpannerSession
from graphiti_core.driver.spanner.transaction import SpannerTransaction

logger = logging.getLogger(__name__)

T = TypeVar('T')

EMULATOR_ENV_VAR = 'SPANNER_EMULATOR_HOST'

StatementKind = Literal['read', 'dml', 'ddl']

# Leading whitespace, comments and statement hints (@{...}) before the first keyword.
_LEADING_NOISE = re.compile(
    r'(?:\s+|--[^\n]*(?:\n|$)|#[^\n]*(?:\n|$)|/\*.*?\*/|@\{[^}]*\})*', re.DOTALL
)
_FIRST_WORD = re.compile(r'[A-Za-z_]+')
_CYPHER_PARAMETER = re.compile(r'\$[A-Za-z_]')
_CYPHER_CLAUSE = re.compile(r'\b(?:MERGE|DETACH\s+DELETE|UNWIND)\b|\bCREATE\s*\(', re.IGNORECASE)
_MATCH = re.compile(r'\bMATCH\b', re.IGNORECASE)
_GRAPH_TABLE = re.compile(r'\bGRAPH_TABLE\s*\(', re.IGNORECASE)

_READ_KEYWORDS = frozenset({'SELECT', 'WITH', 'GRAPH'})
_DDL_KEYWORDS = frozenset({'CREATE', 'ALTER', 'DROP'})

# INFORMATION_SCHEMA.INDEXES.INDEX_TYPE -> ddl_object_name kind
_INDEX_KINDS = {'INDEX': KIND_INDEX, 'SEARCH': KIND_SEARCH_INDEX, 'VECTOR': KIND_VECTOR_INDEX}


def strip_leading_noise(query: str) -> str:
    """Drop leading whitespace, comments and statement hints from a query."""
    match = _LEADING_NOISE.match(query)
    return query[match.end() :] if match else query


def _first_keyword(query: str) -> str:
    match = _FIRST_WORD.match(strip_leading_noise(query))
    return match.group(0).upper() if match else ''


def ensure_not_cypher(query: str) -> None:
    """Raise :class:`SpannerCypherNotSupportedError` if ``query`` looks like Cypher.

    Cypher markers: ``$param`` placeholders, ``MERGE``, ``DETACH DELETE``, ``UNWIND``,
    ``CREATE (``, and ``MATCH`` outside a GQL query (one starting with ``GRAPH``) or a SQL
    ``GRAPH_TABLE(...)``.
    """
    if _CYPHER_PARAMETER.search(query):
        raise SpannerCypherNotSupportedError(query, '$name parameters; Spanner uses @name')
    clause = _CYPHER_CLAUSE.search(query)
    if clause is not None:
        name = ' '.join(clause.group(0).rstrip('(').split()).upper()
        raise SpannerCypherNotSupportedError(query, f'Cypher clause {name}')
    if _MATCH.search(query) and _first_keyword(query) != 'GRAPH' and not _GRAPH_TABLE.search(query):
        raise SpannerCypherNotSupportedError(query, 'MATCH without a leading GRAPH clause')


def classify_statement(query: str) -> StatementKind:
    """``'read'`` for SELECT/WITH/GRAPH (or a parenthesised query), ``'ddl'`` for
    CREATE/ALTER/DROP, ``'dml'`` for everything else (INSERT/UPDATE/DELETE)."""
    if strip_leading_noise(query).startswith('('):
        return 'read'
    keyword = _first_keyword(query)
    if keyword in _READ_KEYWORDS:
        return 'read'
    if keyword in _DDL_KEYWORDS:
        return 'ddl'
    return 'dml'


def _is_duplicate_schema_error(error: BaseException | None) -> bool:
    return isinstance(error, AlreadyExists) or (
        error is not None and 'duplicate name' in str(error).lower()
    )


def close_database_in_background(database: Any) -> None:
    """Close a ``Database`` without blocking the caller.

    ``Database.close()`` in google-cloud-spanner 3.71 joins the multiplexed-session
    maintenance thread, which sleeps in 10-minute steps, so a direct call can block for up
    to 10 minutes. Run it on a daemon thread instead: the session is released as soon as the
    maintenance thread wakes up, and a daemon thread never delays interpreter exit. Only the
    client's public API is used.
    """
    threading.Thread(target=database.close, name='graphiti-spanner-close', daemon=True).start()


class SpannerDriver(GraphDriver):
    """Graphiti driver for Google Cloud Spanner Graph.

    A native GoogleSQL/GQL backend: it never translates Cypher. Reads run in read-only
    snapshots, writes go through the Mutation API (``apply_mutations``) or DML, and schema
    changes through ``update_ddl``. The synchronous Spanner client runs on a bounded thread
    pool so every public method is ``async``.

    Construct with a :class:`SpannerConfig` or the same fields as keyword arguments::

        SpannerDriver(project='p', instance='i', database='d', embedding_dim=1024)

    The schema is not created automatically; call ``build_indices_and_constraints()`` once
    (Graphiti does this from ``Graphiti.build_indices_and_constraints``).
    """

    provider = GraphProvider.SPANNER
    fulltext_syntax: str = ''
    default_group_id: str = ''
    aoss_client: None = None

    def __init__(self, config: SpannerConfig | None = None, **kwargs: Any):
        super().__init__()
        if config is None:
            config = SpannerConfig(**kwargs)
        elif kwargs:
            config = SpannerConfig(**{**dict(config), **kwargs})
        self.config = config
        self.graph_name = config.graph_name
        self.embedding_dim = config.embedding_dim

        if config.emulator_host:
            current = os.environ.get(EMULATOR_ENV_VAR)
            if not current:
                # The Spanner client reads the emulator endpoint from the environment.
                os.environ[EMULATOR_ENV_VAR] = config.emulator_host
            elif current != config.emulator_host:
                logger.warning(
                    f'{EMULATOR_ENV_VAR}={current} overrides emulator_host='
                    f'{config.emulator_host}; the Spanner client reads the environment'
                )
        credentials = AnonymousCredentials() if config.emulator_host else config.credentials

        self.client = spanner.Client(project=config.project, credentials=credentials)
        self.instance = self.client.instance(config.instance)
        self.database = self.instance.database(config.database)
        self._database = config.database
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_workers, thread_name_prefix='graphiti-spanner'
        )
        self._closed = False
        self._property_graph_view: bool | None = None

        self._entity_node_ops = SpannerEntityNodeOperations()
        self._episode_node_ops = SpannerEpisodeNodeOperations()
        self._community_node_ops = SpannerCommunityNodeOperations()
        self._saga_node_ops = SpannerSagaNodeOperations()
        self._entity_edge_ops = SpannerEntityEdgeOperations()
        self._episodic_edge_ops = SpannerEpisodicEdgeOperations()
        self._community_edge_ops = SpannerCommunityEdgeOperations()
        self._has_episode_edge_ops = SpannerHasEpisodeEdgeOperations()
        self._next_episode_edge_ops = SpannerNextEpisodeEdgeOperations()
        self._search_ops = SpannerSearchOperations()
        self._graph_ops = SpannerGraphMaintenanceOperations()

        # Legacy hooks consulted by nodes.py, edges.py, bulk_utils.py and search_utils.py.
        self.graph_operations_interface = SpannerLegacyGraphOperations()
        self.search_interface = SpannerLegacySearchInterface()

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

    # --- I/O boundary: every blocking client call goes through here ---

    async def _run_blocking(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        if self._closed:
            raise SpannerDriverError('SpannerDriver is closed')
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, functools.partial(fn, *args, **kwargs))

    # --- Queries ---

    async def read(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        param_types: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a GoogleSQL or GQL query in a read-only snapshot.

        Returns one dict per row keyed by column name, with values converted to plain Python
        (``datetime``, ``dict``, ``list``). Parameter types are inferred
        (:func:`codecs.infer_param_types`); ``param_types`` overrides them per parameter.
        """
        params = params or {}
        types = infer_param_types(params, param_types)
        encoded = encode_params(params, types)
        database = self.database

        def _read() -> list[dict[str, Any]]:
            with database.snapshot() as snapshot:
                result = snapshot.execute_sql(
                    sql, params=encoded or None, param_types=types or None
                )
                return rows_to_dicts(result)

        return await self._run_blocking(_read)

    async def execute_dml(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        param_types: dict[str, Any] | None = None,
    ) -> int:
        """Run one DML statement in a read-write transaction; returns the row count.

        Uses the client's retrying ``run_in_transaction``, which is safe for a single
        statement. Do not use it for work that has side effects outside Spanner.
        """
        params = params or {}
        types = infer_param_types(params, param_types)
        encoded = encode_params(params, types)
        database = self.database

        def _dml() -> int:
            return database.run_in_transaction(
                lambda txn: txn.execute_update(
                    sql, params=encoded or None, param_types=types or None
                )
            )

        return await self._run_blocking(_dml)

    async def execute_query(
        self, cypher_query_: str, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], list[str], None]:
        """Run GoogleSQL or GQL; raise :class:`SpannerCypherNotSupportedError` on Cypher.

        Keyword arguments are query parameters (``@name`` in the query), except
        ``param_types_`` (explicit Spanner types), ``database_`` and ``routing_`` (accepted
        for Neo4j call compatibility and ignored; use ``with_database`` to switch databases).
        A ``params=`` dict is merged into the parameters, as in ``Neo4jDriver``.

        SELECT/WITH/GRAPH queries return ``(records, column_names, None)``. DML and DDL
        (CREATE/ALTER/DROP, applied with ``update_ddl``) return ``([], [], None)``.
        """
        kwargs.pop('database_', None)
        kwargs.pop('routing_', None)
        explicit_types: dict[str, Any] | None = kwargs.pop('param_types_', None)
        if isinstance(kwargs.get('params'), dict):
            kwargs = {**kwargs.pop('params'), **kwargs}
        params = kwargs

        ensure_not_cypher(cypher_query_)
        kind = classify_statement(cypher_query_)
        try:
            if kind == 'read':
                records = await self.read(cypher_query_, params, explicit_types)
                return records, list(records[0].keys()) if records else [], None
            if kind == 'ddl':
                await self.update_ddl([cypher_query_])
            else:
                await self.execute_dml(cypher_query_, params, explicit_types)
            return [], [], None
        except (GoogleAPICallError, SpannerSchemaError) as e:
            # Log parameter names only; values may contain sensitive data.
            logger.error(
                f'Error executing Spanner query: {e}\n{query_snippet(cypher_query_, 500)}\n'
                f'parameter keys: {sorted(params)}'
            )
            raise

    async def execute_partitioned_dml(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        param_types: dict[str, Any] | None = None,
    ) -> int:
        """Run partitioned DML (bulk UPDATE/DELETE, not atomic); returns a lower bound of the
        rows changed. Delete from edge tables before their parent node tables."""
        ensure_not_cypher(sql)
        params = params or {}
        types = infer_param_types(params, param_types)
        encoded = encode_params(params, types)
        return await self._run_blocking(
            self.database.execute_partitioned_dml,
            sql,
            params=encoded or None,
            param_types=types or None,
        )

    # --- Writes ---

    async def apply_mutations(
        self, mutations: Iterable[Mutation], tx: Transaction | None = None
    ) -> None:
        """Apply mutations, or buffer them in ``tx`` when one is given.

        Without ``tx`` the mutations are applied now, in order, in ``database.batch()``
        commits of at most ``config.mutation_chunk_rows`` rows each. Each commit is atomic;
        the whole call is atomic only when it fits in one commit.
        """
        if tx is not None:
            if not isinstance(tx, SpannerTransaction):
                raise SpannerDriverError(
                    f'apply_mutations needs a SpannerTransaction, got {type(tx).__name__}'
                )
            for mutation in mutations:
                tx.buffer(mutation)
            return

        batches = chunk_mutations(mutations, self.config.mutation_chunk_rows)
        if not batches:
            return
        database = self.database

        def _apply() -> None:
            for batch_mutations in batches:
                with database.batch() as batch:
                    for mutation in batch_mutations:
                        if mutation.kind == 'insert_or_update':
                            batch.insert_or_update(
                                table=mutation.table,
                                columns=list(mutation.columns),
                                values=[list(row) for row in mutation.values],
                            )
                        else:
                            batch.delete(
                                table=mutation.table,
                                keyset=KeySet(keys=[list(key) for key in mutation.keys]),
                            )

        await self._run_blocking(_apply)

    # --- Schema ---

    async def update_ddl(self, statements: list[str]) -> None:
        """Apply DDL statements in one schema change and wait for it to finish.

        Raises :class:`SpannerSchemaError` (caused by the client error) on failure.
        """
        if not statements:
            return
        database = self.database
        timeout = self.config.ddl_timeout_seconds

        def _apply() -> None:
            database.update_ddl(list(statements)).result(timeout)

        try:
            await self._run_blocking(_apply)
        except (GoogleAPICallError, TimeoutError) as e:
            raise SpannerSchemaError(f'Spanner DDL failed: {e}') from e

    async def _existing_property_graphs(self) -> set[str] | None:
        """Names of existing property graphs, or ``None`` when the INFORMATION_SCHEMA view
        is unavailable."""
        if self._property_graph_view is False:
            return None
        try:
            rows = await self.read(
                'SELECT PROPERTY_GRAPH_NAME AS name FROM INFORMATION_SCHEMA.PROPERTY_GRAPHS'
            )
        except GoogleAPICallError:
            self._property_graph_view = False
            return None
        self._property_graph_view = True
        return {row['name'] for row in rows}

    async def existing_schema_objects(self) -> set[tuple[str, str]]:
        """``(kind, name)`` of every table, secondary/search/vector index and property graph
        in the default schema, using the same kinds as ``schema.ddl_object_name``."""
        tables = await self.read(
            "SELECT TABLE_NAME AS name FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ''"
        )
        indexes = await self.read(
            'SELECT INDEX_NAME AS name, INDEX_TYPE AS index_type '
            "FROM INFORMATION_SCHEMA.INDEXES WHERE TABLE_SCHEMA = '' "
            "AND INDEX_TYPE IN ('INDEX', 'SEARCH', 'VECTOR')"
        )
        objects = {(KIND_TABLE, row['name']) for row in tables}
        objects.update((_INDEX_KINDS[row['index_type']], row['name']) for row in indexes)
        graphs = await self._existing_property_graphs()
        if graphs is not None:
            objects.update((KIND_PROPERTY_GRAPH, name) for name in graphs)
        return objects

    def _pending_statements(
        self, statements: list[str], existing: set[tuple[str, str]]
    ) -> list[str]:
        """CREATE statements whose object is missing. Without the PROPERTY_GRAPHS view the
        ``CREATE OR REPLACE PROPERTY GRAPH`` statement is always kept."""
        pending: list[str] = []
        for statement in statements:
            key = ddl_object_name(statement)
            if (
                key[0] == KIND_PROPERTY_GRAPH
                and self._property_graph_view is False
                or key not in existing
            ):
                pending.append(statement)
        return pending

    async def build_indices_and_constraints(self, delete_existing: bool = False) -> list[str]:
        """Create the tables, indexes, search indexes and property graph that are missing.

        Idempotent: existing objects are found through INFORMATION_SCHEMA and skipped, so a
        second call applies nothing. With ``delete_existing=True`` the existing schema
        objects (and their data) are dropped first and everything is recreated. Returns the
        DDL statements applied, in order. The schema is not migrated: an existing table with
        a different layout (e.g. another ``embedding_dim``) is left as is.
        """
        statements = build_ddl(self.embedding_dim, self.graph_name)
        existing = await self.existing_schema_objects()
        applied: list[str] = []

        if delete_existing:
            drops: list[str] = []
            for statement in build_drop_ddl(self.graph_name):
                key = ddl_object_name(statement)
                if key[0] == KIND_PROPERTY_GRAPH and self._property_graph_view is False:
                    drops.append(f'DROP PROPERTY GRAPH IF EXISTS {self.graph_name}')
                elif key in existing:
                    drops.append(statement)
            await self.update_ddl(drops)
            applied.extend(drops)
            existing = set()

        pending = self._pending_statements(statements, existing)
        try:
            await self.update_ddl(pending)
        except SpannerSchemaError as e:
            if not _is_duplicate_schema_error(e.__cause__):
                raise
            # Another process created part of the schema concurrently. Statements in a
            # failed batch that ran before the failure stay applied; retry what is missing.
            existing = await self.existing_schema_objects()
            pending = self._pending_statements(pending, existing)
            await self.update_ddl(pending)
        applied.extend(pending)
        return applied

    async def delete_all_indexes(self) -> list[str]:
        """Drop the secondary and search indexes this driver creates (not tables, not the
        property graph). Returns the DROP statements applied."""
        existing = await self.existing_schema_objects()
        drops = [
            index.drop_ddl()
            for index in reversed(SEARCH_INDEXES)
            if (KIND_SEARCH_INDEX, index.name) in existing
        ]
        drops.extend(
            index.drop_ddl() for index in reversed(INDEXES) if (KIND_INDEX, index.name) in existing
        )
        await self.update_ddl(drops)
        return drops

    async def clear_tables(self) -> int:
        """Delete every row (edge tables first, then node tables) with partitioned DML.

        Not atomic. Returns the lower-bound row count reported by Spanner. Backs
        ``clear_data()`` without group ids and the test fixtures.
        """
        total = 0
        for table in (*EDGE_TABLES, *NODE_TABLES):
            total += await self.execute_partitioned_dml(f'DELETE FROM {table} WHERE TRUE')
        return total

    # --- Sessions, transactions, lifecycle ---

    def session(self, database: str | None = None) -> SpannerSession:
        if database is None or database == self._database:
            return SpannerSession(self)
        return SpannerSession(self.with_database(database))

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[SpannerTransaction]:
        """Yield a :class:`SpannerTransaction`; its buffered mutations are committed on
        clean exit and discarded on exception."""
        tx = SpannerTransaction(self)
        try:
            yield tx
        except BaseException:
            tx.discard()
            raise
        await tx.commit()

    def with_database(self, database: str) -> SpannerDriver:
        """Shallow copy pointing at another database in the same instance.

        The copy shares the client, the instance and the thread pool; close only the
        original driver.
        """
        cloned = copy.copy(self)
        cloned.config = SpannerConfig(**{**dict(self.config), 'database': database})
        cloned.database = self.instance.database(database)
        cloned._database = database
        cloned._property_graph_view = None
        return cloned

    def clone(self, database: str) -> SpannerDriver:
        """Groups live in a ``group_id`` column, not in separate databases."""
        return self

    def build_fulltext_query(
        self, query: str, group_ids: list[str] | None = None, max_query_length: int = 128
    ) -> str:
        return build_fulltext_query(query, group_ids, max_query_length)

    async def health_check(self) -> None:
        await self.read('SELECT 1 AS ok')

    async def close(self) -> None:
        """Release the database session and stop the thread pool. Idempotent."""
        if self._closed:
            return
        self._closed = True
        with suppress(Exception):
            close_database_in_background(self.database)
        self._executor.shutdown(wait=False)
        client_close = getattr(self.client, 'close', None)
        if callable(client_close):
            with suppress(Exception):
                client_close()
