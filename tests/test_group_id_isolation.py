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

Tests for group-ID isolation (race condition fix).

The core invariant: calling add_episode / add_episode_bulk with a group_id
that differs from the driver's current database must NOT mutate
self.driver or self.clients.driver.  Instead a short-lived clone is
injected by @handle_single_group_id and discarded after the call.
"""

import asyncio
import copy
import os
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EpisodeType, EpisodicNode

# ---------------------------------------------------------------------------
# Minimal concrete GraphDriver for unit testing (no real DB connection)
# ---------------------------------------------------------------------------


class _FakeDriverSession(GraphDriverSession):
    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def run(self, query: str, **kwargs: Any) -> Any:
        return []

    async def close(self):
        pass

    async def execute_write(self, func, *args, **kwargs):
        return await func(self, *args, **kwargs)


class _FakeFalkorDriver(GraphDriver):
    """Minimal concrete GraphDriver that behaves like FalkorDB for decorator tests."""

    provider: GraphProvider = GraphProvider.FALKORDB

    def __init__(self, database: str = 'default_db'):
        self._database = database
        # Shared across clones (mirrors FalkorDriver behaviour)
        self._initialized_databases: set[str] = {database}
        self.build_calls: list[str] = []

    # -- Abstract method implementations (all no-ops) --

    async def execute_query(self, cypher_query_: str, **kwargs: Any):
        return [], [], {}

    def session(self, database: str | None = None) -> GraphDriverSession:
        return _FakeDriverSession(self)

    async def close(self):
        pass

    async def delete_all_indexes(self):
        pass

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        self.build_calls.append(self._database)

    # QueryExecutor abstract methods
    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def run(self, query: str, **kwargs: Any) -> Any:
        return []

    async def execute_write(self, func, *args, **kwargs):
        return await func(self, *args, **kwargs)

    # -- FalkorDB-like clone / init --

    def clone(self, database: str) -> '_FakeFalkorDriver':
        if database == self._database:
            return self
        cloned = copy.copy(self)
        cloned._database = database
        return cloned

    async def ensure_database_initialized(self) -> None:
        if self._database not in self._initialized_databases:
            self._initialized_databases.add(self._database)
            await self.build_indices_and_constraints()


class _FakeNeo4jDriver(_FakeFalkorDriver):
    """Variant that reports as Neo4j."""

    provider: GraphProvider = GraphProvider.NEO4J

    def __init__(self, database: str = 'neo4j'):
        super().__init__(database=database)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

NOW = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_episode(group_id: str = 'tenant_db') -> EpisodicNode:
    return EpisodicNode(
        name='test_episode',
        group_id=group_id,
        labels=[],
        source=EpisodeType.message,
        source_description='test',
        content='test body',
        created_at=NOW,
        valid_at=NOW,
        entity_edges=[],
    )


def _make_graphiti(driver: GraphDriver) -> Graphiti:
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.model = 'test'
    mock_llm.small_model = 'test'
    mock_llm.temperature = 0.0
    mock_llm.max_tokens = 1000
    mock_llm.cache_enabled = False
    mock_llm.cache_dir = None
    mock_llm.set_tracer = MagicMock()

    return Graphiti(
        graph_driver=driver,
        llm_client=mock_llm,
        embedder=MagicMock(spec=EmbedderClient),
        cross_encoder=MagicMock(spec=CrossEncoderClient),
    )


@contextmanager
def _patch_add_episode_internals(graphiti: Graphiti, episode: EpisodicNode):
    """Patch all heavy async operations invoked by add_episode."""
    with (
        patch.object(graphiti, 'retrieve_episodes', AsyncMock(return_value=[])),
        patch('graphiti_core.graphiti.extract_nodes', new=AsyncMock(return_value=[])),
        patch(
            'graphiti_core.graphiti.resolve_extracted_nodes',
            new=AsyncMock(return_value=([], {}, [])),
        ),
        patch(
            'graphiti_core.graphiti.extract_attributes_from_nodes',
            new=AsyncMock(return_value=[]),
        ),
        patch.object(
            graphiti,
            '_extract_and_resolve_edges',
            AsyncMock(return_value=([], [], [])),
        ),
        patch.object(
            graphiti,
            '_process_episode_data',
            AsyncMock(return_value=([], episode)),
        ),
    ):
        yield


@contextmanager
def _patch_add_episode_bulk_internals(graphiti: Graphiti):
    """Patch the heavy async operations invoked by add_episode_bulk."""
    with (
        patch('graphiti_core.graphiti.add_nodes_and_edges_bulk', new=AsyncMock()),
        patch(
            'graphiti_core.graphiti.retrieve_previous_episodes_bulk',
            new=AsyncMock(return_value={}),
        ),
        patch(
            'graphiti_core.graphiti.dedupe_edges_bulk',
            new=AsyncMock(return_value={}),
        ),
        patch.object(
            graphiti,
            '_extract_and_dedupe_nodes_bulk',
            AsyncMock(return_value=({}, {}, {})),
        ),
        patch.object(
            graphiti,
            '_resolve_nodes_and_edges_bulk',
            AsyncMock(return_value=([], [], [], {})),
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Unit tests – GraphDriver base class
# ---------------------------------------------------------------------------


class TestGraphDriverBaseClass:
    @pytest.mark.asyncio
    async def test_ensure_database_initialized_is_noop(self):
        """GraphDriver.ensure_database_initialized must be a silent no-op on the base class."""
        driver = _FakeNeo4jDriver()
        # Override to use the base-class implementation directly
        result = await GraphDriver.ensure_database_initialized(driver)
        assert result is None
        # Subclass that tracks build calls should have zero calls via base-class path
        assert driver.build_calls == []


# ---------------------------------------------------------------------------
# Unit tests – add_episode driver isolation
# ---------------------------------------------------------------------------


class TestAddEpisodeDriverIsolation:
    @pytest.mark.asyncio
    async def test_driver_not_mutated_when_group_id_differs(self):
        """add_episode with a different group_id must NOT change self.clients.driver."""
        driver = _FakeFalkorDriver(database='default_db')
        graphiti = _make_graphiti(driver)
        episode = _make_episode(group_id='tenant_db')

        with _patch_add_episode_internals(graphiti, episode):
            await graphiti.add_episode(
                name='test',
                episode_body='test body',
                reference_time=NOW,
                source=EpisodeType.message,
                source_description='test',
                group_id='tenant_db',
            )

        # The original driver must be completely untouched
        assert graphiti.clients.driver is driver
        assert graphiti.clients.driver._database == 'default_db'
        assert graphiti.driver is driver
        assert graphiti.driver._database == 'default_db'

    @pytest.mark.asyncio
    async def test_driver_not_mutated_on_neo4j(self):
        """For Neo4j the decorator is bypassed; driver must still be unchanged after the call."""
        driver = _FakeNeo4jDriver(database='neo4j')
        graphiti = _make_graphiti(driver)
        episode = _make_episode(group_id='tenant_db')

        with _patch_add_episode_internals(graphiti, episode):
            await graphiti.add_episode(
                name='test',
                episode_body='test body',
                reference_time=NOW,
                source=EpisodeType.message,
                source_description='test',
                group_id='tenant_db',
            )

        assert graphiti.clients.driver is driver
        assert graphiti.driver is driver

    @pytest.mark.asyncio
    async def test_cloned_driver_used_for_db_operations(self):
        """The injected cloned driver (not self.clients.driver) is passed to internals."""
        driver = _FakeFalkorDriver(database='default_db')
        graphiti = _make_graphiti(driver)
        episode = _make_episode(group_id='tenant_db')

        captured: dict = {}

        async def capture_process_episode_data(
            episode,
            nodes,
            entity_edges,
            now,
            group_id,
            saga=None,
            saga_previous_episode_uuid=None,
            driver=None,
        ):
            captured['driver'] = driver
            return ([], episode)

        with (
            _patch_add_episode_internals(graphiti, episode),
            patch.object(graphiti, '_process_episode_data', capture_process_episode_data),
        ):
            await graphiti.add_episode(
                name='test',
                episode_body='test body',
                reference_time=NOW,
                source=EpisodeType.message,
                source_description='test',
                group_id='tenant_db',
            )

        # The cloned driver (not the original) must be passed to internals
        assert captured.get('driver') is not None
        assert captured['driver'] is not driver
        assert captured['driver']._database == 'tenant_db'

    @pytest.mark.asyncio
    async def test_no_clone_when_group_id_matches_database(self):
        """When group_id matches the driver database, no clone is created."""
        driver = _FakeFalkorDriver(database='mydb')
        graphiti = _make_graphiti(driver)
        episode = _make_episode(group_id='mydb')

        original_clone = driver.clone

        clones_created: list[str] = []

        def tracking_clone(database: str):
            clones_created.append(database)
            return original_clone(database)

        driver.clone = tracking_clone

        with _patch_add_episode_internals(graphiti, episode):
            await graphiti.add_episode(
                name='test',
                episode_body='test body',
                reference_time=NOW,
                source=EpisodeType.message,
                source_description='test',
                group_id='mydb',
            )

        # clone() should not have been called because group_id == _database
        assert clones_created == []

    @pytest.mark.asyncio
    async def test_ensure_database_initialized_called_for_new_group(self):
        """ensure_database_initialized must be called when a new group_id is seen."""
        driver = _FakeFalkorDriver(database='default_db')
        graphiti = _make_graphiti(driver)
        episode = _make_episode(group_id='new_tenant')

        # 'new_tenant' is not in _initialized_databases initially
        assert 'new_tenant' not in driver._initialized_databases

        with _patch_add_episode_internals(graphiti, episode):
            await graphiti.add_episode(
                name='test',
                episode_body='test body',
                reference_time=NOW,
                source=EpisodeType.message,
                source_description='test',
                group_id='new_tenant',
            )

        # After the call, the new group should be initialized (shared set updated via clone)
        assert 'new_tenant' in driver._initialized_databases

    @pytest.mark.asyncio
    async def test_concurrent_calls_with_different_group_ids_no_interference(self):
        """Concurrent add_episode calls with different group_ids must not interfere."""
        driver = _FakeFalkorDriver(database='default_db')
        graphiti = _make_graphiti(driver)

        group_ids = ['tenant_a', 'tenant_b', 'tenant_c']
        episodes = {gid: _make_episode(group_id=gid) for gid in group_ids}

        drivers_seen: dict[str, str] = {}

        async def capture_process_data(
            episode,
            nodes,
            entity_edges,
            now,
            group_id,
            saga=None,
            saga_previous_episode_uuid=None,
            driver=None,
        ):
            drivers_seen[group_id] = driver._database
            return ([], episode)

        async def run_one(gid: str):
            ep = episodes[gid]
            with (
                _patch_add_episode_internals(graphiti, ep),
                patch.object(graphiti, '_process_episode_data', capture_process_data),
            ):
                await graphiti.add_episode(
                    name='test',
                    episode_body='test body',
                    reference_time=NOW,
                    source=EpisodeType.message,
                    source_description='test',
                    group_id=gid,
                )

        await asyncio.gather(*[run_one(gid) for gid in group_ids])

        # After all concurrent calls the original driver must be completely intact
        assert graphiti.clients.driver is driver
        assert graphiti.clients.driver._database == 'default_db'
        assert graphiti.driver._database == 'default_db'

        # Each call must have seen a driver scoped to its own group_id
        for gid in group_ids:
            assert drivers_seen.get(gid) == gid, (
                f'Expected driver._database={gid!r} for group_id={gid!r}, '
                f'got {drivers_seen.get(gid)!r}'
            )


# ---------------------------------------------------------------------------
# Unit tests – add_episode_bulk driver isolation
# ---------------------------------------------------------------------------


class TestAddEpisodeBulkDriverIsolation:
    @pytest.mark.asyncio
    async def test_driver_not_mutated_when_group_id_differs(self):
        """add_episode_bulk with a different group_id must NOT change self.clients.driver."""
        from graphiti_core.utils.bulk_utils import RawEpisode

        driver = _FakeFalkorDriver(database='default_db')
        graphiti = _make_graphiti(driver)

        raw_episodes = [
            RawEpisode(
                name='ep1',
                content='Alice knows Bob',
                source=EpisodeType.message,
                source_description='chat',
                reference_time=NOW,
            )
        ]

        with _patch_add_episode_bulk_internals(graphiti):
            await graphiti.add_episode_bulk(
                bulk_episodes=raw_episodes,
                group_id='tenant_db',
            )

        assert graphiti.clients.driver is driver
        assert graphiti.clients.driver._database == 'default_db'
        assert graphiti.driver is driver
        assert graphiti.driver._database == 'default_db'

    @pytest.mark.asyncio
    async def test_cloned_driver_used_for_bulk_db_operations(self):
        """add_episode_bulk passes the cloned driver to its internal operations."""
        from graphiti_core.utils.bulk_utils import RawEpisode

        driver = _FakeFalkorDriver(database='default_db')
        graphiti = _make_graphiti(driver)

        captured: dict = {}

        async def capture_resolve_all(
            nodes_by_episode,
            edges_by_episode,
            episode_context,
            entity_types,
            edge_types,
            edge_type_map,
            episodes,
            driver=None,
        ):
            captured['driver'] = driver
            return ([], [], [], {})

        raw_episodes = [
            RawEpisode(
                name='ep1',
                content='Alice knows Bob',
                source=EpisodeType.message,
                source_description='chat',
                reference_time=NOW,
            )
        ]

        with (
            _patch_add_episode_bulk_internals(graphiti),
            patch.object(graphiti, '_resolve_nodes_and_edges_bulk', capture_resolve_all),
        ):
            await graphiti.add_episode_bulk(
                bulk_episodes=raw_episodes,
                group_id='tenant_db',
            )

        assert captured.get('driver') is not None
        assert captured['driver']._database == 'tenant_db'
        assert captured['driver'] is not driver


# ---------------------------------------------------------------------------
# Integration tests – require real FalkorDB
# ---------------------------------------------------------------------------

try:
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    HAS_FALKORDB = True
except ImportError:
    FalkorDriver = None  # type: ignore[assignment,misc]
    HAS_FALKORDB = False

FALKORDB_HOST = os.getenv('FALKORDB_HOST', 'localhost')
FALKORDB_PORT = os.getenv('FALKORDB_PORT', '6379')


class TestGroupIdIsolationIntegration:
    """Integration tests that require a live FalkorDB instance."""

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_add_episode_does_not_mutate_driver_int(self):
        """
        add_episode with a different group_id must leave the original driver's
        _database unchanged even when a real FalkorDB driver is used.
        """
        pytest.importorskip('falkordb')

        try:
            real_driver = FalkorDriver(
                host=FALKORDB_HOST, port=FALKORDB_PORT, database='default_db'
            )
        except Exception as exc:
            pytest.skip(f'FalkorDB not available: {exc}')

        original_db = real_driver._database
        graphiti = _make_graphiti(real_driver)
        episode = _make_episode(group_id='isolation_test_db')

        with _patch_add_episode_internals(graphiti, episode):
            await graphiti.add_episode(
                name='test',
                episode_body='test body',
                reference_time=NOW,
                source=EpisodeType.message,
                source_description='test',
                group_id='isolation_test_db',
            )

        assert graphiti.driver._database == original_db
        assert graphiti.clients.driver._database == original_db

        await real_driver.close()

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_clone_initializes_new_graph_exactly_once_int(self):
        """
        When add_episode is called with a new group_id, the cloned FalkorDriver
        must call build_indices_and_constraints exactly once for that graph even
        across repeated calls.
        """
        pytest.importorskip('falkordb')

        try:
            real_driver = FalkorDriver(
                host=FALKORDB_HOST, port=FALKORDB_PORT, database='default_db'
            )
        except Exception as exc:
            pytest.skip(f'FalkorDB not available: {exc}')

        new_db = 'init_once_test_db'
        assert new_db not in real_driver._initialized_databases

        graphiti = _make_graphiti(real_driver)
        episode = _make_episode(group_id=new_db)

        with _patch_add_episode_internals(graphiti, episode):
            # First call: should initialize new_db
            await graphiti.add_episode(
                name='ep1',
                episode_body='episode body',
                reference_time=NOW,
                source=EpisodeType.message,
                source_description='test',
                group_id=new_db,
            )

            assert new_db in real_driver._initialized_databases

            # Second call: new_db already initialized, build must NOT be called again
            with patch.object(
                real_driver, 'build_indices_and_constraints', new_callable=AsyncMock
            ) as mock_build:
                await graphiti.add_episode(
                    name='ep2',
                    episode_body='episode body 2',
                    reference_time=NOW,
                    source=EpisodeType.message,
                    source_description='test',
                    group_id=new_db,
                )
                mock_build.assert_not_awaited()

        await real_driver.close()
