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

# Regression tests for concurrent multi-group_id database routing
# (issues #1676 / #1876).
#
# These tests are database-free: they exercise ``Graphiti._resolve_request_scope``
# and the read-path helper ``_resolve_read_scope`` with a fake driver. The
# historical write bug (#1676) was that ``add_episode`` / ``add_episode_bulk``
# reassigned the shared ``self.driver`` when a ``group_id`` mapped to a different
# database. Because those coroutines have many ``await`` points, a concurrent
# call for a different ``group_id`` could reassign ``self.driver`` mid-execution
# and silently persist episodes under the wrong graph.
#
# The matching read bug (#1876) was that ``search`` / ``search_`` /
# ``retrieve_episodes`` kept the configured driver, so a write to the cloned
# tenant database was invisible to the subsequent read. Both paths now return a
# request-scoped driver/clients bundle instead of mutating shared instance state.

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graphiti import Graphiti
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.llm_client import LLMClient
from graphiti_core.search.search_config import SearchResults
from graphiti_core.tracer import Tracer

pytest_plugins = ('pytest_asyncio',)


class FakeDriver(GraphDriver):
    """Minimal in-memory GraphDriver whose ``clone`` records the target database.

    Each ``clone`` returns a brand new instance bound to the requested database,
    mirroring how the real drivers hand back a call-scoped copy without mutating
    the original.
    """

    provider = GraphProvider.NEO4J

    def __init__(self, database: str = 'default_db'):
        self._database = database
        self.clone_calls: list[str] = []

    def clone(self, database: str) -> 'FakeDriver':
        self.clone_calls.append(database)
        cloned = FakeDriver(database=database)
        return cloned

    # --- Abstract methods: unused by these tests, kept as no-ops. ---
    async def execute_query(self, cypher_query_: str, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def session(self, database: str | None = None):  # pragma: no cover
        raise NotImplementedError

    def close(self):  # pragma: no cover
        raise NotImplementedError

    def delete_all_indexes(self):  # pragma: no cover
        raise NotImplementedError

    async def build_indices_and_constraints(
        self, delete_existing: bool = False
    ):  # pragma: no cover
        raise NotImplementedError


def _make_graphiti(database: str = 'default_db') -> tuple[Graphiti, FakeDriver]:
    """Build a Graphiti instance around a FakeDriver, bypassing __init__ side effects."""
    driver = FakeDriver(database=database)
    clients = GraphitiClients(  # type: ignore[call-arg]
        driver=driver,
        llm_client=Mock(spec=LLMClient),
        embedder=Mock(spec=EmbedderClient),
        cross_encoder=Mock(spec=CrossEncoderClient),
        tracer=Mock(spec=Tracer),
    )
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = driver
    graphiti.clients = clients
    return graphiti, driver


def test_resolve_request_scope_none_group_id_reuses_shared_driver():
    graphiti, driver = _make_graphiti()

    group_id, scoped_driver, scoped_clients = graphiti._resolve_request_scope(None)

    # Default group id for Neo4j is the empty string.
    assert group_id == ''
    # No clone occurred and the shared instances are reused as-is.
    assert scoped_driver is driver
    assert scoped_clients is graphiti.clients
    assert driver.clone_calls == []


def test_resolve_request_scope_matching_group_id_reuses_shared_driver():
    graphiti, driver = _make_graphiti(database='tenant_a')

    group_id, scoped_driver, scoped_clients = graphiti._resolve_request_scope('tenant_a')

    assert group_id == 'tenant_a'
    assert scoped_driver is driver
    assert scoped_clients is graphiti.clients
    assert driver.clone_calls == []


def test_resolve_request_scope_different_group_id_does_not_mutate_shared_state():
    graphiti, driver = _make_graphiti(database='default_db')
    original_clients = graphiti.clients

    group_id, scoped_driver, scoped_clients = graphiti._resolve_request_scope('tenant_b')

    # A request-scoped clone targeting the requested database is returned.
    assert group_id == 'tenant_b'
    assert scoped_driver is not driver
    assert scoped_driver._database == 'tenant_b'
    assert driver.clone_calls == ['tenant_b']

    # The scoped clients bundle points at the clone but preserves the other clients.
    assert scoped_clients is not original_clients
    assert scoped_clients.driver is scoped_driver
    assert scoped_clients.llm_client is original_clients.llm_client
    assert scoped_clients.embedder is original_clients.embedder
    assert scoped_clients.cross_encoder is original_clients.cross_encoder

    # Crucially, the shared instance state is untouched (this is the #1676 fix).
    assert graphiti.driver is driver
    assert graphiti.driver._database == 'default_db'
    assert graphiti.clients is original_clients
    assert graphiti.clients.driver is driver


@pytest.mark.asyncio
async def test_concurrent_group_ids_keep_independent_drivers():
    """Two interleaved calls for different group_ids must not clobber each other.

    Each worker resolves its request scope, then yields control (simulating the
    many ``await`` points inside ``add_episode``) while the other worker runs.
    With the pre-fix code that reassigned ``self.driver``, the first worker's
    driver would be silently repointed at the second worker's database. The
    request-scoped bundle keeps them isolated.
    """
    graphiti, base_driver = _make_graphiti(database='default_db')

    observations: dict[str, str] = {}

    async def worker(gid: str, hold: float):
        resolved_gid, scoped_driver, scoped_clients = graphiti._resolve_request_scope(gid)
        # Yield control so a concurrent worker interleaves here.
        await asyncio.sleep(hold)
        # After the interleave, the scoped driver must still target this worker's db.
        assert scoped_driver._database == gid
        assert scoped_clients.driver._database == gid
        # And the shared driver must never have been mutated to this or any group.
        assert graphiti.driver is base_driver
        assert graphiti.driver._database == 'default_db'
        observations[gid] = scoped_driver._database

    await asyncio.gather(
        worker('tenant_a', hold=0.02),
        worker('tenant_b', hold=0.0),
        worker('tenant_c', hold=0.01),
    )

    assert observations == {
        'tenant_a': 'tenant_a',
        'tenant_b': 'tenant_b',
        'tenant_c': 'tenant_c',
    }
    # Shared driver still points at the original default database.
    assert graphiti.driver._database == 'default_db'
    assert graphiti.clients.driver._database == 'default_db'


def test_resolve_read_scope_none_group_ids_reuses_shared_driver():
    graphiti, driver = _make_graphiti()

    scoped_driver, scoped_clients = graphiti._resolve_read_scope(None)

    assert scoped_driver is driver
    assert scoped_clients is graphiti.clients
    assert driver.clone_calls == []


def test_resolve_read_scope_matching_group_id_reuses_shared_driver():
    graphiti, driver = _make_graphiti(database='tenant_a')

    scoped_driver, scoped_clients = graphiti._resolve_read_scope(['tenant_a'])

    assert scoped_driver is driver
    assert scoped_clients is graphiti.clients
    assert driver.clone_calls == []


def test_resolve_read_scope_single_group_id_does_not_mutate_shared_state():
    graphiti, driver = _make_graphiti(database='default_db')
    original_clients = graphiti.clients

    scoped_driver, scoped_clients = graphiti._resolve_read_scope(['tenant_b'])

    assert scoped_driver is not driver
    assert scoped_driver._database == 'tenant_b'
    assert driver.clone_calls == ['tenant_b']
    assert scoped_clients is not original_clients
    assert scoped_clients.driver is scoped_driver

    assert graphiti.driver is driver
    assert graphiti.driver._database == 'default_db'
    assert graphiti.clients is original_clients
    assert graphiti.clients.driver is driver


def test_resolve_read_scope_multiple_group_ids_reuses_shared_driver():
    """Multi-group reads stay on the shared driver; FalkorDB fans out elsewhere."""
    graphiti, driver = _make_graphiti()

    scoped_driver, scoped_clients = graphiti._resolve_read_scope(['tenant_a', 'tenant_b'])

    assert scoped_driver is driver
    assert scoped_clients is graphiti.clients
    assert driver.clone_calls == []


def test_resolve_read_scope_explicit_driver_is_honored():
    graphiti, driver = _make_graphiti()
    explicit = FakeDriver(database='already_scoped')

    scoped_driver, scoped_clients = graphiti._resolve_read_scope(['tenant_b'], driver=explicit)

    assert scoped_driver is explicit
    assert scoped_driver._database == 'already_scoped'
    assert scoped_clients is not graphiti.clients
    assert scoped_clients.driver is explicit
    # Must not clone the shared driver when a caller-supplied driver is present.
    assert driver.clone_calls == []
    assert graphiti.driver is driver
    assert graphiti.driver._database == 'default_db'


def test_resolve_read_scope_explicit_shared_driver_reuses_clients():
    graphiti, driver = _make_graphiti()

    scoped_driver, scoped_clients = graphiti._resolve_read_scope(['tenant_b'], driver=driver)

    assert scoped_driver is driver
    assert scoped_clients is graphiti.clients
    assert driver.clone_calls == []


@pytest.mark.asyncio
async def test_retrieve_episodes_single_group_id_uses_scoped_driver():
    graphiti, driver = _make_graphiti()
    reference_time = datetime.now(timezone.utc)

    with patch('graphiti_core.graphiti.retrieve_episodes', new_callable=AsyncMock) as mock_retrieve:
        mock_retrieve.return_value = []
        await graphiti.retrieve_episodes(reference_time, group_ids=['tenant_b'])

    used_driver = mock_retrieve.call_args.args[0]
    assert used_driver is not driver
    assert used_driver._database == 'tenant_b'
    assert driver.clone_calls == ['tenant_b']
    assert graphiti.driver is driver
    assert graphiti.driver._database == 'default_db'


@pytest.mark.asyncio
async def test_retrieve_episodes_explicit_driver_is_not_recloned():
    graphiti, driver = _make_graphiti()
    explicit = FakeDriver(database='already_scoped')
    reference_time = datetime.now(timezone.utc)

    with patch('graphiti_core.graphiti.retrieve_episodes', new_callable=AsyncMock) as mock_retrieve:
        mock_retrieve.return_value = []
        await graphiti.retrieve_episodes(reference_time, group_ids=['tenant_b'], driver=explicit)

    used_driver = mock_retrieve.call_args.args[0]
    assert used_driver is explicit
    assert driver.clone_calls == []
    assert graphiti.driver is driver


@pytest.mark.asyncio
async def test_search_single_group_id_uses_scoped_driver():
    graphiti, driver = _make_graphiti()
    captured: dict[str, Any] = {}

    async def fake_search(clients, query, group_ids, config, search_filter, **kwargs):
        captured['driver'] = kwargs.get('driver')
        captured['clients_driver'] = clients.driver
        captured['query'] = query
        captured['group_ids'] = group_ids
        return SearchResults()

    with patch('graphiti_core.graphiti.search', side_effect=fake_search):
        result = await graphiti.search('hello', group_ids=['tenant_b'])

    assert result == []
    assert captured['query'] == 'hello'
    assert captured['group_ids'] == ['tenant_b']
    assert captured['driver'] is not driver
    assert captured['driver']._database == 'tenant_b'
    assert captured['clients_driver']._database == 'tenant_b'
    assert driver.clone_calls == ['tenant_b']
    assert graphiti.driver is driver
    assert graphiti.driver._database == 'default_db'
    assert graphiti.clients.driver is driver


@pytest.mark.asyncio
async def test_search_underscore_single_group_id_uses_scoped_driver():
    graphiti, driver = _make_graphiti()
    captured: dict[str, Any] = {}

    async def fake_search(clients, query, group_ids, config, search_filter, *args, **kwargs):
        captured['driver'] = kwargs.get('driver')
        captured['clients_driver'] = clients.driver
        return SearchResults()

    with patch('graphiti_core.graphiti.search', side_effect=fake_search):
        result = await graphiti.search_('hello', group_ids=['tenant_b'])

    assert isinstance(result, SearchResults)
    assert captured['driver'] is not driver
    assert captured['driver']._database == 'tenant_b'
    assert captured['clients_driver']._database == 'tenant_b'
    assert graphiti.driver is driver
    assert graphiti.driver._database == 'default_db'


@pytest.mark.asyncio
async def test_search_matching_group_id_reuses_shared_driver():
    graphiti, driver = _make_graphiti(database='tenant_a')
    captured: dict[str, Any] = {}

    async def fake_search(clients, query, group_ids, config, search_filter, **kwargs):
        captured['driver'] = kwargs.get('driver')
        captured['clients'] = clients
        return SearchResults()

    with patch('graphiti_core.graphiti.search', side_effect=fake_search):
        await graphiti.search('hello', group_ids=['tenant_a'])

    assert captured['driver'] is driver
    assert captured['clients'] is graphiti.clients
    assert driver.clone_calls == []


@pytest.mark.asyncio
async def test_concurrent_reads_keep_independent_drivers():
    """Interleaved reads for different group_ids must not clobber each other."""
    graphiti, base_driver = _make_graphiti(database='default_db')
    observations: dict[str, str] = {}

    async def worker(gid: str, hold: float):
        scoped_driver, scoped_clients = graphiti._resolve_read_scope([gid])
        await asyncio.sleep(hold)
        assert scoped_driver._database == gid
        assert scoped_clients.driver._database == gid
        assert graphiti.driver is base_driver
        assert graphiti.driver._database == 'default_db'
        observations[gid] = scoped_driver._database

    await asyncio.gather(
        worker('tenant_a', hold=0.02),
        worker('tenant_b', hold=0.0),
        worker('tenant_c', hold=0.01),
    )

    assert observations == {
        'tenant_a': 'tenant_a',
        'tenant_b': 'tenant_b',
        'tenant_c': 'tenant_c',
    }
    assert graphiti.driver._database == 'default_db'
    assert graphiti.clients.driver._database == 'default_db'
