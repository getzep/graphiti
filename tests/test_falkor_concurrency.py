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

Tests for FalkorDB concurrency / group isolation.

These tests cover the fix for the race condition where concurrent
``add_episode`` calls with different ``group_id`` values could interleave
and end up writing to the wrong graph because of shared mutable
``self.driver`` state.

The tests are written so they don't need a live FalkorDB instance: the
client/connection is mocked, and we patch the heavy code paths
(``build_indices_and_constraints`` for the driver tests, and the bulk
write helper / LLM extraction helpers for the Graphiti tests) to keep
the focus on driver routing and group isolation.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.nodes import EntityNode

pytest_plugins = ('pytest_asyncio',)


def _make_falkor_driver(database: str = 'default_db') -> FalkorDriver:
    """Construct a FalkorDriver with a mocked underlying client.

    We bypass the real ``FalkorDB`` connection by passing a ``MagicMock``
    as ``falkor_db``. The driver only needs the object to be present for
    ``__init__`` to succeed; the tests never exercise the real Redis
    connection because the heavy paths are patched.
    """
    fake_client = MagicMock()
    return FalkorDriver(falkor_db=fake_client, database=database)


class _MockLLMClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(LLMConfig())
        self._mock_generate = AsyncMock(return_value={'edges': []})

    async def _generate_response(
        self, messages, response_model=None, max_tokens=100, model_size=None
    ):
        return await self._mock_generate(messages, response_model, max_tokens, model_size)


class _MockEmbedderClient(EmbedderClient):
    def __init__(self) -> None:
        self._mock_create = AsyncMock(return_value=[0.1] * 1536)

    async def create(self, input_data):
        return await self._mock_create(input_data)

    async def create_batch(self, input_data_list):
        return [[0.1] * 1536 for _ in input_data_list]


@pytest.mark.asyncio
async def test_clone_does_not_trigger_init():
    """``clone()`` must not run ``build_indices_and_constraints`` on the
    new database. Index building is deferred to
    ``ensure_database_initialized``.
    """
    driver = _make_falkor_driver(database='base')

    with patch.object(
        FalkorDriver, 'build_indices_and_constraints', new_callable=AsyncMock
    ) as build:
        cloned = driver.clone(database='other_db')

        # Sanity check: clone returned a different object pointed at the new DB.
        assert cloned is not driver
        assert cloned._database == 'other_db'
        # The fix: cloning must NOT call build_indices_and_constraints.
        assert build.call_count == 0


@pytest.mark.asyncio
async def test_ensure_database_initialized_idempotent():
    """``ensure_database_initialized`` should call
    ``build_indices_and_constraints`` exactly once per database, even
    across repeated calls and across clones that share the
    ``_initialized_databases`` set.
    """
    driver = _make_falkor_driver(database='base')

    with patch.object(
        FalkorDriver, 'build_indices_and_constraints', new_callable=AsyncMock
    ) as build:
        cloned = driver.clone(database='group_x')

        # Clear out any side-effects from __init__ tracking 'base'.
        driver._initialized_databases.discard('group_x')

        await cloned.ensure_database_initialized()
        await cloned.ensure_database_initialized()

        # Second call should be a no-op for the same database.
        assert build.call_count == 1
        assert 'group_x' in cloned._initialized_databases
        # The set is shared via shallow copy.
        assert 'group_x' in driver._initialized_databases


@pytest.mark.asyncio
async def test_concurrent_add_episode_isolates_drivers():
    """Two concurrent ``add_episode`` calls with different ``group_id``
    values must each see a driver targeted at their own database. With
    the old buggy code, Task A would resume to find ``self.driver``
    pointing at Task B's database.
    """
    driver = _make_falkor_driver(database='default_db')
    # Pre-mark all relevant DBs as initialized so ensure_database_initialized
    # is a no-op (no real Redis available in this test).
    driver._initialized_databases.update({'default_db', 'group_a', 'group_b'})

    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=_MockLLMClient(),
        embedder=_MockEmbedderClient(),
    )

    # Capture the driver's _database at the moment of the bulk write.
    captured_databases: list[str] = []
    capture_lock = asyncio.Lock()

    async def capture_database(driver_arg, *args, **kwargs):
        async with capture_lock:
            captured_databases.append(driver_arg._database)

    # Task A waits for Task B to start before completing extraction so we
    # exercise the interleaving path.
    task_a_in_extract = asyncio.Event()
    task_b_done = asyncio.Event()

    async def mocked_extract_nodes(
        clients, episode, previous_episodes, entity_types, excluded_entity_types,
        custom_extraction_instructions=None,
    ):
        if episode.group_id == 'group_a':
            task_a_in_extract.set()
            await task_b_done.wait()
        nodes = [
            EntityNode(
                name=f'Node_{episode.group_id}',
                labels=['Entity'],
                group_id=episode.group_id,
            )
        ]
        return nodes, {}

    async def mocked_resolve_nodes(clients, extracted_nodes, episode, previous_episodes, entity_types):
        return extracted_nodes, {n.uuid: n.uuid for n in extracted_nodes}, []

    async def mocked_extract_attributes(
        clients, nodes, episode, previous_episodes, entity_types, edges=None,
    ):
        return nodes

    async def mocked_extract_resolve_edges(*args, **kwargs):
        return [], [], []

    async def mocked_retrieve_episodes(*args, **kwargs):
        return []

    import graphiti_core.graphiti as graphiti_module

    with (
        patch.object(graphiti_module, 'extract_nodes', side_effect=mocked_extract_nodes),
        patch.object(graphiti_module, 'resolve_extracted_nodes', side_effect=mocked_resolve_nodes),
        patch.object(
            graphiti_module, 'extract_attributes_from_nodes', side_effect=mocked_extract_attributes
        ),
        patch.object(graphiti_module, 'add_nodes_and_edges_bulk', side_effect=capture_database),
        patch.object(graphiti_module, 'retrieve_episodes', side_effect=mocked_retrieve_episodes),
        patch.object(
            Graphiti,
            '_extract_and_resolve_edges',
            new=AsyncMock(side_effect=mocked_extract_resolve_edges),
        ),
    ):

        async def run_a():
            await graphiti.add_episode(
                name='Episode A',
                episode_body='Content A',
                source_description='Source A',
                reference_time=datetime.now(timezone.utc),
                group_id='group_a',
            )

        async def run_b():
            await task_a_in_extract.wait()
            try:
                await graphiti.add_episode(
                    name='Episode B',
                    episode_body='Content B',
                    source_description='Source B',
                    reference_time=datetime.now(timezone.utc),
                    group_id='group_b',
                )
            finally:
                task_b_done.set()

        await asyncio.gather(run_a(), run_b())

    # Each task should have written to its own database. With the bug,
    # Task A would have written to 'group_b' (or vice versa).
    assert sorted(captured_databases) == ['group_a', 'group_b'], (
        f'Cross-group leak detected: writes went to {captured_databases}'
    )
