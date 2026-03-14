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

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.decorators import (
    get_parameter_position,
    handle_multiple_group_ids,
    handle_single_group_id,
)
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_config import SearchResults

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_falkor_self(database: str = 'default_db') -> MagicMock:
    """Return a mock `self` that looks like a Graphiti instance with FalkorDB."""
    mock_self = MagicMock()
    mock_self.clients.driver.provider = GraphProvider.FALKORDB
    mock_self.clients.driver._database = database
    mock_self.max_coroutines = None
    return mock_self


def _make_neo4j_self() -> MagicMock:
    mock_self = MagicMock()
    mock_self.clients.driver.provider = GraphProvider.NEO4J
    return mock_self


# ---------------------------------------------------------------------------
# get_parameter_position
# ---------------------------------------------------------------------------


class TestGetParameterPosition:
    def test_returns_index_of_existing_param(self):
        def func(self, a, b, group_ids):
            pass

        assert get_parameter_position(func, 'group_ids') == 3

    def test_returns_none_for_missing_param(self):
        def func(self, a, b):
            pass

        assert get_parameter_position(func, 'group_ids') is None

    def test_returns_zero_for_self(self):
        def func(self, a):
            pass

        assert get_parameter_position(func, 'self') == 0

    def test_returns_correct_index_for_second_param(self):
        def func(self, group_id, b):
            pass

        assert get_parameter_position(func, 'group_id') == 1

    def test_returns_correct_index_when_param_in_middle(self):
        def func(self, a, group_id, c):
            pass

        assert get_parameter_position(func, 'group_id') == 2


# ---------------------------------------------------------------------------
# handle_multiple_group_ids
# ---------------------------------------------------------------------------


class TestHandleMultipleGroupIds:
    @pytest.mark.asyncio
    async def test_non_falkordb_calls_func_normally(self):
        """Non-FalkorDB provider should bypass the splitting logic."""
        mock_self = _make_neo4j_self()

        @handle_multiple_group_ids
        async def func(self, group_ids=None, driver=None):
            return ['result']

        result = await func(mock_self, group_ids=['g1', 'g2'])
        assert result == ['result']

    @pytest.mark.asyncio
    async def test_falkordb_no_group_ids_calls_func_normally(self):
        """FalkorDB with no group_ids should bypass the splitting logic."""
        mock_self = _make_falkor_self()

        @handle_multiple_group_ids
        async def func(self, group_ids=None, driver=None):
            return ['result']

        result = await func(mock_self, group_ids=None)
        assert result == ['result']

    @pytest.mark.asyncio
    async def test_falkordb_empty_group_ids_calls_func_normally(self):
        mock_self = _make_falkor_self()

        @handle_multiple_group_ids
        async def func(self, group_ids=None, driver=None):
            return ['result']

        result = await func(mock_self, group_ids=[])
        assert result == ['result']

    @pytest.mark.asyncio
    async def test_falkordb_single_group_id_clones_driver_and_injects(self):
        """Single group_id triggers clone + ensure_database_initialized."""
        mock_self = _make_falkor_self()
        cloned_driver = AsyncMock()
        mock_self.clients.driver.clone.return_value = cloned_driver

        received_driver = None

        @handle_multiple_group_ids
        async def func(self, group_ids=None, driver=None):
            nonlocal received_driver
            received_driver = driver
            return ['result']

        result = await func(mock_self, group_ids=['g1'])

        mock_self.clients.driver.clone.assert_called_once_with(database='g1')
        cloned_driver.ensure_database_initialized.assert_awaited_once()
        assert received_driver is cloned_driver
        assert result == ['result']

    @pytest.mark.asyncio
    async def test_falkordb_multiple_group_ids_merges_list_results(self):
        """Multiple group_ids: list results are concatenated."""
        mock_self = _make_falkor_self()

        call_count = 0

        def make_clone(database):
            cloned = AsyncMock()
            return cloned

        mock_self.clients.driver.clone.side_effect = make_clone

        @handle_multiple_group_ids
        async def func(self, group_ids=None, driver=None):
            nonlocal call_count
            call_count += 1
            return [f'item_{call_count}']

        result = await func(mock_self, group_ids=['g1', 'g2'])

        assert call_count == 2
        assert set(result) == {'item_1', 'item_2'}

    @pytest.mark.asyncio
    async def test_falkordb_multiple_group_ids_merges_search_results(self):
        """Multiple group_ids: SearchResults are merged."""
        mock_self = _make_falkor_self()
        mock_self.clients.driver.clone.return_value = AsyncMock()

        from graphiti_core.nodes import EntityNode

        node1 = MagicMock(spec=EntityNode)
        node2 = MagicMock(spec=EntityNode)
        sr1 = SearchResults(nodes=[node1], node_reranker_scores=[0.9])
        sr2 = SearchResults(nodes=[node2], node_reranker_scores=[0.8])

        calls = iter([sr1, sr2])

        @handle_multiple_group_ids
        async def func(self, group_ids=None, driver=None):
            return next(calls)

        result = await func(mock_self, group_ids=['g1', 'g2'])

        assert isinstance(result, SearchResults)
        assert len(result.nodes) == 2

    @pytest.mark.asyncio
    async def test_falkordb_multiple_group_ids_merges_tuple_results(self):
        """Multiple group_ids: tuple results have each component merged."""
        mock_self = _make_falkor_self()
        mock_self.clients.driver.clone.return_value = AsyncMock()

        calls = iter([(['node_a'], ['edge_a']), (['node_b'], ['edge_b'])])

        @handle_multiple_group_ids
        async def func(self, group_ids=None, driver=None):
            return next(calls)

        result = await func(mock_self, group_ids=['g1', 'g2'])

        assert isinstance(result, tuple)
        assert set(result[0]) == {'node_a', 'node_b'}
        assert set(result[1]) == {'edge_a', 'edge_b'}

    @pytest.mark.asyncio
    async def test_falkordb_group_ids_passed_positionally(self):
        """group_ids can be supplied as a positional argument."""
        mock_self = _make_falkor_self()
        cloned_driver = AsyncMock()
        mock_self.clients.driver.clone.return_value = cloned_driver

        received_group_ids = None

        @handle_multiple_group_ids
        async def func(self, other_arg, group_ids=None, driver=None):
            nonlocal received_group_ids
            received_group_ids = group_ids
            return ['result']

        # Pass group_ids as the second positional arg (after self)
        result = await func(mock_self, 'arg_value', ['g1'])

        mock_self.clients.driver.clone.assert_called_once_with(database='g1')
        assert received_group_ids == ['g1']
        assert result == ['result']

    @pytest.mark.asyncio
    async def test_each_group_id_gets_its_own_clone(self):
        """Each group_id gets a separately cloned driver."""
        mock_self = _make_falkor_self()

        clones_created = []

        def make_clone(database):
            cloned = AsyncMock()
            clones_created.append(database)
            return cloned

        mock_self.clients.driver.clone.side_effect = make_clone

        @handle_multiple_group_ids
        async def func(self, group_ids=None, driver=None):
            return ['x']

        await func(mock_self, group_ids=['g1', 'g2', 'g3'])

        assert sorted(clones_created) == ['g1', 'g2', 'g3']


# ---------------------------------------------------------------------------
# handle_single_group_id
# ---------------------------------------------------------------------------


class TestHandleSingleGroupId:
    @pytest.mark.asyncio
    async def test_non_falkordb_calls_func_normally(self):
        mock_self = _make_neo4j_self()

        @handle_single_group_id
        async def func(self, group_id=None, driver=None):
            return 'result'

        result = await func(mock_self, group_id='g1')
        assert result == 'result'

    @pytest.mark.asyncio
    async def test_no_group_id_calls_func_normally(self):
        mock_self = _make_falkor_self()

        @handle_single_group_id
        async def func(self, group_id=None, driver=None):
            return 'result'

        result = await func(mock_self, group_id=None)
        assert result == 'result'
        mock_self.clients.driver.clone.assert_not_called()

    @pytest.mark.asyncio
    async def test_group_id_matches_current_database_no_clone(self):
        """If group_id == current database, no cloning should happen."""
        mock_self = _make_falkor_self(database='my_db')

        @handle_single_group_id
        async def func(self, group_id=None, driver=None):
            return 'result'

        result = await func(mock_self, group_id='my_db')
        assert result == 'result'
        mock_self.clients.driver.clone.assert_not_called()

    @pytest.mark.asyncio
    async def test_falkordb_different_group_id_clones_and_injects_driver(self):
        """Different group_id: driver is cloned and injected as kwarg."""
        mock_self = _make_falkor_self(database='default_db')
        cloned_driver = AsyncMock()
        mock_self.clients.driver.clone.return_value = cloned_driver

        received_driver = None

        @handle_single_group_id
        async def func(self, group_id=None, driver=None):
            nonlocal received_driver
            received_driver = driver
            return 'result'

        result = await func(mock_self, group_id='tenant_db')

        mock_self.clients.driver.clone.assert_called_once_with(database='tenant_db')
        cloned_driver.ensure_database_initialized.assert_awaited_once()
        assert received_driver is cloned_driver
        assert result == 'result'

    @pytest.mark.asyncio
    async def test_group_id_passed_positionally(self):
        """group_id can be supplied as a positional argument."""
        mock_self = _make_falkor_self(database='default_db')
        cloned_driver = AsyncMock()
        mock_self.clients.driver.clone.return_value = cloned_driver

        @handle_single_group_id
        async def func(self, name, group_id=None, driver=None):
            return 'result'

        # group_id is the second positional arg (after self)
        await func(mock_self, 'episode_name', 'tenant_db')

        mock_self.clients.driver.clone.assert_called_once_with(database='tenant_db')

    @pytest.mark.asyncio
    async def test_ensure_database_initialized_awaited(self):
        """ensure_database_initialized must be awaited before calling func."""
        mock_self = _make_falkor_self(database='default_db')
        init_order = []

        cloned_driver = AsyncMock()

        async def record_init():
            init_order.append('init')

        cloned_driver.ensure_database_initialized.side_effect = record_init
        mock_self.clients.driver.clone.return_value = cloned_driver

        @handle_single_group_id
        async def func(self, group_id=None, driver=None):
            init_order.append('func')
            return 'result'

        await func(mock_self, group_id='other_db')
        assert init_order == ['init', 'func']
