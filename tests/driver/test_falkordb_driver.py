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

import os
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.driver.driver import GraphProvider

try:
    from graphiti_core.driver.falkordb_driver import FalkorDriver, FalkorDriverSession

    HAS_FALKORDB = True
except ImportError:
    FalkorDriver = None
    HAS_FALKORDB = False


class TestFalkorDriver:
    """Comprehensive test suite for FalkorDB driver."""

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        with patch('graphiti_core.driver.falkordb_driver.FalkorDB'):
            self.driver = FalkorDriver()
        self.driver.client = self.mock_client

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_init_with_connection_params(self):
        """Test initialization with connection parameters."""
        with patch('graphiti_core.driver.falkordb_driver.FalkorDB') as mock_falkor_db:
            driver = FalkorDriver(
                host='test-host', port='1234', username='test-user', password='test-pass'
            )
            assert driver.provider == GraphProvider.FALKORDB
            mock_falkor_db.assert_called_once_with(
                host='test-host', port='1234', username='test-user', password='test-pass'
            )

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_init_with_falkor_db_instance(self):
        """Test initialization with a FalkorDB instance."""
        with patch('graphiti_core.driver.falkordb_driver.FalkorDB') as mock_falkor_db_class:
            mock_falkor_db = MagicMock()
            driver = FalkorDriver(falkor_db=mock_falkor_db)
            assert driver.provider == GraphProvider.FALKORDB
            assert driver.client is mock_falkor_db
            mock_falkor_db_class.assert_not_called()

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_provider(self):
        """Test driver provider identification."""
        assert self.driver.provider == GraphProvider.FALKORDB

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_get_graph_with_name(self):
        """Test _get_graph with specific graph name."""
        mock_graph = MagicMock()
        self.mock_client.select_graph.return_value = mock_graph

        result = self.driver._get_graph('test_graph')

        self.mock_client.select_graph.assert_called_once_with('test_graph')
        assert result is mock_graph

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_get_graph_with_none_defaults_to_default_database(self):
        """Test _get_graph with None defaults to default_db."""
        mock_graph = MagicMock()
        self.mock_client.select_graph.return_value = mock_graph

        result = self.driver._get_graph(None)

        self.mock_client.select_graph.assert_called_once_with('default_db')
        assert result is mock_graph

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_execute_query_success(self):
        """Test successful query execution."""
        mock_graph = MagicMock()
        mock_result = MagicMock()
        mock_result.header = [('col1', 'column1'), ('col2', 'column2')]
        mock_result.result_set = [['row1col1', 'row1col2']]
        mock_graph.query = AsyncMock(return_value=mock_result)
        self.mock_client.select_graph.return_value = mock_graph

        result = await self.driver.execute_query('MATCH (n) RETURN n', param1='value1')

        mock_graph.query.assert_called_once_with('MATCH (n) RETURN n', {'param1': 'value1'})

        result_set, header, summary = result
        assert result_set == [{'column1': 'row1col1', 'column2': 'row1col2'}]
        assert header == ['column1', 'column2']
        assert summary is None

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_execute_query_handles_index_already_exists_error(self):
        """Test handling of 'already indexed' error."""
        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(side_effect=Exception('Index already indexed'))
        self.mock_client.select_graph.return_value = mock_graph

        with patch('graphiti_core.driver.falkordb_driver.logger') as mock_logger:
            result = await self.driver.execute_query('CREATE INDEX ...')

            mock_logger.info.assert_called_once()
            assert result is None

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_execute_query_propagates_other_exceptions(self):
        """Test that other exceptions are properly propagated."""
        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(side_effect=Exception('Other error'))
        self.mock_client.select_graph.return_value = mock_graph

        with patch('graphiti_core.driver.falkordb_driver.logger') as mock_logger:
            with pytest.raises(Exception, match='Other error'):
                await self.driver.execute_query('INVALID QUERY')

            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_execute_query_converts_datetime_parameters(self):
        """Test that datetime objects in kwargs are converted to ISO strings."""
        mock_graph = MagicMock()
        mock_result = MagicMock()
        mock_result.header = []
        mock_result.result_set = []
        mock_graph.query = AsyncMock(return_value=mock_result)
        self.mock_client.select_graph.return_value = mock_graph

        test_datetime = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        await self.driver.execute_query(
            'CREATE (n:Node) SET n.created_at = $created_at', created_at=test_datetime
        )

        call_args = mock_graph.query.call_args[0]
        assert call_args[1]['created_at'] == test_datetime.isoformat()

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_session_creation(self):
        """Test session creation with specific database."""
        mock_graph = MagicMock()
        self.mock_client.select_graph.return_value = mock_graph

        session = self.driver.session()

        assert isinstance(session, FalkorDriverSession)
        assert session.graph is mock_graph

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_session_creation_with_none_uses_default_database(self):
        """Test session creation with None uses default database."""
        mock_graph = MagicMock()
        self.mock_client.select_graph.return_value = mock_graph

        session = self.driver.session()

        assert isinstance(session, FalkorDriverSession)

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_close_calls_connection_close(self):
        """Test driver close method calls connection close."""
        mock_connection = MagicMock()
        mock_connection.close = AsyncMock()
        self.mock_client.connection = mock_connection

        # Ensure hasattr checks work correctly
        del self.mock_client.aclose  # Remove aclose if it exists

        with patch('builtins.hasattr') as mock_hasattr:
            # hasattr(self.client, 'aclose') returns False
            # hasattr(self.client.connection, 'aclose') returns False
            # hasattr(self.client.connection, 'close') returns True
            mock_hasattr.side_effect = lambda obj, attr: (
                attr == 'close' and obj is mock_connection
            )

            await self.driver.close()

        mock_connection.close.assert_called_once()

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_delete_all_indexes(self):
        """Test delete_all_indexes method."""
        with patch.object(self.driver, 'execute_query', new_callable=AsyncMock) as mock_execute:
            # Return None to simulate no indexes found
            mock_execute.return_value = None

            await self.driver.delete_all_indexes()

            mock_execute.assert_called_once_with('CALL db.indexes()')


class TestFalkorDriverSession:
    """Test FalkorDB driver session functionality."""

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_graph = MagicMock()
        self.session = FalkorDriverSession(self.mock_graph)

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_session_async_context_manager(self):
        """Test session can be used as async context manager."""
        async with self.session as s:
            assert s is self.session

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_close_method(self):
        """Test session close method doesn't raise exceptions."""
        await self.session.close()  # Should not raise

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_execute_write_passes_session_and_args(self):
        """Test execute_write method passes session and arguments correctly."""

        async def test_func(session, *args, **kwargs):
            assert session is self.session
            assert args == ('arg1', 'arg2')
            assert kwargs == {'key': 'value'}
            return 'result'

        result = await self.session.execute_write(test_func, 'arg1', 'arg2', key='value')
        assert result == 'result'

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_run_single_query_with_parameters(self):
        """Test running a single query with parameters."""
        self.mock_graph.query = AsyncMock()

        await self.session.run('MATCH (n) RETURN n', param1='value1', param2='value2')

        self.mock_graph.query.assert_called_once_with(
            'MATCH (n) RETURN n', {'param1': 'value1', 'param2': 'value2'}
        )

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_run_multiple_queries_as_list(self):
        """Test running multiple queries passed as list."""
        self.mock_graph.query = AsyncMock()

        queries = [
            ('MATCH (n) RETURN n', {'param1': 'value1'}),
            ('CREATE (n:Node)', {'param2': 'value2'}),
        ]

        await self.session.run(queries)

        assert self.mock_graph.query.call_count == 2
        calls = self.mock_graph.query.call_args_list
        assert calls[0][0] == ('MATCH (n) RETURN n', {'param1': 'value1'})
        assert calls[1][0] == ('CREATE (n:Node)', {'param2': 'value2'})

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_run_converts_datetime_objects_to_iso_strings(self):
        """Test that datetime objects are converted to ISO strings."""
        self.mock_graph.query = AsyncMock()
        test_datetime = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        await self.session.run(
            'CREATE (n:Node) SET n.created_at = $created_at', created_at=test_datetime
        )

        self.mock_graph.query.assert_called_once()
        call_args = self.mock_graph.query.call_args[0]
        assert call_args[1]['created_at'] == test_datetime.isoformat()


class TestDatetimeConversion:
    """Test datetime conversion utility function."""

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_convert_datetime_dict(self):
        """Test datetime conversion in nested dictionary."""
        from graphiti_core.driver.falkordb_driver import convert_datetimes_to_strings

        test_datetime = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        input_dict = {
            'string_val': 'test',
            'datetime_val': test_datetime,
            'nested_dict': {'nested_datetime': test_datetime, 'nested_string': 'nested_test'},
        }

        result = convert_datetimes_to_strings(input_dict)

        assert result['string_val'] == 'test'
        assert result['datetime_val'] == test_datetime.isoformat()
        assert result['nested_dict']['nested_datetime'] == test_datetime.isoformat()
        assert result['nested_dict']['nested_string'] == 'nested_test'

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_convert_datetime_list_and_tuple(self):
        """Test datetime conversion in lists and tuples."""
        from graphiti_core.driver.falkordb_driver import convert_datetimes_to_strings

        test_datetime = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Test list
        input_list = ['test', test_datetime, ['nested', test_datetime]]
        result_list = convert_datetimes_to_strings(input_list)
        assert result_list[0] == 'test'
        assert result_list[1] == test_datetime.isoformat()
        assert result_list[2][1] == test_datetime.isoformat()

        # Test tuple
        input_tuple = ('test', test_datetime)
        result_tuple = convert_datetimes_to_strings(input_tuple)
        assert isinstance(result_tuple, tuple)
        assert result_tuple[0] == 'test'
        assert result_tuple[1] == test_datetime.isoformat()

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_convert_single_datetime(self):
        """Test datetime conversion for single datetime object."""
        from graphiti_core.driver.falkordb_driver import convert_datetimes_to_strings

        test_datetime = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = convert_datetimes_to_strings(test_datetime)
        assert result == test_datetime.isoformat()

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_convert_other_types_unchanged(self):
        """Test that non-datetime types are returned unchanged."""
        from graphiti_core.driver.falkordb_driver import convert_datetimes_to_strings

        assert convert_datetimes_to_strings('string') == 'string'
        assert convert_datetimes_to_strings(123) == 123
        assert convert_datetimes_to_strings(None) is None
        assert convert_datetimes_to_strings(True) is True


class TestFalkorDriverClone:
    """Tests for FalkorDriver.clone() and ensure_database_initialized()."""

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def setup_method(self):
        with patch('graphiti_core.driver.falkordb_driver.FalkorDB'):
            self.driver = FalkorDriver(database='default_db')
        self.driver.client = MagicMock()
        # Mark default_db as already initialized so it's a known baseline
        self.driver._initialized_databases = {'default_db'}

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_clone_same_database_returns_self(self):
        result = self.driver.clone(database='default_db')
        assert result is self.driver

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_clone_different_database_returns_new_driver(self):
        cloned = self.driver.clone(database='tenant_db')
        assert cloned is not self.driver
        assert cloned._database == 'tenant_db'

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_clone_preserves_original_database(self):
        self.driver.clone(database='tenant_db')
        assert self.driver._database == 'default_db'

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_clone_shares_initialized_databases_set(self):
        """Clones share the same _initialized_databases set by reference."""
        cloned = self.driver.clone(database='tenant_db')
        assert cloned._initialized_databases is self.driver._initialized_databases

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_clone_shares_client(self):
        """Clones reuse the same underlying connection."""
        cloned = self.driver.clone(database='tenant_db')
        assert cloned.client is self.driver.client

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_ensure_database_initialized_calls_build_on_first_call(self):
        """First call for a new database triggers build_indices_and_constraints."""
        cloned = self.driver.clone(database='new_db')
        with patch.object(
            cloned, 'build_indices_and_constraints', new_callable=AsyncMock
        ) as mock_build:
            await cloned.ensure_database_initialized()
            mock_build.assert_awaited_once()

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_ensure_database_initialized_skips_on_subsequent_calls(self):
        """Subsequent calls for the same database are no-ops."""
        cloned = self.driver.clone(database='new_db')
        with patch.object(
            cloned, 'build_indices_and_constraints', new_callable=AsyncMock
        ) as mock_build:
            await cloned.ensure_database_initialized()
            await cloned.ensure_database_initialized()
            await cloned.ensure_database_initialized()
            mock_build.assert_awaited_once()

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_ensure_database_initialized_marks_database_as_initialized(self):
        """After init, the database is recorded in _initialized_databases."""
        cloned = self.driver.clone(database='new_db')
        with patch.object(cloned, 'build_indices_and_constraints', new_callable=AsyncMock):
            assert 'new_db' not in cloned._initialized_databases
            await cloned.ensure_database_initialized()
            assert 'new_db' in cloned._initialized_databases

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_ensure_database_initialized_shared_across_clones(self):
        """Init via one clone is visible to sibling clones sharing the same set."""
        clone_a = self.driver.clone(database='shared_db')
        clone_b = self.driver.clone(database='shared_db')
        assert clone_a._initialized_databases is clone_b._initialized_databases

        with (
            patch.object(
                clone_a, 'build_indices_and_constraints', new_callable=AsyncMock
            ) as mock_a,
            patch.object(
                clone_b, 'build_indices_and_constraints', new_callable=AsyncMock
            ) as mock_b,
        ):
            await clone_a.ensure_database_initialized()
            # clone_b should see shared_db already initialized and skip build
            await clone_b.ensure_database_initialized()
            mock_a.assert_awaited_once()
            mock_b.assert_not_awaited()

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_ensure_database_initialized_skips_if_already_in_set(self):
        """Databases pre-populated in _initialized_databases are not re-initialized."""
        with patch.object(
            self.driver, 'build_indices_and_constraints', new_callable=AsyncMock
        ) as mock_build:
            await self.driver.ensure_database_initialized()
            mock_build.assert_not_awaited()


# Simple integration test
class TestFalkorDriverIntegration:
    """Simple integration test for FalkorDB driver."""

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_basic_integration_with_real_falkordb(self):
        """Basic integration test with real FalkorDB instance."""
        pytest.importorskip('falkordb')

        falkor_host = os.getenv('FALKORDB_HOST', 'localhost')
        falkor_port = os.getenv('FALKORDB_PORT', '6379')

        try:
            driver = FalkorDriver(host=falkor_host, port=falkor_port)

            # Test basic query execution
            result = await driver.execute_query('RETURN 1 as test')
            assert result is not None

            result_set, header, summary = result
            assert header == ['test']
            assert result_set == [{'test': 1}]

            await driver.close()

        except Exception as e:
            pytest.skip(f'FalkorDB not available for integration test: {e}')

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_clone_and_ensure_database_initialized_int(self):
        """Integration: cloned driver initialises a fresh graph exactly once."""
        pytest.importorskip('falkordb')

        falkor_host = os.getenv('FALKORDB_HOST', 'localhost')
        falkor_port = os.getenv('FALKORDB_PORT', '6379')

        try:
            driver = FalkorDriver(host=falkor_host, port=falkor_port, database='default_db')

            test_db = 'test_clone_init_db'
            cloned = driver.clone(database=test_db)

            assert cloned is not driver
            assert cloned._database == test_db
            assert cloned._initialized_databases is driver._initialized_databases
            assert test_db not in cloned._initialized_databases

            # First call should build indices
            await cloned.ensure_database_initialized()
            assert test_db in cloned._initialized_databases

            # Second call should be a no-op (no error raised)
            await cloned.ensure_database_initialized()

            await driver.close()

        except Exception as e:
            pytest.skip(f'FalkorDB not available for integration test: {e}')
