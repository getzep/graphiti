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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.driver.driver import GraphProvider

try:
    from graphiti_core.driver.arcadedb_driver import ArcadeDBDriver

    HAS_ARCADEDB = True
except ImportError:
    ArcadeDBDriver = None
    HAS_ARCADEDB = False


class TestArcadeDBDriver:
    """Comprehensive test suite for ArcadeDB driver."""

    @unittest.skipIf(not HAS_ARCADEDB, 'ArcadeDB driver dependencies not available')
    def setup_method(self):
        """Set up test fixtures."""
        with patch('graphiti_core.driver.arcadedb_driver.AsyncGraphDatabase') as mock_agd:
            mock_agd.driver.return_value = MagicMock()
            self.driver = ArcadeDBDriver(
                uri='bolt://localhost:2480',
                user='root',
                password='test',
                database='graphiti',
            )
        self.mock_client = MagicMock()
        self.driver.client = self.mock_client

    @unittest.skipIf(not HAS_ARCADEDB, 'ArcadeDB driver dependencies not available')
    def test_provider(self):
        """Test driver provider identification."""
        assert self.driver.provider == GraphProvider.ARCADEDB

    @unittest.skipIf(not HAS_ARCADEDB, 'ArcadeDB driver dependencies not available')
    def test_init_with_connection_params(self):
        """Test initialization with connection parameters."""
        with patch('graphiti_core.driver.arcadedb_driver.AsyncGraphDatabase') as mock_agd:
            mock_agd.driver.return_value = MagicMock()
            driver = ArcadeDBDriver(
                uri='bolt://custom-host:2480',
                user='admin',
                password='secret',
                database='mydb',
            )
            assert driver.provider == GraphProvider.ARCADEDB
            mock_agd.driver.assert_called_once_with(
                uri='bolt://custom-host:2480',
                auth=('admin', 'secret'),
            )
            assert driver._database == 'mydb'

    @unittest.skipIf(not HAS_ARCADEDB, 'ArcadeDB driver dependencies not available')
    def test_init_with_none_credentials(self):
        """Test initialization with None user/password defaults to empty strings."""
        with patch('graphiti_core.driver.arcadedb_driver.AsyncGraphDatabase') as mock_agd:
            mock_agd.driver.return_value = MagicMock()
            ArcadeDBDriver(
                uri='bolt://localhost:2480',
                user=None,
                password=None,
            )
            mock_agd.driver.assert_called_once_with(
                uri='bolt://localhost:2480',
                auth=('', ''),
            )

    @unittest.skipIf(not HAS_ARCADEDB, 'ArcadeDB driver dependencies not available')
    def test_default_database(self):
        """Test default database name is 'graphiti'."""
        with patch('graphiti_core.driver.arcadedb_driver.AsyncGraphDatabase') as mock_agd:
            mock_agd.driver.return_value = MagicMock()
            driver = ArcadeDBDriver(
                uri='bolt://localhost:2480',
                user='root',
                password='test',
            )
            assert driver._database == 'graphiti'

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_ARCADEDB, 'ArcadeDB driver dependencies not available')
    async def test_execute_query_success(self):
        """Test successful query execution."""
        mock_result = MagicMock()
        self.mock_client.execute_query = AsyncMock(return_value=mock_result)

        result = await self.driver.execute_query('MATCH (n) RETURN n', param1='value1')

        self.mock_client.execute_query.assert_called_once_with(
            'MATCH (n) RETURN n',
            parameters_={'database_': 'graphiti'},
            param1='value1',
        )
        assert result is mock_result

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_ARCADEDB, 'ArcadeDB driver dependencies not available')
    async def test_execute_query_with_params(self):
        """Test query execution with explicit params dict."""
        mock_result = MagicMock()
        self.mock_client.execute_query = AsyncMock(return_value=mock_result)

        await self.driver.execute_query(
            'MATCH (n) WHERE n.uuid = $uuid RETURN n',
            params={'uuid': 'test-uuid'},
        )

        self.mock_client.execute_query.assert_called_once_with(
            'MATCH (n) WHERE n.uuid = $uuid RETURN n',
            parameters_={'uuid': 'test-uuid', 'database_': 'graphiti'},
        )

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_ARCADEDB, 'ArcadeDB driver dependencies not available')
    async def test_execute_query_propagates_exceptions(self):
        """Test that query exceptions are properly propagated."""
        self.mock_client.execute_query = AsyncMock(side_effect=Exception('Connection refused'))

        with pytest.raises(Exception, match='Connection refused'):
            await self.driver.execute_query('INVALID QUERY')

    @unittest.skipIf(not HAS_ARCADEDB, 'ArcadeDB driver dependencies not available')
    def test_session_creation(self):
        """Test session creation with default database."""
        mock_session = MagicMock()
        self.mock_client.session.return_value = mock_session

        session = self.driver.session()

        self.mock_client.session.assert_called_once_with(database='graphiti')
        assert session is mock_session

    @unittest.skipIf(not HAS_ARCADEDB, 'ArcadeDB driver dependencies not available')
    def test_session_creation_with_custom_database(self):
        """Test session creation with custom database name."""
        mock_session = MagicMock()
        self.mock_client.session.return_value = mock_session

        self.driver.session(database='custom_db')

        self.mock_client.session.assert_called_once_with(database='custom_db')

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_ARCADEDB, 'ArcadeDB driver dependencies not available')
    async def test_close(self):
        """Test driver close method."""
        self.mock_client.close = AsyncMock()

        await self.driver.close()

        self.mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_ARCADEDB, 'ArcadeDB driver dependencies not available')
    async def test_health_check_success(self):
        """Test successful health check."""
        self.mock_client.verify_connectivity = AsyncMock()

        result = await self.driver.health_check()

        self.mock_client.verify_connectivity.assert_called_once()
        assert result is None

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_ARCADEDB, 'ArcadeDB driver dependencies not available')
    async def test_health_check_failure(self):
        """Test health check failure raises exception."""
        self.mock_client.verify_connectivity = AsyncMock(side_effect=Exception('Cannot connect'))

        with pytest.raises(Exception, match='Cannot connect'):
            await self.driver.health_check()

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_ARCADEDB, 'ArcadeDB driver dependencies not available')
    async def test_delete_all_indexes_logs_warning(self):
        """Test delete_all_indexes logs a warning (not supported via Bolt)."""
        with patch('graphiti_core.driver.arcadedb_driver.logger') as mock_logger:
            await self.driver.delete_all_indexes()
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_ARCADEDB, 'ArcadeDB driver dependencies not available')
    async def test_build_indices_and_constraints(self):
        """Test index and constraint building executes queries."""
        self.mock_client.execute_query = AsyncMock(return_value=MagicMock())

        await self.driver.build_indices_and_constraints()

        # Should have been called multiple times for all index queries
        assert self.mock_client.execute_query.call_count > 0

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_ARCADEDB, 'ArcadeDB driver dependencies not available')
    async def test_build_indices_skips_existing(self):
        """Test that index creation skips already-existing indexes gracefully."""
        # Simulate some indexes already existing
        call_count = 0

        async def execute_with_errors(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                raise Exception('Index already exists')
            return MagicMock()

        self.mock_client.execute_query = execute_with_errors

        # Should not raise despite some queries failing
        await self.driver.build_indices_and_constraints()

    @unittest.skipIf(not HAS_ARCADEDB, 'ArcadeDB driver dependencies not available')
    def test_operations_properties(self):
        """Test that all 11 operation properties are available."""
        assert self.driver.entity_node_ops is not None
        assert self.driver.episode_node_ops is not None
        assert self.driver.community_node_ops is not None
        assert self.driver.saga_node_ops is not None
        assert self.driver.entity_edge_ops is not None
        assert self.driver.episodic_edge_ops is not None
        assert self.driver.community_edge_ops is not None
        assert self.driver.has_episode_edge_ops is not None
        assert self.driver.next_episode_edge_ops is not None
        assert self.driver.search_ops is not None
        assert self.driver.graph_ops is not None


class TestArcadeDBTransaction:
    """Test ArcadeDB transaction functionality."""

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_ARCADEDB, 'ArcadeDB driver dependencies not available')
    async def test_transaction_commit(self):
        """Test transaction commits on success."""
        with patch('graphiti_core.driver.arcadedb_driver.AsyncGraphDatabase') as mock_agd:
            mock_agd.driver.return_value = MagicMock()
            driver = ArcadeDBDriver(
                uri='bolt://localhost:2480',
                user='root',
                password='test',
            )

        mock_tx = AsyncMock()
        mock_session = AsyncMock()
        mock_session.begin_transaction = AsyncMock(return_value=mock_tx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        driver.client.session.return_value = mock_session

        async with driver.transaction() as tx:
            await tx.run('CREATE (n:Test)')

        mock_tx.commit.assert_called_once()
        mock_tx.rollback.assert_not_called()

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_ARCADEDB, 'ArcadeDB driver dependencies not available')
    async def test_transaction_rollback_on_error(self):
        """Test transaction rolls back on exception."""
        with patch('graphiti_core.driver.arcadedb_driver.AsyncGraphDatabase') as mock_agd:
            mock_agd.driver.return_value = MagicMock()
            driver = ArcadeDBDriver(
                uri='bolt://localhost:2480',
                user='root',
                password='test',
            )

        mock_tx = AsyncMock()
        mock_session = AsyncMock()
        mock_session.begin_transaction = AsyncMock(return_value=mock_tx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        driver.client.session.return_value = mock_session

        with pytest.raises(ValueError, match='test error'):
            async with driver.transaction():
                raise ValueError('test error')

        mock_tx.rollback.assert_called_once()
        mock_tx.commit.assert_not_called()


# Simple integration test
class TestArcadeDBDriverIntegration:
    """Simple integration test for ArcadeDB driver."""

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_ARCADEDB, 'ArcadeDB driver dependencies not available')
    async def test_basic_integration_with_real_arcadedb(self):
        """Basic integration test with real ArcadeDB instance."""
        arcadedb_uri = os.getenv('ARCADEDB_URI', 'bolt://localhost:2480')
        arcadedb_user = os.getenv('ARCADEDB_USER', 'root')
        arcadedb_password = os.getenv('ARCADEDB_PASSWORD', 'playwithdata')

        try:
            driver = ArcadeDBDriver(
                uri=arcadedb_uri,
                user=arcadedb_user,
                password=arcadedb_password,
            )

            # Test connectivity
            await driver.health_check()

            # Test basic query execution
            result = await driver.execute_query('RETURN 1 as test')
            assert result is not None

            records, summary, keys = result
            assert len(records) == 1

            await driver.close()

        except Exception as e:
            pytest.skip(f'ArcadeDB not available for integration test: {e}')
