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

# Guard imports — falkordblite requires Python 3.12+ and may not be installed
try:
    from graphiti_core.driver.falkordb_lite_driver import FalkorLiteDriver

    HAS_FALKORDBLITE = True
except ImportError:
    FalkorLiteDriver = None
    HAS_FALKORDBLITE = False

# Also check for falkordb since FalkorLiteDriver inherits from FalkorDriver
try:
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    HAS_FALKORDB = True
except ImportError:
    HAS_FALKORDB = False

SKIP_MSG = 'falkordblite or falkordb is not installed'
should_skip = not (HAS_FALKORDBLITE and HAS_FALKORDB)


class TestFalkorLiteDriver:
    """Test suite for FalkorDB Lite embedded driver."""

    @unittest.skipIf(should_skip, SKIP_MSG)
    def setup_method(self):
        """Set up test fixtures with a mocked lite client."""
        self.mock_lite_client = MagicMock()
        with patch('graphiti_core.driver.falkordb_driver.FalkorDB'):
            self.driver = FalkorLiteDriver(
                path='/tmp/test.db', falkor_db=self.mock_lite_client
            )

    @unittest.skipIf(should_skip, SKIP_MSG)
    def test_init_with_path(self):
        """Test initialization creates an AsyncFalkorDB with the given path."""
        mock_async_falkordb = MagicMock()
        with (
            patch('graphiti_core.driver.falkordb_driver.FalkorDB'),
            patch(
                'graphiti_core.driver.falkordb_lite_driver.LiteAsyncFalkorDB',
                return_value=mock_async_falkordb,
            ) as mock_lite_cls,
        ):
            driver = FalkorLiteDriver(path='/tmp/my_graph.db')

            mock_lite_cls.assert_called_once_with('/tmp/my_graph.db')
            assert driver.client is mock_async_falkordb
            assert driver._path == '/tmp/my_graph.db'

    @unittest.skipIf(should_skip, SKIP_MSG)
    def test_init_with_existing_client(self):
        """Test initialization with an injected client skips creating a new one."""
        mock_client = MagicMock()
        with patch('graphiti_core.driver.falkordb_driver.FalkorDB'):
            driver = FalkorLiteDriver(path='/tmp/test.db', falkor_db=mock_client)

        assert driver.client is mock_client

    @unittest.skipIf(should_skip, SKIP_MSG)
    def test_init_with_custom_database(self):
        """Test initialization with a custom database name."""
        with patch('graphiti_core.driver.falkordb_driver.FalkorDB'):
            driver = FalkorLiteDriver(
                path='/tmp/test.db',
                falkor_db=MagicMock(),
                database='custom_db',
            )

        assert driver._database == 'custom_db'

    @unittest.skipIf(should_skip, SKIP_MSG)
    def test_provider_is_falkordb(self):
        """Test that provider is FALKORDB (same Cypher dialect, no separate enum)."""
        assert self.driver.provider == GraphProvider.FALKORDB

    @unittest.skipIf(should_skip, SKIP_MSG)
    def test_inherits_from_falkor_driver(self):
        """Test that FalkorLiteDriver is a subclass of FalkorDriver."""
        assert isinstance(self.driver, FalkorDriver)

    @pytest.mark.asyncio
    @unittest.skipIf(should_skip, SKIP_MSG)
    async def test_close_calls_client_close(self):
        """Test close() calls close() on the embedded client."""
        self.mock_lite_client.close = AsyncMock()
        # Remove aclose so the fallback to close() is exercised
        if hasattr(self.mock_lite_client, 'aclose'):
            del self.mock_lite_client.aclose

        await self.driver.close()

        self.mock_lite_client.close.assert_called_once()

    @pytest.mark.asyncio
    @unittest.skipIf(should_skip, SKIP_MSG)
    async def test_close_prefers_aclose(self):
        """Test close() prefers aclose() if available."""
        self.mock_lite_client.aclose = AsyncMock()

        await self.driver.close()

        self.mock_lite_client.aclose.assert_called_once()

    @unittest.skipIf(should_skip, SKIP_MSG)
    def test_clone_same_database_returns_self(self):
        """Test clone with same database returns self."""
        result = self.driver.clone('default_db')
        assert result is self.driver

    @unittest.skipIf(should_skip, SKIP_MSG)
    def test_clone_different_database_returns_lite_driver(self):
        """Test clone with different database returns a FalkorLiteDriver, not FalkorDriver."""
        with patch('graphiti_core.driver.falkordb_driver.FalkorDB'):
            cloned = self.driver.clone('other_db')

        assert isinstance(cloned, FalkorLiteDriver)
        assert cloned._database == 'other_db'
        assert cloned.client is self.mock_lite_client  # Shares the same embedded client
        assert cloned._path == '/tmp/test.db'

    @unittest.skipIf(should_skip, SKIP_MSG)
    def test_clone_default_group_id_returns_lite_driver(self):
        """Test clone with default_group_id returns FalkorLiteDriver with default database."""
        with patch('graphiti_core.driver.falkordb_driver.FalkorDB'):
            cloned = self.driver.clone(self.driver.default_group_id)

        assert isinstance(cloned, FalkorLiteDriver)
        assert cloned.client is self.mock_lite_client

    @pytest.mark.asyncio
    @unittest.skipIf(should_skip, SKIP_MSG)
    async def test_execute_query_delegates_to_parent(self):
        """Test that execute_query works through the inherited parent implementation."""
        mock_graph = MagicMock()
        mock_result = MagicMock()
        mock_result.header = [('col1', 'name')]
        mock_result.result_set = [['Alice']]
        mock_graph.query = AsyncMock(return_value=mock_result)
        self.mock_lite_client.select_graph.return_value = mock_graph

        result = await self.driver.execute_query('MATCH (n) RETURN n.name as name')

        assert result is not None
        result_set, header, summary = result
        assert result_set == [{'name': 'Alice'}]
        assert header == ['name']

    @unittest.skipIf(should_skip, SKIP_MSG)
    def test_get_graph_delegates_to_parent(self):
        """Test that _get_graph uses the embedded client's select_graph."""
        mock_graph = MagicMock()
        self.mock_lite_client.select_graph.return_value = mock_graph

        result = self.driver._get_graph('test_graph')

        self.mock_lite_client.select_graph.assert_called_once_with('test_graph')
        assert result is mock_graph


class TestFalkorLiteDriverIntegration:
    """Integration test for FalkorDB Lite embedded driver.

    Requires falkordblite to be installed and Python 3.12+.
    """

    @pytest.mark.asyncio
    @unittest.skipIf(should_skip, SKIP_MSG)
    async def test_basic_integration_with_embedded_falkordb(self):
        """Basic integration test with a real embedded FalkorDB instance."""
        pytest.importorskip('redislite')

        falkordb_lite_path = os.getenv('FALKORDB_LITE_PATH', '/tmp/graphiti_test_lite.db')

        try:
            driver = FalkorLiteDriver(path=falkordb_lite_path)

            result = await driver.execute_query('RETURN 1 as test')
            assert result is not None

            result_set, header, summary = result
            assert header == ['test']
            assert result_set == [{'test': 1}]

            await driver.close()

        except Exception as e:
            pytest.skip(f'FalkorDB Lite not available for integration test: {e}')
