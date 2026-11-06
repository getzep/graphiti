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

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.driver.driver import GraphProvider

try:
    from graphiti_core.driver.falkordb_driver import FalkorDriver, FalkorDriverSession

    HAS_FALKORDB = FalkorDriverSession is not None
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
    async def test_execute_query_reraises_already_indexed(self):
        """execute_query must not swallow index errors; the caller demotes them."""
        driver = FalkorDriver.__new__(FalkorDriver)
        driver.client = MagicMock()
        driver._database = 'GRAPHITI'

        graph = MagicMock()
        graph.query = AsyncMock(side_effect=Exception("Attribute 'uuid' is already indexed"))
        driver.client.select_graph = MagicMock(return_value=graph)

        with pytest.raises(Exception, match='already indexed'):
            await driver.execute_query('CREATE INDEX FOR (n:Entity) ON (n.uuid)')

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_execute_query_reraises_stopwords_already_set(self):
        """FalkorDB 6.x stopwords idempotence error follows the re-raise contract."""
        driver = FalkorDriver.__new__(FalkorDriver)
        driver.client = MagicMock()
        driver._database = 'GRAPHITI'

        graph = MagicMock()
        graph.query = AsyncMock(
            side_effect=Exception(
                "Can not override index configuration: Stopwords are already set for label 'Entity'"
            )
        )
        driver.client.select_graph = MagicMock(return_value=graph)

        with pytest.raises(Exception, match='Stopwords are already set'):
            await driver.execute_query('CREATE FULLTEXT INDEX ...')

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_build_indices_demotes_already_indexed(self, caplog):
        """build_indices_and_constraints demotes idempotent index errors to INFO."""
        driver = FalkorDriver.__new__(FalkorDriver)
        driver.client = MagicMock()
        driver._database = 'GRAPHITI'
        driver.provider = GraphProvider.FALKORDB

        graph = MagicMock()
        graph.query = AsyncMock(side_effect=Exception("Attribute 'uuid' is already indexed"))
        driver.client.select_graph = MagicMock(return_value=graph)

        import logging as _logging

        with caplog.at_level(_logging.INFO, logger='graphiti_core.driver.falkordb_driver'):
            await driver.build_indices_and_constraints()

        assert any('Index already exists' in r.message for r in caplog.records)
