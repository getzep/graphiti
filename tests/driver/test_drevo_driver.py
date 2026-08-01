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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.driver.capabilities import GraphCapabilities
from graphiti_core.driver.driver import GraphProvider

try:
    from graphiti_core.driver.drevo_driver import DrevoDriver

    HAS_NEO4J = True
except ImportError:
    DrevoDriver = None
    HAS_NEO4J = False


class TestGraphCapabilities:
    """The capability declaration a connector uses to negotiate features."""

    def test_defaults_are_conservative(self):
        """A bare capability set claims nothing native — the safe default for a
        minimal Bolt/Cypher backend."""
        caps = GraphCapabilities()
        assert caps.supports_transactions is False
        assert caps.supports_native_fulltext_search is False
        assert caps.supports_native_vector_search is False
        assert caps.supports_vector_index is False

    def test_explicit_flags(self):
        caps = GraphCapabilities(
            supports_transactions=True,
            supports_native_fulltext_search=True,
            supports_native_vector_search=True,
            supports_vector_index=True,
        )
        assert caps.supports_transactions is True
        assert caps.supports_native_fulltext_search is True
        assert caps.supports_native_vector_search is True
        assert caps.supports_vector_index is True


@pytest.mark.skipif(not HAS_NEO4J, reason='neo4j driver package is not installed')
class TestDrevoDriver:
    """drevo is Bolt/Cypher-compatible, so the connector reuses the Neo4j Bolt
    machinery but declares a reduced capability set: no index DDL, no explicit
    transactions (Bolt exposes only autocommit RUN)."""

    def _make_driver(self) -> 'DrevoDriver':
        with patch('graphiti_core.driver.neo4j_driver.AsyncGraphDatabase') as mock_gdb:
            mock_gdb.driver.return_value = MagicMock()
            return DrevoDriver(uri='bolt://localhost:7687', user='neo4j', password='password')

    def test_provider_is_drevo(self):
        driver = self._make_driver()
        assert driver.provider == GraphProvider.DREVO

    def test_declares_no_native_indexes_or_transactions(self):
        driver = self._make_driver()
        caps = driver.capabilities
        assert isinstance(caps, GraphCapabilities)
        # drevo has no CREATE INDEX/VECTOR/FULLTEXT DDL in its Cypher subset
        assert caps.supports_vector_index is False
        # Bolt handshake advertises only RUN/PULL/... — no BEGIN/COMMIT
        assert caps.supports_transactions is False
        # Native vector similarity via cosine_similarity (drevo#202) and native BM25
        # node full-text via fts.search (drevo#208/#227).
        assert caps.supports_native_vector_search is True
        assert caps.supports_native_fulltext_search is True

    def test_search_interface_is_wired(self):
        """search_utils routes all search through driver.search_interface when set,
        so the drevo overlay must be installed on the instance."""
        from graphiti_core.driver.drevo_search_interface import DrevoSearchInterface

        driver = self._make_driver()
        assert isinstance(driver.search_interface, DrevoSearchInterface)

    @pytest.mark.asyncio
    async def test_build_indices_is_noop(self):
        """drevo manages indexes out-of-band; the connector must not emit Neo4j
        index procedures, so build_indices_and_constraints issues no queries."""
        driver = self._make_driver()
        with patch.object(driver, 'execute_query') as mock_exec:
            await driver.build_indices_and_constraints()
            mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_all_indexes_is_noop(self):
        driver = self._make_driver()
        with patch.object(driver, 'execute_query') as mock_exec:
            result = driver.delete_all_indexes()
            if result is not None:
                await result
            mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_transaction_is_immediate_mode(self):
        """No explicit-transaction support over Bolt, so transaction() must not
        take the Neo4j begin_transaction path."""
        from graphiti_core.driver.driver import _SessionTransaction

        driver = self._make_driver()
        mock_session = MagicMock()
        mock_session.close = AsyncMock()
        driver.client.session = MagicMock(return_value=mock_session)
        async with driver.transaction() as tx:
            assert isinstance(tx, _SessionTransaction)
