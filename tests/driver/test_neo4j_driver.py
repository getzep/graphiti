"""Tests for Neo4jDriver index-creation fast path."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.graph_queries import get_neo4j_expected_index_names


def _make_driver():
    """Create a Neo4jDriver with a mocked Neo4j client (no real connection).

    Patches AsyncGraphDatabase so no real connection is made, and temporarily
    stubs out build_indices_and_constraints so the __init__ background task
    (scheduled via loop.create_task) does not interfere with tests.
    """
    from graphiti_core.driver.neo4j_driver import Neo4jDriver

    with (
        patch('graphiti_core.driver.neo4j_driver.AsyncGraphDatabase') as mock_agd,
        patch.object(Neo4jDriver, 'build_indices_and_constraints', new_callable=AsyncMock),
    ):
        mock_agd.driver.return_value = MagicMock()
        driver = Neo4jDriver(uri='bolt://localhost:7687', user='neo4j', password='test')
    # Restore the real method after construction
    driver._indices_verified = False
    return driver


def _mock_show_indexes_result(names: set[str]):
    """Build a fake EagerResult whose .records yield {'name': n} dicts."""
    result = MagicMock()
    result.records = [{'name': n} for n in names]
    return result


class TestGetNeo4jExpectedIndexNames:
    def test_returns_expected_count(self):
        """get_neo4j_expected_index_names returns exactly 31 index names."""
        names = get_neo4j_expected_index_names()
        assert len(names) == 31
        assert all(isinstance(n, str) for n in names)

    def test_contains_known_indices(self):
        """Spot-check that well-known index names are present."""
        names = get_neo4j_expected_index_names()
        for expected in ('entity_uuid', 'episode_content', 'node_name_and_summary'):
            assert expected in names


class TestBuildIndicesFastPath:
    @pytest.mark.asyncio
    async def test_fast_path_all_exist(self):
        """When all indices already exist, only 1 SHOW INDEXES query is executed."""
        driver = _make_driver()
        expected_names = get_neo4j_expected_index_names()
        # Superset: DB has all expected + some extras
        existing = expected_names | {'extra_index'}

        driver.execute_query = AsyncMock(return_value=_mock_show_indexes_result(existing))

        await driver.build_indices_and_constraints()

        # Only the SHOW INDEXES query should have been called
        driver.execute_query.assert_called_once()
        call_args = driver.execute_query.call_args
        assert 'SHOW INDEXES' in call_args[0][0]
        assert driver._indices_verified is True

    @pytest.mark.asyncio
    async def test_partial_missing_falls_through(self):
        """When some indices are missing, all CREATE queries are executed."""
        driver = _make_driver()
        # Return only a subset of expected indices
        partial = {'entity_uuid', 'episode_uuid'}
        driver.execute_query = AsyncMock(return_value=_mock_show_indexes_result(partial))

        await driver.build_indices_and_constraints()

        # 1 SHOW INDEXES + 31 CREATE INDEX queries = 32
        assert driver.execute_query.call_count == 32
        assert driver._indices_verified is True

    @pytest.mark.asyncio
    async def test_delete_existing_bypasses_fast_path(self):
        """delete_existing=True always runs delete + CREATE queries."""
        driver = _make_driver()
        driver._indices_verified = True  # Should be ignored

        driver.execute_query = AsyncMock(return_value=MagicMock(records=[]))
        driver.client.execute_query = AsyncMock()  # for delete_all_indexes

        await driver.build_indices_and_constraints(delete_existing=True)

        # Flag should have been reset then set again
        assert driver._indices_verified is True
        # Should have executed the 31 CREATE INDEX queries (no SHOW INDEXES)
        assert driver.execute_query.call_count == 31

    @pytest.mark.asyncio
    async def test_show_indexes_failure_falls_through(self):
        """If SHOW INDEXES fails, fall through to individual CREATE queries."""
        driver = _make_driver()

        call_count = 0

        async def side_effect(query, **kwargs):
            nonlocal call_count
            call_count += 1
            if 'SHOW INDEXES' in query:
                raise RuntimeError('Simulated SHOW INDEXES failure')
            return MagicMock()

        driver.execute_query = AsyncMock(side_effect=side_effect)

        await driver.build_indices_and_constraints()

        # 1 failed SHOW INDEXES + 31 CREATE INDEX queries = 32
        assert call_count == 32
        assert driver._indices_verified is True

    @pytest.mark.asyncio
    async def test_indices_verified_flag_skips_second_call(self):
        """Second call with _indices_verified=True does zero queries."""
        driver = _make_driver()
        driver._indices_verified = True

        driver.execute_query = AsyncMock()

        await driver.build_indices_and_constraints()

        driver.execute_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_existing_resets_verified_flag(self):
        """delete_existing=True resets _indices_verified even if it was True."""
        driver = _make_driver()
        driver._indices_verified = True

        driver.execute_query = AsyncMock(return_value=MagicMock(records=[]))
        driver.client.execute_query = AsyncMock()

        await driver.build_indices_and_constraints(delete_existing=True)

        # Flag should be True again after successful recreation
        assert driver._indices_verified is True
        # All 31 CREATE queries should have been issued
        assert driver.execute_query.call_count == 31
