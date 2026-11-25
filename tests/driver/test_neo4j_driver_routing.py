"""Unit tests for Neo4jDriver query routing to the configured database.

Neo4j's client takes ``database_`` as a top-level routing keyword, distinct from
Cypher ``parameters_``. These tests assert ``execute_query`` (and index deletion)
pass the driver database through that routing slot rather than stuffing it into
query parameters, so a non-default database name is actually used.

No live Neo4j is required: the scheduled init task is cancelled and ``client``
is replaced with a mock.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.driver.neo4j_driver import Neo4jDriver

pytestmark = pytest.mark.asyncio


def _make_driver(**init_kwargs) -> Neo4jDriver:
    """Build a Neo4jDriver whose client is a mock and whose init task is inert."""
    mock_client = MagicMock()
    mock_client.execute_query = AsyncMock()
    mock_client.close = AsyncMock()

    with (
        patch(
            'graphiti_core.driver.neo4j_driver.AsyncGraphDatabase.driver',
            return_value=mock_client,
        ),
        patch.object(Neo4jDriver, 'build_indices_and_constraints', new_callable=AsyncMock),
    ):
        driver = Neo4jDriver(uri='bolt://x', user='u', password='p', **init_kwargs)

    if driver._init_task is not None:
        driver._init_task.cancel()

    assert driver.client is mock_client
    return driver


async def test_execute_query_routes_to_configured_database():
    driver = _make_driver(database='custom')
    try:
        await driver.execute_query('RETURN 1')

        driver.client.execute_query.assert_awaited_once()
        call_kwargs = driver.client.execute_query.await_args.kwargs
        assert call_kwargs['database_'] == 'custom'
        assert 'database_' not in call_kwargs['parameters_']
    finally:
        await driver.close()


async def test_execute_query_per_call_database_override_wins():
    driver = _make_driver(database='custom')
    try:
        await driver.execute_query('RETURN 1', database_='override')

        driver.client.execute_query.assert_awaited_once()
        call_kwargs = driver.client.execute_query.await_args.kwargs
        assert call_kwargs['database_'] == 'override'
        assert 'database_' not in call_kwargs['parameters_']
    finally:
        await driver.close()


async def test_execute_query_explicit_none_database_requests_home_database():
    driver = _make_driver(database='custom')
    try:
        await driver.execute_query('RETURN 1', database_=None)

        driver.client.execute_query.assert_awaited_once()
        call_kwargs = driver.client.execute_query.await_args.kwargs
        assert call_kwargs['database_'] is None
        assert 'database_' not in call_kwargs['parameters_']
    finally:
        await driver.close()


async def test_execute_query_defaults_to_neo4j_database():
    driver = _make_driver()
    try:
        await driver.execute_query('RETURN 1')

        driver.client.execute_query.assert_awaited_once()
        call_kwargs = driver.client.execute_query.await_args.kwargs
        assert call_kwargs['database_'] == 'neo4j'
        assert 'database_' not in call_kwargs['parameters_']
    finally:
        await driver.close()


async def test_delete_all_indexes_routes_through_execute_query():
    driver = _make_driver(database='custom')
    try:
        await driver.delete_all_indexes()

        driver.client.execute_query.assert_awaited_once()
        call_kwargs = driver.client.execute_query.await_args.kwargs
        assert call_kwargs['database_'] == 'custom'
        assert 'database_' not in call_kwargs['parameters_']
    finally:
        await driver.close()
