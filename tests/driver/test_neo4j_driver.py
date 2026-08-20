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

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.neo4j_driver import Neo4jDriver


def _build_driver(database: str = 'neo4j') -> tuple[Neo4jDriver, MagicMock]:
    """Instantiate a Neo4jDriver with a mocked underlying async client.

    Suppresses the background ``build_indices_and_constraints`` task that
    ``__init__`` normally schedules when a running loop is present, so tests
    exercise only the code under test.
    """
    mock_client = MagicMock()
    mock_client.execute_query = AsyncMock(return_value=MagicMock())
    with (
        patch(
            'graphiti_core.driver.neo4j_driver.AsyncGraphDatabase.driver',
            return_value=mock_client,
        ),
        patch(
            'graphiti_core.driver.neo4j_driver.asyncio.get_running_loop',
            side_effect=RuntimeError,
        ),
    ):
        driver = Neo4jDriver(
            uri='bolt://localhost:7687',
            user='neo4j',
            password='password',
            database=database,
        )
    return driver, mock_client


class TestNeo4jDriverProvider:
    def test_provider(self):
        driver, _ = _build_driver()
        assert driver.provider == GraphProvider.NEO4J


class TestNeo4jDriverDatabaseRouting:
    """Regression tests for issue #1481.

    Neo4j's driver expects `database_` as a top-level keyword argument to
    ``execute_query``; when it is nested inside ``parameters_`` it becomes an
    unused Cypher variable and the query is silently routed to the connection's
    home database rather than the configured one.
    """

    @pytest.mark.asyncio
    async def test_configured_database_is_passed_as_connection_kwarg(self):
        driver, mock_client = _build_driver(database='mydb')

        await driver.execute_query('MATCH (n) RETURN n')

        call_kwargs = mock_client.execute_query.call_args.kwargs
        assert call_kwargs.get('database_') == 'mydb'
        # `database_` must NOT be smuggled into Cypher parameters.
        assert 'database_' not in call_kwargs.get('parameters_', {})

    @pytest.mark.asyncio
    async def test_default_database_is_used_when_none_configured(self):
        driver, mock_client = _build_driver()  # defaults to 'neo4j'

        await driver.execute_query('MATCH (n) RETURN n')

        call_kwargs = mock_client.execute_query.call_args.kwargs
        assert call_kwargs.get('database_') == 'neo4j'
        assert 'database_' not in call_kwargs.get('parameters_', {})

    @pytest.mark.asyncio
    async def test_explicit_database_kwarg_overrides_configured_default(self):
        driver, mock_client = _build_driver(database='mydb')

        await driver.execute_query('MATCH (n) RETURN n', database_='override_db')

        call_kwargs = mock_client.execute_query.call_args.kwargs
        assert call_kwargs.get('database_') == 'override_db'
        assert 'database_' not in call_kwargs.get('parameters_', {})

    @pytest.mark.asyncio
    async def test_database_nested_in_params_is_promoted_to_connection_kwarg(self):
        """Backwards compatibility: honor callers that stuck database_ inside `params`."""
        driver, mock_client = _build_driver(database='mydb')

        await driver.execute_query(
            'MATCH (n) RETURN n',
            params={'database_': 'legacy_db', 'name': 'Alice'},
        )

        call_kwargs = mock_client.execute_query.call_args.kwargs
        assert call_kwargs.get('database_') == 'legacy_db'
        # The Cypher-level parameters must not carry the driver-level knob.
        assert 'database_' not in call_kwargs.get('parameters_', {})
        # But other Cypher parameters must pass through untouched.
        assert call_kwargs.get('parameters_', {}).get('name') == 'Alice'


class TestNeo4jDriverExecuteQueryPassthrough:
    """Sanity checks that unrelated kwargs continue to work as before."""

    @pytest.mark.asyncio
    async def test_routing_kwarg_is_forwarded_at_connection_level(self):
        driver, mock_client = _build_driver()

        await driver.execute_query('MATCH (n) RETURN n', routing_='r')

        call_kwargs = mock_client.execute_query.call_args.kwargs
        assert call_kwargs.get('routing_') == 'r'

    @pytest.mark.asyncio
    async def test_extra_kwargs_are_forwarded(self):
        driver, mock_client = _build_driver()

        await driver.execute_query('MATCH (n {name: $name}) RETURN n', name='Alice')

        call_kwargs = mock_client.execute_query.call_args.kwargs
        assert call_kwargs.get('name') == 'Alice'
