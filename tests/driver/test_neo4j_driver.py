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

from graphiti_core.driver.neo4j_driver import Neo4jDriver


def build_driver(database: str = 'test_db') -> tuple[Neo4jDriver, MagicMock]:
    mock_client = MagicMock()
    mock_client.execute_query = AsyncMock(return_value=([], None, None))

    with (
        patch('graphiti_core.driver.neo4j_driver.AsyncGraphDatabase.driver') as mock_driver,
        patch('asyncio.get_running_loop', side_effect=RuntimeError),
    ):
        mock_driver.return_value = mock_client
        driver = Neo4jDriver('bolt://localhost:7687', 'neo4j', 'password', database=database)

    return driver, mock_client


@pytest.mark.asyncio
async def test_execute_query_passes_configured_database_as_driver_argument():
    driver, mock_client = build_driver(database='tenant_db')

    await driver.execute_query('MATCH (n {uuid: $uuid}) RETURN n', uuid='node-uuid', routing_='r')

    mock_client.execute_query.assert_awaited_once_with(
        'MATCH (n {uuid: $uuid}) RETURN n',
        parameters_={},
        uuid='node-uuid',
        routing_='r',
        database_='tenant_db',
    )


@pytest.mark.asyncio
async def test_execute_query_keeps_cypher_params_separate_from_database_argument():
    driver, mock_client = build_driver(database='tenant_db')
    params = {'uuid': 'node-uuid'}

    await driver.execute_query('MATCH (n {uuid: $uuid}) RETURN n', params=params)

    mock_client.execute_query.assert_awaited_once_with(
        'MATCH (n {uuid: $uuid}) RETURN n',
        parameters_=params,
        database_='tenant_db',
    )
    assert 'database_' not in params


@pytest.mark.asyncio
async def test_execute_query_preserves_explicit_database_argument():
    driver, mock_client = build_driver(database='tenant_db')

    await driver.execute_query('RETURN 1', database_='override_db')

    mock_client.execute_query.assert_awaited_once_with(
        'RETURN 1',
        parameters_={},
        database_='override_db',
    )
