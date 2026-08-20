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


@pytest.fixture
def neo4j_driver() -> Neo4jDriver:
    client = MagicMock()
    client.execute_query = AsyncMock(return_value=MagicMock())

    with (
        patch('graphiti_core.driver.neo4j_driver.AsyncGraphDatabase.driver', return_value=client),
        patch.object(Neo4jDriver, 'build_indices_and_constraints', new_callable=AsyncMock),
    ):
        return Neo4jDriver('bolt://localhost:7687', 'neo4j', 'password')


@pytest.mark.asyncio
async def test_execute_query_accepts_neo4j_parameters_keyword(neo4j_driver: Neo4jDriver):
    result = await neo4j_driver.execute_query(
        'MATCH (n {group_id: $group_id}) RETURN n',
        parameters_={'group_id': 'test-group'},
    )

    assert result is neo4j_driver.client.execute_query.return_value
    neo4j_driver.client.execute_query.assert_awaited_once_with(
        'MATCH (n {group_id: $group_id}) RETURN n',
        parameters_={'group_id': 'test-group', 'database_': 'neo4j'},
    )


@pytest.mark.asyncio
async def test_execute_query_preserves_graphiti_params_keyword(neo4j_driver: Neo4jDriver):
    result = await neo4j_driver.execute_query(
        'MATCH (n {group_id: $group_id}) RETURN n',
        params={'group_id': 'test-group'},
    )

    assert result is neo4j_driver.client.execute_query.return_value
    neo4j_driver.client.execute_query.assert_awaited_once_with(
        'MATCH (n {group_id: $group_id}) RETURN n',
        parameters_={'group_id': 'test-group', 'database_': 'neo4j'},
    )
