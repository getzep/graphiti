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

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode


@pytest.mark.parametrize(
    'provider',
    [GraphProvider.NEO4J, GraphProvider.FALKORDB, GraphProvider.NEPTUNE],
)
@pytest.mark.asyncio
async def test_episodic_get_by_uuids_fallback_filters_group_id(provider):
    driver = MagicMock()
    driver.provider = provider
    driver.graph_operations_interface = None
    driver.execute_query = AsyncMock(return_value=([], [], None))

    await EpisodicNode.get_by_uuids(driver, ['episode-1'], group_id='group-a')

    query = driver.execute_query.await_args.args[0]
    parameters = driver.execute_query.await_args.kwargs
    assert 'AND e.group_id = $group_id' in query
    assert parameters['group_id'] == 'group-a'


@pytest.mark.parametrize(
    'provider',
    [GraphProvider.NEO4J, GraphProvider.FALKORDB, GraphProvider.NEPTUNE],
)
@pytest.mark.asyncio
async def test_entity_edge_get_by_uuids_fallback_filters_group_id(provider):
    driver = MagicMock()
    driver.provider = provider
    driver.graph_operations_interface = None
    driver.execute_query = AsyncMock(return_value=([], [], None))

    await EntityEdge.get_by_uuids(driver, ['edge-1'], group_id='group-a')

    query = driver.execute_query.await_args.args[0]
    parameters = driver.execute_query.await_args.kwargs
    assert 'AND e.group_id = $group_id' in query
    assert parameters['group_id'] == 'group-a'
