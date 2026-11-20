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
