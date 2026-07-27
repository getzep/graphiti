"""Unit tests for the FalkorDB ``group_routing`` mode (#1684).

``group_routing='database'`` (the default) keeps the historical behavior where a
``group_id`` selects its own FalkorDB graph. ``group_routing='record'`` pins the
driver to the configured graph and treats ``group_id`` as a record-level tenant
scope, so a single shared graph can hold several groups.

The tests are hermetic: the driver is built with a mocked FalkorDB client and the
operations layer runs against ``spec=GraphDriver`` mocks, so no live database,
LLM, embedder, or credentials are required.
"""

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from graphiti_core.driver.driver import GraphDriver, GraphProvider, GroupRouting
from graphiti_core.driver.falkordb.operations.entity_node_ops import (
    FalkorEntityNodeOperations,
)
from graphiti_core.driver.falkordb.operations.episode_node_ops import (
    FalkorEpisodeNodeOperations,
)
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

try:
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    HAS_FALKORDB = True
except ImportError:
    FalkorDriver = None
    HAS_FALKORDB = False

requires_falkordb = pytest.mark.skipif(not HAS_FALKORDB, reason='FalkorDB is not installed')


def _make_driver(**kwargs):
    """Build a FalkorDriver against a mocked client (no connection, no index build)."""
    with patch('graphiti_core.driver.falkordb_driver.FalkorDB'):
        return FalkorDriver(**kwargs)


# --- driver-level routing -------------------------------------------------


@requires_falkordb
def test_default_routing_is_database():
    """Default behavior is unchanged: group_id selects its own graph."""
    driver = _make_driver(database='default_db')

    assert driver.group_routing is GroupRouting.DATABASE
    assert driver.routes_group_ids_to_databases is True


@requires_falkordb
def test_default_routing_clone_selects_group_graph():
    driver = _make_driver(database='default_db')

    cloned = driver.clone(database='projecta')

    assert cloned is not driver
    assert cloned._database == 'projecta'
    assert cloned.client is driver.client


@requires_falkordb
def test_record_routing_pins_driver_to_configured_graph():
    """Record routing never re-points the driver, whatever group_id asks for."""
    driver = _make_driver(database='graphiti_personal', group_routing='record')

    assert driver.group_routing is GroupRouting.RECORD
    assert driver.routes_group_ids_to_databases is False
    assert driver.clone(database='projecta') is driver
    assert driver.clone(database=driver.default_group_id) is driver
    assert driver._database == 'graphiti_personal'


@requires_falkordb
def test_record_routing_accepts_enum_member():
    driver = _make_driver(database='shared', group_routing=GroupRouting.RECORD)

    assert driver.group_routing is GroupRouting.RECORD


@requires_falkordb
def test_database_routing_propagates_to_clones():
    driver = _make_driver(database='default_db', group_routing='database')

    cloned = driver.clone(database='projecta')

    assert cloned.group_routing is GroupRouting.DATABASE


@requires_falkordb
def test_invalid_group_routing_is_rejected():
    with pytest.raises(ValueError, match='group_routing'):
        _make_driver(group_routing='per-tenant')


@requires_falkordb
def test_record_routing_session_uses_configured_graph():
    """Sessions (used by group-scoped cleanup) stay on the configured graph."""
    driver = _make_driver(database='graphiti_personal', group_routing='record')
    driver.client = MagicMock()

    driver.session()

    driver.client.select_graph.assert_called_once_with('graphiti_personal')


@requires_falkordb
@pytest.mark.asyncio
async def test_record_routing_group_cleanup_stays_in_fixed_graph():
    """Group-scoped cleanup deletes by group_id inside the one configured graph."""
    graph = MagicMock()
    graph.query = AsyncMock(return_value=MagicMock(header=[], result_set=[]))

    with patch.object(FalkorDriver, 'build_indices_and_constraints', new=AsyncMock()):
        driver = _make_driver(database='graphiti_personal', group_routing='record')
    driver.client = MagicMock()
    driver.client.select_graph = MagicMock(return_value=graph)

    await clear_data(driver, group_ids=['projecta'])

    assert driver.client.select_graph.call_args_list == [call('graphiti_personal')]
    # One delete per label, each scoped by group_id rather than by graph.
    assert graph.query.await_count == 3
    for query_call in graph.query.await_args_list:
        cypher, params = query_call.args
        assert 'n.group_id IN $group_ids' in cypher
        assert params == {'group_ids': ['projecta']}


# --- operations-layer routing ---------------------------------------------


def _record_mode_driver(execute_return=None):
    """A GraphDriver-shaped mock configured for record-level routing."""
    driver = MagicMock(spec=GraphDriver)
    driver.provider = GraphProvider.FALKORDB
    driver.group_routing = GroupRouting.RECORD
    driver.routes_group_ids_to_databases = False
    driver.execute_query = AsyncMock(return_value=execute_return or ([], None, None))
    driver.clone = MagicMock(return_value=driver)
    return driver


@pytest.mark.asyncio
async def test_episode_get_by_group_ids_record_mode_does_not_clone():
    ops = FalkorEpisodeNodeOperations()
    driver = _record_mode_driver()

    result = await ops.get_by_group_ids(driver, ['projecta'])

    driver.clone.assert_not_called()
    driver.execute_query.assert_awaited_once()
    assert driver.execute_query.await_args.kwargs['group_ids'] == ['projecta']
    assert result == []


@pytest.mark.asyncio
async def test_entity_node_get_by_group_ids_record_mode_queries_all_groups_at_once():
    """Multiple group_ids resolve in a single filtered query, not one graph per group."""
    ops = FalkorEntityNodeOperations()
    driver = _record_mode_driver()

    await ops.get_by_group_ids(driver, ['projecta', 'projectb'])

    driver.clone.assert_not_called()
    driver.execute_query.assert_awaited_once()
    assert driver.execute_query.await_args.kwargs['group_ids'] == ['projecta', 'projectb']
