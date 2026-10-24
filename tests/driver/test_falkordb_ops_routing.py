"""Unit tests for FalkorDB per-group-id query routing in the operations layer.

These cover the *namespace/ops* read path (e.g. ``graphiti.nodes.episode.
get_by_group_ids``), which delegates to the Falkor ``*Operations`` classes with
the base driver (pointed at ``default_db``). FalkorDB stores each ``group_id`` in
its own physical graph, so a group-scoped read must clone the driver to that
group's graph or it queries an empty default database and returns nothing.

The tests use a ``spec=GraphDriver`` mock, so they need no live FalkorDB and run
in CI regardless of whether the ``falkordb`` package is installed.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.driver.falkordb.operations.entity_edge_ops import (
    FalkorEntityEdgeOperations,
)
from graphiti_core.driver.falkordb.operations.entity_node_ops import (
    FalkorEntityNodeOperations,
)
from graphiti_core.driver.falkordb.operations.episode_node_ops import (
    FalkorEpisodeNodeOperations,
)

pytestmark = pytest.mark.asyncio


def _make_falkor_driver(execute_return=None):
    """A GraphDriver-shaped mock whose ``clone`` returns a routed child driver."""
    child = MagicMock(spec=GraphDriver)
    child.provider = GraphProvider.FALKORDB
    child.execute_query = AsyncMock(return_value=execute_return or ([], None, None))

    base = MagicMock(spec=GraphDriver)
    base.provider = GraphProvider.FALKORDB
    base.execute_query = AsyncMock(return_value=execute_return or ([], None, None))
    base.clone = MagicMock(return_value=child)
    return base, child


def _wired_falkor_driver():
    """FalkorDriver with the graph-operations facade, no live database."""
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    with patch.object(FalkorDriver, 'build_indices_and_constraints', new=AsyncMock()):
        driver = FalkorDriver(falkor_db=MagicMock())
    driver.execute_query = AsyncMock(return_value=([], None, None))
    return driver


async def test_episode_get_by_group_ids_single_group_routes_to_its_graph():
    """The single-group case must clone the base driver to the group's graph."""
    ops = FalkorEpisodeNodeOperations()
    base, child = _make_falkor_driver()

    result = await ops.get_by_group_ids(base, ['group-a'])

    # Routed: cloned to the group's database, query ran against the clone,
    # never against the base (default_db) driver.
    base.clone.assert_called_once_with(database='group-a')
    child.execute_query.assert_awaited_once()
    base.execute_query.assert_not_called()
    assert result == []


async def test_entity_node_get_by_group_ids_single_group_routes_to_its_graph():
    ops = FalkorEntityNodeOperations()
    base, child = _make_falkor_driver()

    await ops.get_by_group_ids(base, ['tenant-1'])

    base.clone.assert_called_once_with(database='tenant-1')
    child.execute_query.assert_awaited_once()
    base.execute_query.assert_not_called()


async def test_entity_edge_get_by_group_ids_single_group_routes_to_its_graph():
    ops = FalkorEntityEdgeOperations()
    base, child = _make_falkor_driver()

    await ops.get_by_group_ids(base, ['tenant-1'])

    base.clone.assert_called_once_with(database='tenant-1')
    child.execute_query.assert_awaited_once()
    base.execute_query.assert_not_called()


async def test_episode_get_by_group_ids_multi_group_fans_out_per_graph():
    """The multi-group case must clone once per group_id and aggregate."""
    ops = FalkorEpisodeNodeOperations()
    base = MagicMock(spec=GraphDriver)
    base.provider = GraphProvider.FALKORDB
    base.execute_query = AsyncMock(return_value=([], None, None))

    clones = {}

    def _clone(database):
        child = MagicMock(spec=GraphDriver)
        child.provider = GraphProvider.FALKORDB
        child.execute_query = AsyncMock(return_value=([], None, None))
        # Mirror FalkorDriver.clone: re-cloning to the same database returns self,
        # so the per-group recursion's single-group clone is a no-op.
        child.clone = MagicMock(return_value=child)
        clones[database] = child
        return child

    base.clone = MagicMock(side_effect=_clone)

    await ops.get_by_group_ids(base, ['g1', 'g2', 'g3'])

    # One clone per group; the base driver itself never runs the query.
    assert sorted(clones) == ['g1', 'g2', 'g3']
    base.execute_query.assert_not_called()
    for child in clones.values():
        child.execute_query.assert_awaited_once()


async def test_get_by_group_ids_empty_group_ids_does_not_route():
    """No group_ids => no cloning; falls through to the normal (base) query."""
    ops = FalkorEpisodeNodeOperations()
    base, child = _make_falkor_driver()

    await ops.get_by_group_ids(base, [])

    base.clone.assert_not_called()
    base.execute_query.assert_awaited_once()


async def test_falkor_driver_wires_graph_operations_interface():
    """The driver must expose a non-None interface so classmethods dispatch to ops."""
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    with patch.object(FalkorDriver, 'build_indices_and_constraints', new=AsyncMock()):
        driver = FalkorDriver(falkor_db=MagicMock())
    assert driver.graph_operations_interface is not None


async def test_episodic_get_by_group_ids_classmethod_routes_via_interface():
    """EpisodicNode.get_by_group_ids must reach the Falkor ops routing path.

    Regression for the interface never being wired: the classmethod's
    ``if driver.graph_operations_interface`` branch was always skipped, so
    non-default groups returned [].
    """
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.nodes import EpisodicNode

    with patch.object(FalkorDriver, 'build_indices_and_constraints', new=AsyncMock()):
        driver = FalkorDriver(falkor_db=MagicMock())
    base, child = _make_falkor_driver()
    driver.clone = base.clone
    driver.execute_query = base.execute_query

    await EpisodicNode.get_by_group_ids(driver, ['tenant_b'], limit=10)

    base.clone.assert_called_once_with(database='tenant_b')
    child.execute_query.assert_awaited_once()
    base.execute_query.assert_not_called()


async def test_unmapped_facade_hooks_raise_not_implemented():
    """Unsupported hooks must raise NotImplementedError so caller fallbacks run.

    ``remove_communities``, ``node_get_by_names``, and the two saga hooks are
    absent from ``_METHOD_MAP``. Raising AttributeError on attribute access
    skipped the callers' ``except NotImplementedError`` fallbacks.
    """
    from graphiti_core.driver.falkordb.operations.graph_operations_interface import (
        FalkorGraphOperationsInterface,
    )

    interface = FalkorGraphOperationsInterface(driver=MagicMock())
    for name in (
        'remove_communities',
        'node_get_by_names',
        'saga_get_previous_episode_uuid',
        'saga_get_episode_contents',
    ):
        hook = getattr(interface, name)
        with pytest.raises(NotImplementedError):
            await hook()


async def test_remove_communities_falls_back_to_graph_ops():
    from graphiti_core.utils.maintenance.community_operations import remove_communities

    driver = _wired_falkor_driver()
    driver.graph_ops.remove_communities = AsyncMock()

    await remove_communities(driver, group_ids=['group_a'])

    driver.graph_ops.remove_communities.assert_awaited_once_with(driver, group_ids=['group_a'])
    driver.execute_query.assert_not_awaited()


async def test_entity_node_get_by_names_falls_back_to_query():
    from graphiti_core.nodes import EntityNode

    driver = _wired_falkor_driver()

    result = await EntityNode.get_by_names(driver, ['Alice'], ['group_a'])

    driver.execute_query.assert_awaited_once()
    assert result == []


async def test_saga_hooks_fall_back_to_query_and_none_sentinel():
    from graphiti_core.graphiti import Graphiti

    driver = _wired_falkor_driver()
    driver.execute_query = AsyncMock(return_value=([{'uuid': 'ep-prev'}], None, None))
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = driver

    previous = await graphiti._saga_get_previous_episode_uuid('saga-1', 'ep-now')
    assert previous == 'ep-prev'
    driver.execute_query.assert_awaited_once()
    query = driver.execute_query.await_args.args[0]
    assert 'HAS_EPISODE' in query

    driver.execute_query.reset_mock()
    sentinel = await graphiti._saga_get_episode_contents('saga-1')
    assert sentinel is None
    driver.execute_query.assert_not_awaited()


async def test_summarize_saga_uses_query_fallback_when_interface_returns_none():
    from graphiti_core.graphiti import Graphiti
    from graphiti_core.nodes import SagaNode

    driver = _wired_falkor_driver()
    saga = SagaNode(
        uuid='saga-1',
        name='s',
        group_id='g',
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = driver

    with patch.object(SagaNode, 'get_by_uuid', new=AsyncMock(return_value=saga)):
        result = await graphiti.summarize_saga('saga-1')

    driver.execute_query.assert_awaited_once()
    query = driver.execute_query.await_args.args[0]
    assert 'HAS_EPISODE' in query
    assert result is saga
    assert saga._summary_episodes_fetched == 0
