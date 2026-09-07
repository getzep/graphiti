"""Unit tests for FalkorDB per-group-id query routing in the operations layer.

These cover the *namespace/ops* read path (e.g. ``graphiti.nodes.episode.
get_by_group_ids``), which delegates to the Falkor ``*Operations`` classes with
the base driver (pointed at ``default_db``). FalkorDB stores each ``group_id`` in
its own physical graph, so a group-scoped read must clone the driver to that
group's graph or it queries an empty default database and returns nothing.

The tests use a ``spec=GraphDriver`` mock, so they need no live FalkorDB and run
in CI regardless of whether the ``falkordb`` package is installed.
"""

from unittest.mock import AsyncMock, MagicMock

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
