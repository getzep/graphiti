import logging
import threading

import pytest

from graphiti_core.driver.operations.graph_utils import Neighbor, label_propagation
from graphiti_core.utils.maintenance.community_operations import (
    Neighbor as MaintenanceNeighbor,
)
from graphiti_core.utils.maintenance.community_operations import (
    label_propagation as maintenance_label_propagation,
)

TIMEOUT_SECONDS = 5.0


def _two_node_flip(neighbor_cls):
    """Minimal non-converging case: two nodes that swap communities every pass.

    With edge_count > 1, ``candidate_rank > 1`` holds, so each node adopts its
    neighbour's community rather than falling back to the monotonic
    ``max(candidate, curr)`` tie-break. The two assignments then swap forever.
    """
    return {
        'x': [neighbor_cls(node_uuid='y', edge_count=2)],
        'y': [neighbor_cls(node_uuid='x', edge_count=2)],
    }


def _balanced_triangle(neighbor_cls):
    """Three mutually connected nodes with equal weights: no plurality winner."""
    return {
        'a': [
            neighbor_cls(node_uuid='b', edge_count=2),
            neighbor_cls(node_uuid='c', edge_count=2),
        ],
        'b': [
            neighbor_cls(node_uuid='a', edge_count=2),
            neighbor_cls(node_uuid='c', edge_count=2),
        ],
        'c': [
            neighbor_cls(node_uuid='a', edge_count=2),
            neighbor_cls(node_uuid='b', edge_count=2),
        ],
    }


def _converging_triangle(neighbor_cls):
    """edge_count=1 throughout, so the max() tie-break converges in a few passes."""
    return {
        'a': [
            neighbor_cls(node_uuid='b', edge_count=1),
            neighbor_cls(node_uuid='c', edge_count=1),
        ],
        'b': [
            neighbor_cls(node_uuid='a', edge_count=1),
            neighbor_cls(node_uuid='c', edge_count=1),
        ],
        'c': [
            neighbor_cls(node_uuid='a', edge_count=1),
            neighbor_cls(node_uuid='b', edge_count=1),
        ],
    }


def _run_with_timeout(fn):
    """Run fn() on a worker thread and fail if it does not return in time.

    A non-terminating run cannot be interrupted, so the thread is daemonised and
    the test fails rather than hanging the suite.
    """
    box: dict[str, object] = {}

    def target():
        try:
            box['value'] = fn()
        except Exception as exc:  # pragma: no cover - re-raised below
            box['exc'] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(TIMEOUT_SECONDS)
    if thread.is_alive():
        pytest.fail(f'label_propagation did not terminate within {TIMEOUT_SECONDS}s')
    if 'exc' in box:
        raise box['exc']  # type: ignore[misc]
    return box['value']


def _assert_partitions_every_node(partition, expected_nodes):
    seen: set[str] = set()
    for cluster in partition:
        for node in cluster:
            assert node not in seen, f'node {node!r} appears in more than one cluster'
            seen.add(node)
    assert seen == expected_nodes


@pytest.mark.parametrize(
    ('func', 'neighbor_cls'),
    [
        (label_propagation, Neighbor),
        (maintenance_label_propagation, MaintenanceNeighbor),
    ],
    ids=['graph_utils', 'community_operations'],
)
@pytest.mark.parametrize(
    ('builder', 'nodes'),
    [
        (_two_node_flip, {'x', 'y'}),
        (_balanced_triangle, {'a', 'b', 'c'}),
    ],
    ids=['two_node_flip', 'balanced_triangle'],
)
def test_terminates_on_non_converging_graphs(func, neighbor_cls, builder, nodes):
    partition = _run_with_timeout(lambda: func(builder(neighbor_cls)))
    _assert_partitions_every_node(partition, nodes)


@pytest.mark.parametrize(
    ('func', 'neighbor_cls'),
    [
        (label_propagation, Neighbor),
        (maintenance_label_propagation, MaintenanceNeighbor),
    ],
    ids=['graph_utils', 'community_operations'],
)
def test_converging_graph_is_unaffected_by_the_cap(func, neighbor_cls, caplog):
    """A graph that settles must exit via no_change, not via the iteration cap."""
    with caplog.at_level(logging.WARNING):
        partition = func(_converging_triangle(neighbor_cls))

    assert len(partition) == 1
    _assert_partitions_every_node(partition, {'a', 'b', 'c'})
    assert not [r for r in caplog.records if 'did not converge' in r.getMessage()]


@pytest.mark.parametrize(
    ('func', 'neighbor_cls'),
    [
        (label_propagation, Neighbor),
        (maintenance_label_propagation, MaintenanceNeighbor),
    ],
    ids=['graph_utils', 'community_operations'],
)
def test_warns_when_the_cap_is_reached(func, neighbor_cls, caplog):
    with caplog.at_level(logging.WARNING):
        func(_two_node_flip(neighbor_cls))

    assert [r for r in caplog.records if 'did not converge' in r.getMessage()]
