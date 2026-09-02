"""Regression tests for label_propagation() (#1355).

label_propagation() and Neighbor live in graphiti_core.driver.operations.graph_utils -
every driver-specific get_community_clusters() (Neo4j/FalkorDB/Kuzu/Neptune) imports
them from there, and graphiti_core.utils.maintenance.community_operations re-exports
the same objects rather than defining its own copy (see test_label_propagation_is_not_duplicated
below for why that matters). label_propagation() is a pure function (no DB/LLM
dependency), so it is tested directly rather than through the async
get_community_clusters/build_communities pipeline.
"""

import queue
import threading
from collections.abc import Callable
from typing import Any, TypeVar

import pytest

from graphiti_core.driver.falkordb.operations.graph_ops import (
    label_propagation as falkordb_label_propagation,
)
from graphiti_core.driver.kuzu.operations.graph_ops import (
    label_propagation as kuzu_label_propagation,
)
from graphiti_core.driver.neo4j.operations.graph_ops import (
    label_propagation as neo4j_label_propagation,
)
from graphiti_core.driver.neptune.operations.graph_ops import (
    label_propagation as neptune_label_propagation,
)
from graphiti_core.driver.operations.graph_utils import Neighbor, label_propagation
from graphiti_core.utils.maintenance import community_operations

T = TypeVar('T')


def test_label_propagation_is_not_duplicated():
    """Guard against the exact regression this fix started as.

    The first pass at #1355 fixed graphiti_core.utils.maintenance.community_operations's
    own copy of label_propagation(), but every real backend (Neo4j/FalkorDB/Kuzu/
    Neptune) actually calls the separate copy in graphiti_core.driver.operations.graph_utils
    via their own get_community_clusters(), so the fix never reached the code path
    that matters for real usage. Fixed by deduplicating: community_operations now
    imports the canonical function instead of defining its own. This test asserts
    all five import sites resolve to the exact same function object, so a future
    re-introduction of a second copy fails loudly here instead of silently
    reintroducing the hang for real backends.
    """
    assert community_operations.label_propagation is label_propagation
    assert neo4j_label_propagation is label_propagation
    assert falkordb_label_propagation is label_propagation
    assert kuzu_label_propagation is label_propagation
    assert neptune_label_propagation is label_propagation


# label_propagation() with the pre-fix implementation never returns for the
# repro case below, so the whole test process would hang without a hard
# timeout. A daemon thread lets the test fail fast (raising TimeoutError)
# instead of blocking the suite forever; if the bug regresses, at most one
# leaked daemon thread lingers, which does not block interpreter exit.
def _run_with_timeout(fn: Callable[..., T], *args: Any, timeout: float = 2.0, **kwargs: Any) -> T:
    result_queue: queue.Queue = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            result_queue.put(('ok', fn(*args, **kwargs)))
        except Exception as exc:  # noqa: BLE001 - re-raised on the test thread below
            result_queue.put(('error', exc))

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError(
            f'{fn.__name__} did not complete within {timeout}s (likely an infinite loop)'
        )

    status, value = result_queue.get_nowait()
    if status == 'error':
        raise value
    return value


def test_label_propagation_converges_with_mutual_double_edge():
    """Regression test for #1355.

    Two nodes connected by a single neighbor relationship with edge_count=2
    each start in their own community. The pre-fix synchronous update had
    both nodes swap to the other's community every round, oscillating
    forever instead of converging.
    """
    projection = {
        'a': [Neighbor(node_uuid='b', edge_count=2)],
        'b': [Neighbor(node_uuid='a', edge_count=2)],
    }

    clusters = _run_with_timeout(label_propagation, projection, timeout=2.0)

    assert len(clusters) == 1
    assert set(clusters[0]) == {'a', 'b'}


def test_label_propagation_converges_with_triangle_of_double_edges():
    """Three mutually connected nodes, each pair sharing edge_count=2.

    This is closer to the real-world repro in #1355 (multiple episodes
    referencing the same entities produce edge_count >= 2 between several
    pairs at once).
    """
    projection = {
        'a': [
            Neighbor(node_uuid='b', edge_count=2),
            Neighbor(node_uuid='c', edge_count=2),
        ],
        'b': [
            Neighbor(node_uuid='a', edge_count=2),
            Neighbor(node_uuid='c', edge_count=2),
        ],
        'c': [
            Neighbor(node_uuid='a', edge_count=2),
            Neighbor(node_uuid='b', edge_count=2),
        ],
    }

    clusters = _run_with_timeout(label_propagation, projection, timeout=2.0)

    assert len(clusters) == 1
    assert set(clusters[0]) == {'a', 'b', 'c'}


def test_label_propagation_isolated_nodes_stay_separate():
    projection = {
        'a': [],
        'b': [],
    }

    clusters = _run_with_timeout(label_propagation, projection, timeout=2.0)

    assert sorted(clusters) == [['a'], ['b']]


def test_label_propagation_joins_plurality_community():
    """A node with a clear-majority neighbor community should join it."""
    projection = {
        'center': [
            Neighbor(node_uuid='x1', edge_count=1),
            Neighbor(node_uuid='x2', edge_count=1),
            Neighbor(node_uuid='y1', edge_count=1),
        ],
        'x1': [Neighbor(node_uuid='x2', edge_count=1)],
        'x2': [Neighbor(node_uuid='x1', edge_count=1)],
        'y1': [],
    }

    clusters = _run_with_timeout(label_propagation, projection, timeout=2.0)

    cluster_containing_center = next(c for c in clusters if 'center' in c)
    assert {'x1', 'x2'}.issubset(set(cluster_containing_center))


@pytest.mark.parametrize('edge_count', [2, 3, 5, 10])
def test_label_propagation_converges_for_various_high_edge_counts(edge_count):
    """edge_count >= 2 was the reported trigger; make sure larger counts
    (e.g. many episodes mentioning the same pair) don't reintroduce the
    oscillation either."""
    projection = {
        'a': [Neighbor(node_uuid='b', edge_count=edge_count)],
        'b': [Neighbor(node_uuid='a', edge_count=edge_count)],
    }

    clusters = _run_with_timeout(label_propagation, projection, timeout=2.0)

    assert len(clusters) == 1
    assert set(clusters[0]) == {'a', 'b'}


def test_label_propagation_result_independent_of_projection_insertion_order():
    """In-place updates make each round sensitive to processing order, and
    projection is normally built from unordered DB query results. Nodes are
    processed in sorted-uuid order internally specifically so that two
    logically identical graphs handed in with different dict insertion
    orders still converge to the same community grouping.
    """
    neighbors_by_node = {
        'a': [Neighbor(node_uuid='b', edge_count=2), Neighbor(node_uuid='c', edge_count=1)],
        'b': [Neighbor(node_uuid='a', edge_count=2)],
        'c': [Neighbor(node_uuid='a', edge_count=1)],
        'd': [],
    }

    forward = {k: neighbors_by_node[k] for k in ['a', 'b', 'c', 'd']}
    reversed_order = {k: neighbors_by_node[k] for k in ['d', 'c', 'b', 'a']}

    clusters_forward = _run_with_timeout(label_propagation, forward, timeout=2.0)
    clusters_reversed = _run_with_timeout(label_propagation, reversed_order, timeout=2.0)

    normalize = lambda clusters: sorted(sorted(c) for c in clusters)  # noqa: E731
    assert normalize(clusters_forward) == normalize(clusters_reversed)
