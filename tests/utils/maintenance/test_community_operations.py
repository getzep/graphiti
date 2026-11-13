import threading

import pytest

from graphiti_core.utils.maintenance.community_operations import Neighbor, label_propagation


def _run_label_propagation_with_timeout(
    projection: dict[str, list[Neighbor]], timeout_seconds: float = 10.0
) -> list[list[str]]:
    """Run label_propagation in a daemon thread so a non-terminating regression fails
    the test instead of hanging the test process."""
    result: dict[str, list[list[str]]] = {}

    def target():
        result['clusters'] = label_propagation(projection)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        pytest.fail(
            f'label_propagation did not terminate within {timeout_seconds}s '
            '(infinite-loop regression)'
        )

    return result['clusters']


def _sorted_clusters(clusters: list[list[str]]) -> list[list[str]]:
    return sorted(sorted(cluster) for cluster in clusters)


def test_parallel_edge_pair_terminates_and_merges():
    """Regression test: two nodes joined by parallel edges (edge_count > 1) must not
    oscillate forever.

    With synchronous label updates, each node adopts the other's label every round
    (edge_count >= 2 passes the plurality gate), so the labels swap indefinitely and
    the loop never converges. Asynchronous (in-place) updates converge: the pair must
    merge into a single community.
    """
    projection = {
        'node_a': [Neighbor(node_uuid='node_b', edge_count=2)],
        'node_b': [Neighbor(node_uuid='node_a', edge_count=2)],
    }

    clusters = _run_label_propagation_with_timeout(projection)

    assert _sorted_clusters(clusters) == [['node_a', 'node_b']]


def test_single_edge_pair_merges():
    projection = {
        'node_a': [Neighbor(node_uuid='node_b', edge_count=1)],
        'node_b': [Neighbor(node_uuid='node_a', edge_count=1)],
    }

    clusters = _run_label_propagation_with_timeout(projection)

    assert _sorted_clusters(clusters) == [['node_a', 'node_b']]


def test_triangle_merges_into_single_community():
    projection = {
        'node_a': [
            Neighbor(node_uuid='node_b', edge_count=1),
            Neighbor(node_uuid='node_c', edge_count=1),
        ],
        'node_b': [
            Neighbor(node_uuid='node_a', edge_count=1),
            Neighbor(node_uuid='node_c', edge_count=1),
        ],
        'node_c': [
            Neighbor(node_uuid='node_a', edge_count=1),
            Neighbor(node_uuid='node_b', edge_count=1),
        ],
    }

    clusters = _run_label_propagation_with_timeout(projection)

    assert _sorted_clusters(clusters) == [['node_a', 'node_b', 'node_c']]


def test_isolated_nodes_stay_in_own_communities():
    projection: dict[str, list[Neighbor]] = {
        'node_a': [],
        'node_b': [],
    }

    clusters = _run_label_propagation_with_timeout(projection)

    assert _sorted_clusters(clusters) == [['node_a'], ['node_b']]


def test_empty_projection_returns_no_clusters():
    assert label_propagation({}) == []


def test_disconnected_components_form_separate_communities():
    projection = {
        'node_a': [Neighbor(node_uuid='node_b', edge_count=2)],
        'node_b': [Neighbor(node_uuid='node_a', edge_count=2)],
        'node_c': [Neighbor(node_uuid='node_d', edge_count=2)],
        'node_d': [Neighbor(node_uuid='node_c', edge_count=2)],
    }

    clusters = _run_label_propagation_with_timeout(projection)

    assert _sorted_clusters(clusters) == [['node_a', 'node_b'], ['node_c', 'node_d']]
