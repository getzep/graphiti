import logging
import threading

from graphiti_core.utils.maintenance.community_operations import Neighbor, label_propagation


def _call_with_timeout(func, timeout: float = 10.0):
    """Run ``func`` in a daemon thread and return (result, finished).

    ``label_propagation`` is a pure, CPU-bound function, so a thread that is
    stuck in an unbounded loop cannot be interrupted; the daemon thread is
    abandoned when the test process exits. This lets a non-terminating call
    fail the assertion quickly instead of hanging the whole test session.
    """
    box: dict = {}

    def target():
        box['result'] = func()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return None, False
    return box['result'], True


def _non_convergent_projection() -> dict[str, list[Neighbor]]:
    """A small symmetric weighted graph on which synchronous label propagation
    oscillates with period two and never converges.

    Starting from the one-community-per-node seed, every pass flips the two
    halves of the graph, so ``no_change`` is never reached. Without an
    iteration bound the propagation loop runs forever on this input.
    """
    edges = {
        'n0': [('n1', 2), ('n2', 2), ('n3', 1)],
        'n1': [('n0', 2), ('n2', 3), ('n3', 2)],
        'n2': [('n0', 2), ('n1', 3)],
        'n3': [('n0', 1), ('n1', 2)],
    }
    return {
        uuid: [Neighbor(node_uuid=nb, edge_count=weight) for nb, weight in neighbors]
        for uuid, neighbors in edges.items()
    }


def test_label_propagation_terminates_on_non_convergent_graph():
    projection = _non_convergent_projection()

    clusters, finished = _call_with_timeout(lambda: label_propagation(projection))

    assert finished, 'label_propagation did not terminate on a non-convergent graph'

    # The result must still be a valid partition: every node appears in exactly
    # one cluster and no cluster is empty.
    assigned = [uuid for cluster in clusters for uuid in cluster]
    assert sorted(assigned) == sorted(projection.keys())
    assert all(len(cluster) > 0 for cluster in clusters)


def test_label_propagation_warns_when_iteration_cap_is_reached(caplog):
    projection = _non_convergent_projection()

    with caplog.at_level(
        logging.WARNING, logger='graphiti_core.utils.maintenance.community_operations'
    ):
        _, finished = _call_with_timeout(lambda: label_propagation(projection))

    assert finished
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_label_propagation_still_converges_on_separable_graph():
    # Two disconnected pairs form two obvious communities; propagation converges
    # normally, so the iteration bound must not change the result.
    projection: dict[str, list[Neighbor]] = {
        'a': [Neighbor(node_uuid='b', edge_count=1)],
        'b': [Neighbor(node_uuid='a', edge_count=1)],
        'c': [Neighbor(node_uuid='d', edge_count=1)],
        'd': [Neighbor(node_uuid='c', edge_count=1)],
    }

    clusters = label_propagation(projection)

    partition = {frozenset(cluster) for cluster in clusters}
    assert partition == {frozenset({'a', 'b'}), frozenset({'c', 'd'})}
