from graphiti_core.utils.maintenance.community_operations import Neighbor, label_propagation


def test_label_propagation_terminates_on_bipartite_oscillation() -> None:
    projection = {
        'a': [Neighbor(node_uuid='b', edge_count=1)],
        'b': [Neighbor(node_uuid='a', edge_count=1)],
    }

    clusters = label_propagation(projection, max_iterations=3)

    assert sorted(uuid for cluster in clusters for uuid in cluster) == ['a', 'b']


def test_label_propagation_ignores_dangling_neighbors() -> None:
    projection = {
        'a': [Neighbor(node_uuid='missing', edge_count=10)],
        'b': [],
    }

    clusters = label_propagation(projection)

    assert sorted(sorted(cluster) for cluster in clusters) == [['a'], ['b']]
