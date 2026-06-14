from graphiti_core.utils.maintenance.community_operations import Neighbor, label_propagation


def _cluster_sets(clusters: list[list[str]]) -> set[frozenset[str]]:
    return {frozenset(cluster) for cluster in clusters}


def test_label_propagation_terminates_for_repeated_entity_edges():
    projection = {
        'node-a': [Neighbor(node_uuid='node-b', edge_count=2)],
        'node-b': [Neighbor(node_uuid='node-a', edge_count=2)],
    }

    clusters = label_propagation(projection)

    assert _cluster_sets(clusters) == {frozenset({'node-a', 'node-b'})}


def test_label_propagation_keeps_disconnected_components_separate():
    projection = {
        'node-a': [Neighbor(node_uuid='node-b', edge_count=2)],
        'node-b': [Neighbor(node_uuid='node-a', edge_count=2)],
        'node-c': [Neighbor(node_uuid='node-d', edge_count=3)],
        'node-d': [Neighbor(node_uuid='node-c', edge_count=3)],
    }

    clusters = label_propagation(projection)

    assert _cluster_sets(clusters) == {
        frozenset({'node-a', 'node-b'}),
        frozenset({'node-c', 'node-d'}),
    }
