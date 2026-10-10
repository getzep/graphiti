import pytest

from graphiti_core.driver.operations import graph_utils
from graphiti_core.utils.maintenance import community_operations

modules = [community_operations, graph_utils]


def normalize(clusters: list[list[str]]) -> list[list[str]]:
    return sorted(sorted(cluster) for cluster in clusters)


@pytest.mark.parametrize('module', modules)
def test_label_propagation_converges_on_tied_weight_cycle(module):
    # Two connected nodes whose weights tie would oscillate forever under
    # synchronous updates (each round the nodes swap labels).
    projection = {
        'a': [module.Neighbor(node_uuid='b', edge_count=2)],
        'b': [module.Neighbor(node_uuid='a', edge_count=2)],
    }

    clusters = module.label_propagation(projection)

    assert normalize(clusters) == [['a', 'b']]


@pytest.mark.parametrize('module', modules)
def test_label_propagation_partitions_all_nodes_on_cyclic_graph(module):
    projection = {
        'a': [
            module.Neighbor(node_uuid='b', edge_count=2),
            module.Neighbor(node_uuid='c', edge_count=2),
        ],
        'b': [
            module.Neighbor(node_uuid='a', edge_count=2),
            module.Neighbor(node_uuid='c', edge_count=1),
        ],
        'c': [
            module.Neighbor(node_uuid='a', edge_count=2),
            module.Neighbor(node_uuid='b', edge_count=1),
        ],
        'd': [module.Neighbor(node_uuid='d', edge_count=1)],
    }

    clusters = module.label_propagation(projection, seed=7)

    members = [node for cluster in clusters for node in cluster]
    assert sorted(members) == ['a', 'b', 'c', 'd']


@pytest.mark.parametrize('module', modules)
def test_label_propagation_is_deterministic_for_same_seed(module):
    projection = {
        'hub': [
            module.Neighbor(node_uuid='x', edge_count=2),
            module.Neighbor(node_uuid='y', edge_count=2),
            module.Neighbor(node_uuid='z', edge_count=2),
        ],
        'x': [
            module.Neighbor(node_uuid='hub', edge_count=2),
            module.Neighbor(node_uuid='y', edge_count=1),
        ],
        'y': [
            module.Neighbor(node_uuid='hub', edge_count=2),
            module.Neighbor(node_uuid='x', edge_count=1),
        ],
        'z': [
            module.Neighbor(node_uuid='hub', edge_count=2),
            module.Neighbor(node_uuid='w', edge_count=2),
        ],
        'w': [
            module.Neighbor(node_uuid='z', edge_count=2),
            module.Neighbor(node_uuid='hub', edge_count=1),
        ],
    }

    first = module.label_propagation(projection, seed=42)
    second = module.label_propagation(projection, seed=42)

    assert normalize(first) == normalize(second)
    assert sorted(node for cluster in first for node in cluster) == [
        'hub',
        'w',
        'x',
        'y',
        'z',
    ]
