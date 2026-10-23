"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import faulthandler

import pytest

from graphiti_core.driver.operations.graph_utils import Neighbor
from graphiti_core.driver.operations.graph_utils import (
    label_propagation as label_propagation_driver,
)
from graphiti_core.utils.maintenance import community_operations
from graphiti_core.utils.maintenance.community_operations import (
    label_propagation as label_propagation_maintenance,
)

# Last-resort guard: label_propagation used to loop forever on oscillating
# graphs (#402, #1355). If a regression reintroduces a non-terminating loop,
# dump tracebacks and abort the test process instead of hanging CI.
HANG_TIMEOUT_SECONDS = 10


@pytest.fixture(autouse=True)
def _abort_on_hang():
    faulthandler.dump_traceback_later(HANG_TIMEOUT_SECONDS, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def test_both_copies_are_the_same_function():
    # The bug was fixed twice historically and each fix patched only one copy
    # (#402). Both import paths must expose a single shared implementation so
    # they cannot drift apart again.
    assert community_operations.label_propagation is label_propagation_driver
    assert label_propagation_maintenance is label_propagation_driver


def test_parallel_edge_pair_terminates_as_single_cluster():
    # Two nodes joined by parallel edges (edge_count >= 2) oscillated forever
    # under synchronous updates: each pass swapped their labels (#402, #1355).
    projection = {
        'A': [Neighbor(node_uuid='B', edge_count=2)],
        'B': [Neighbor(node_uuid='A', edge_count=2)],
    }

    clusters = label_propagation_driver(projection)

    assert sorted(sorted(cluster) for cluster in clusters) == [['A', 'B']]


def test_chain_above_100_nodes_converges_fully():
    # A chain needs O(|V|) propagation rounds. A fixed 100-iteration cap
    # truncates a 150-node chain into ~50 communities; the cap must grow with
    # the graph (max(100, len(projection))) so the chain fully merges (#402).
    size = 150
    projection: dict[str, list[Neighbor]] = {}
    for i in range(size):
        neighbors = []
        if i > 0:
            neighbors.append(Neighbor(node_uuid=f'n{i - 1}', edge_count=1))
        if i < size - 1:
            neighbors.append(Neighbor(node_uuid=f'n{i + 1}', edge_count=1))
        projection[f'n{i}'] = neighbors

    clusters = label_propagation_driver(projection)

    assert len(clusters) == 1
    assert sorted(clusters[0]) == sorted(f'n{i}' for i in range(size))


def test_bipartite_complete_graph_terminates():
    # K2,2 with multi-edges hung on the synchronous update rule (#402).
    projection = {
        'A': [Neighbor(node_uuid='C', edge_count=2), Neighbor(node_uuid='D', edge_count=2)],
        'B': [Neighbor(node_uuid='C', edge_count=2), Neighbor(node_uuid='D', edge_count=2)],
        'C': [Neighbor(node_uuid='A', edge_count=2), Neighbor(node_uuid='B', edge_count=2)],
        'D': [Neighbor(node_uuid='A', edge_count=2), Neighbor(node_uuid='B', edge_count=2)],
    }

    clusters = label_propagation_driver(projection)

    assert len(clusters) == 1


def test_disconnected_parallel_edge_pairs_stay_separate():
    projection = {
        'A': [Neighbor(node_uuid='B', edge_count=2)],
        'B': [Neighbor(node_uuid='A', edge_count=2)],
        'C': [Neighbor(node_uuid='D', edge_count=2)],
        'D': [Neighbor(node_uuid='C', edge_count=2)],
    }

    clusters = label_propagation_driver(projection)

    assert sorted(sorted(cluster) for cluster in clusters) == [['A', 'B'], ['C', 'D']]


def test_star_graph_with_parallel_edges_converges():
    projection = {
        'hub': [Neighbor(node_uuid=f'leaf{i}', edge_count=2) for i in range(4)],
        **{f'leaf{i}': [Neighbor(node_uuid='hub', edge_count=2)] for i in range(4)},
    }

    clusters = label_propagation_driver(projection)

    assert len(clusters) == 1


def test_result_is_independent_of_projection_insertion_order():
    # The projection is built from DB query results whose order can vary, so
    # propagation must visit nodes in a stable order: same graph inserted in a
    # different order must yield identical communities.
    edges = {
        'A': [('B', 2)],
        'B': [('A', 2)],
        'C': [('D', 1)],
        'D': [('C', 1)],
    }

    def build(order: list[str]) -> dict[str, list[Neighbor]]:
        return {
            uuid: [
                Neighbor(node_uuid=neighbor, edge_count=count) for neighbor, count in edges[uuid]
            ]
            for uuid in order
        }

    forward = label_propagation_driver(build(['A', 'B', 'C', 'D']))
    backward = label_propagation_driver(build(['D', 'C', 'B', 'A']))

    assert sorted(sorted(cluster) for cluster in forward) == sorted(
        sorted(cluster) for cluster in backward
    )
