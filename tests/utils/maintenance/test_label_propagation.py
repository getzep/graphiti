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

import pytest

from graphiti_core.driver.operations import graph_utils
from graphiti_core.driver.operations.graph_utils import Neighbor
from graphiti_core.utils.maintenance import community_operations


def _clusters(result: list[list[str]]) -> set[frozenset[str]]:
    return {frozenset(cluster) for cluster in result}


def _pair(edge_count: int) -> dict[str, list[Neighbor]]:
    return {
        'a': [Neighbor(node_uuid='b', edge_count=edge_count)],
        'b': [Neighbor(node_uuid='a', edge_count=edge_count)],
    }


IMPLEMENTATIONS = [
    pytest.param(graph_utils.label_propagation, id='graph_utils'),
    pytest.param(community_operations.label_propagation, id='community_operations'),
]


@pytest.mark.parametrize('label_propagation', IMPLEMENTATIONS)
def test_parallel_edges_between_two_nodes_terminate(label_propagation):
    # Two nodes joined only by parallel edges used to swap labels forever.
    assert _clusters(label_propagation(_pair(edge_count=2))) == {frozenset({'a', 'b'})}
    assert _clusters(label_propagation(_pair(edge_count=5))) == {frozenset({'a', 'b'})}


@pytest.mark.parametrize('label_propagation', IMPLEMENTATIONS)
def test_single_edge_pair_and_triangle(label_propagation):
    assert _clusters(label_propagation(_pair(edge_count=1))) == {frozenset({'a', 'b'})}

    triangle = {
        'a': [Neighbor(node_uuid='b', edge_count=1), Neighbor(node_uuid='c', edge_count=1)],
        'b': [Neighbor(node_uuid='a', edge_count=1), Neighbor(node_uuid='c', edge_count=1)],
        'c': [Neighbor(node_uuid='a', edge_count=1), Neighbor(node_uuid='b', edge_count=1)],
    }
    assert _clusters(label_propagation(triangle)) == {frozenset({'a', 'b', 'c'})}


@pytest.mark.parametrize('label_propagation', IMPLEMENTATIONS)
def test_disconnected_components_stay_separate(label_propagation):
    projection = {
        'a': [Neighbor(node_uuid='b', edge_count=3)],
        'b': [Neighbor(node_uuid='a', edge_count=3)],
        'c': [Neighbor(node_uuid='d', edge_count=2)],
        'd': [Neighbor(node_uuid='c', edge_count=2)],
        'e': [],
    }
    assert _clusters(label_propagation(projection)) == {
        frozenset({'a', 'b'}),
        frozenset({'c', 'd'}),
        frozenset({'e'}),
    }


@pytest.mark.parametrize('label_propagation', IMPLEMENTATIONS)
def test_empty_projection(label_propagation):
    assert label_propagation({}) == []
