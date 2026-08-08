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

from graphiti_core.utils.maintenance.community_operations import (
    Neighbor,
    label_propagation,
)


def _as_cluster_sets(clusters: list[list[str]]) -> set[frozenset[str]]:
    return {frozenset(cluster) for cluster in clusters}


def test_label_propagation_converges_to_single_community():
    # A fully-connected triangle collapses into one community.
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

    clusters = label_propagation(projection)

    assert _as_cluster_sets(clusters) == {frozenset({'a', 'b', 'c'})}


def test_label_propagation_keeps_isolated_node():
    projection = {'lone': []}

    assert label_propagation(projection) == [['lone']]


def test_label_propagation_terminates_on_oscillating_projection():
    # These two nodes swap communities on every iteration; without the
    # iteration cap label_propagation never returns for this projection.
    projection = {
        'a': [Neighbor(node_uuid='b', edge_count=2)],
        'b': [Neighbor(node_uuid='a', edge_count=2)],
    }

    clusters = label_propagation(projection)

    assert _as_cluster_sets(clusters) == {frozenset({'a'}), frozenset({'b'})}
