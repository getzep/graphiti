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

from graphiti_core.driver.operations.graph_utils import (
    Neighbor as DriverNeighbor,
)
from graphiti_core.driver.operations.graph_utils import (
    label_propagation as driver_label_propagation,
)
from graphiti_core.utils.maintenance.community_operations import (
    Neighbor as MaintenanceNeighbor,
)
from graphiti_core.utils.maintenance.community_operations import (
    label_propagation as maintenance_label_propagation,
)

# Two duplicate copies of label_propagation exist in the codebase: one in
# graphiti_core.driver.operations.graph_utils (used by every database driver
# via graph_operations_interface) and one in
# graphiti_core.utils.maintenance.community_operations (used as the legacy
# fallback). Run every test against both so any fix lands in lockstep and
# does not silently regress one site.
LABEL_PROPAGATION_IMPLS = [
    pytest.param(driver_label_propagation, DriverNeighbor, id='driver'),
    pytest.param(maintenance_label_propagation, MaintenanceNeighbor, id='maintenance'),
]


def _undirected(
    cls,
    edges: list[tuple[str, str, int]],
    isolated: list[str] | None = None,
) -> dict[str, list]:
    """Build a symmetric `projection` dict from undirected weighted edges.

    The Cypher in get_community_clusters matches RELATES_TO undirected, so
    a real projection contains both (u -> v) and (v -> u) entries. Tests
    that omit one direction would not exercise the shape of input the
    function actually receives in production.
    """
    projection: dict[str, list] = {}
    for u, v, w in edges:
        projection.setdefault(u, []).append(cls(node_uuid=v, edge_count=w))
        projection.setdefault(v, []).append(cls(node_uuid=u, edge_count=w))
    for n in isolated or []:
        projection.setdefault(n, [])
    return projection


@pytest.mark.parametrize('impl, neighbor_cls', LABEL_PROPAGATION_IMPLS)
def test_label_propagation_terminates_on_oscillation_inducing_graph(impl, neighbor_cls):
    """Regression test for non-convergence under synchronous label
    propagation. The original implementation reads from a snapshot and
    writes to a parallel map (synchronous LPA), then swaps. Synchronous
    LPA is known to oscillate on graphs with ties between weighted
    candidate communities and a tiebreak that pulls labels in different
    directions on alternating rounds. This 5-node graph exhibits exactly
    that pathology with the original tiebreak (max(candidate, current)
    when candidate_rank <= 1, raw candidate when > 1) — labels for the
    central X / Y / Z triangle alternate between two states forever.

    The fix must guarantee termination on this input. Output correctness
    is checked separately; this test only asserts the function returns.
    """
    projection = _undirected(
        neighbor_cls,
        edges=[
            ('X', 'Y', 2),
            ('X', 'Z', 2),
            ('Y', 'A', 1),
            ('Z', 'B', 1),
        ],
    )

    clusters = impl(projection)

    flat = sorted([uuid for cluster in clusters for uuid in cluster])
    assert flat == ['A', 'B', 'X', 'Y', 'Z']


@pytest.mark.parametrize('impl, neighbor_cls', LABEL_PROPAGATION_IMPLS)
def test_label_propagation_assigns_every_node_exactly_once(impl, neighbor_cls):
    """Output invariant: every input UUID appears in exactly one cluster.
    No node is dropped, no node is duplicated across clusters.
    """
    projection = _undirected(
        neighbor_cls,
        edges=[
            ('a', 'b', 1),
            ('b', 'c', 1),
            ('c', 'd', 1),
            ('d', 'e', 1),
            ('e', 'a', 1),
        ],
    )

    clusters = impl(projection)

    flat = [uuid for cluster in clusters for uuid in cluster]
    assert sorted(flat) == ['a', 'b', 'c', 'd', 'e']
    assert len(flat) == len(set(flat))


@pytest.mark.parametrize('impl, neighbor_cls', LABEL_PROPAGATION_IMPLS)
def test_label_propagation_disconnected_graph_yields_singletons(impl, neighbor_cls):
    """Three nodes with no edges between them must produce three
    singleton clusters. A node with an empty neighbour list has no
    candidate community to adopt and must keep its initial assignment.
    """
    projection = _undirected(neighbor_cls, edges=[], isolated=['a', 'b', 'c'])

    clusters = impl(projection)

    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 1, 1]
    flat = sorted([uuid for cluster in clusters for uuid in cluster])
    assert flat == ['a', 'b', 'c']


@pytest.mark.parametrize('impl, neighbor_cls', LABEL_PROPAGATION_IMPLS)
def test_label_propagation_complete_graph_collapses_to_one_cluster(impl, neighbor_cls):
    """A fully-connected graph of 4 nodes must converge to a single
    cluster: every node sees every other node voting for the same shared
    plurality, so all nodes adopt the same community.
    """
    nodes = ['a', 'b', 'c', 'd']
    edges = [(u, v, 1) for i, u in enumerate(nodes) for v in nodes[i + 1 :]]
    projection = _undirected(neighbor_cls, edges=edges)

    clusters = impl(projection)

    assert len(clusters) == 1
    assert sorted(clusters[0]) == sorted(nodes)


@pytest.mark.parametrize('impl, neighbor_cls', LABEL_PROPAGATION_IMPLS)
def test_label_propagation_two_disjoint_components_yield_two_clusters(impl, neighbor_cls):
    """Two disjoint complete subgraphs of size 3 must produce two
    clusters of 3 each — the algorithm must not bridge components.
    """
    projection = _undirected(
        neighbor_cls,
        edges=[
            ('a', 'b', 1),
            ('b', 'c', 1),
            ('a', 'c', 1),
            ('x', 'y', 1),
            ('y', 'z', 1),
            ('x', 'z', 1),
        ],
    )

    clusters = impl(projection)

    cluster_sets = [frozenset(c) for c in clusters]
    assert frozenset({'a', 'b', 'c'}) in cluster_sets
    assert frozenset({'x', 'y', 'z'}) in cluster_sets
    assert len(clusters) == 2


@pytest.mark.parametrize('impl, neighbor_cls', LABEL_PROPAGATION_IMPLS)
def test_label_propagation_is_deterministic(impl, neighbor_cls):
    """Same input, same output across repeated calls. Reproducibility
    matters for downstream community-summary caching and for tests that
    depend on cluster identity.
    """
    projection = _undirected(
        neighbor_cls,
        edges=[
            ('a', 'b', 2),
            ('b', 'c', 1),
            ('c', 'd', 2),
            ('d', 'e', 1),
            ('e', 'a', 1),
        ],
    )

    first = impl(projection)
    second = impl(projection)

    canonical_first = sorted(sorted(c) for c in first)
    canonical_second = sorted(sorted(c) for c in second)
    assert canonical_first == canonical_second
