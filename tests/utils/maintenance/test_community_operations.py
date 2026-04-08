"""Tests for label_propagation community detection.

Focuses on the oscillation-prevention fix: graphs with high-degree hub
nodes previously caused the synchronous batch implementation to loop
forever. The asynchronous form (visit nodes in shuffled order, update
the map in place) converges quickly on every case we throw at it.
"""

from __future__ import annotations

import time

import pytest

from graphiti_core.nodes import EntityNode
from graphiti_core.utils.maintenance.community_operations import (
    Neighbor,
    _select_representative_members,
    label_propagation,
)


def _make_projection(edges: list[tuple[str, str, int]]) -> dict[str, list[Neighbor]]:
    """Build an undirected projection from a weighted edge list."""
    projection: dict[str, list[Neighbor]] = {}
    for a, b, weight in edges:
        projection.setdefault(a, []).append(Neighbor(node_uuid=b, edge_count=weight))
        projection.setdefault(b, []).append(Neighbor(node_uuid=a, edge_count=weight))
    return projection


def _assert_partition(clusters: list[list[str]], expected_nodes: set[str]) -> None:
    """Every node appears exactly once across clusters."""
    seen: set[str] = set()
    for cluster in clusters:
        for node in cluster:
            assert node not in seen, f"node {node} appears in multiple clusters"
            seen.add(node)
    assert seen == expected_nodes, f"missing nodes: {expected_nodes - seen}"


def test_empty_projection_returns_empty():
    assert label_propagation({}) == []


def test_single_isolated_node():
    projection = {"a": []}
    clusters = label_propagation(projection)
    _assert_partition(clusters, {"a"})
    assert len(clusters) == 1


def test_two_disconnected_triangles():
    projection = _make_projection(
        [
            ("a1", "a2", 1),
            ("a2", "a3", 1),
            ("a3", "a1", 1),
            ("b1", "b2", 1),
            ("b2", "b3", 1),
            ("b3", "b1", 1),
        ]
    )
    clusters = label_propagation(projection)
    _assert_partition(clusters, {"a1", "a2", "a3", "b1", "b2", "b3"})
    assert len(clusters) == 2


def test_complete_graph_collapses_to_one_community():
    edges = [(f"n{i}", f"n{j}", 1) for i in range(8) for j in range(i + 1, 8)]
    projection = _make_projection(edges)
    clusters = label_propagation(projection)
    assert len(clusters) == 1
    assert len(clusters[0]) == 8


def test_hub_with_leaves_converges():
    """Regression: central hub with many leaves used to oscillate.

    The synchronous batch implementation flipped leaves between the hub's
    community and their own community every iteration, never converging.
    """
    edges = [(f"leaf{i}", "hub", 1) for i in range(20)]
    projection = _make_projection(edges)
    start = time.time()
    clusters = label_propagation(projection)
    elapsed = time.time() - start
    _assert_partition(clusters, {"hub", *(f"leaf{i}" for i in range(20))})
    assert elapsed < 1.0, f"hub graph should converge quickly; took {elapsed:.2f}s"


def test_two_stars_joined_by_bridge():
    """Two hub+leaves clusters connected by one bridge edge.

    A correct community detector should identify two communities (one per
    star). Earlier synchronous implementations could oscillate here.
    """
    edges = [
        *[(f"a_leaf{i}", "hub_a", 1) for i in range(10)],
        *[(f"b_leaf{i}", "hub_b", 1) for i in range(10)],
        ("hub_a", "hub_b", 1),
    ]
    projection = _make_projection(edges)
    clusters = label_propagation(projection)
    _assert_partition(
        clusters,
        {"hub_a", "hub_b", *(f"a_leaf{i}" for i in range(10)), *(f"b_leaf{i}" for i in range(10))},
    )
    assert len(clusters) == 2


def test_real_world_pathological_graph_converges():
    """Regression test from an observed production failure.

    A 48-node knowledge graph with a central "Threshold" node
    (uuid `d689c03c`) connected to 14+ entities caused the synchronous
    batch implementation to oscillate indefinitely — a fixed subset of
    19 nodes kept flipping between two states forever.

    This projection is a simplified version of the failing graph. With
    the synchronous implementation it never returned; the async form
    converges in milliseconds.
    """
    # Hub node with heavy ties to several satellites
    hub = "hub"
    sat_heavy = [f"sat_h{i}" for i in range(4)]  # strong connections to hub
    sat_light = [f"sat_l{i}" for i in range(10)]  # weak connections to hub

    edges: list[tuple[str, str, int]] = []
    # Strong ties: hub ↔ each heavy satellite (edge count 29)
    edges.extend((hub, sat, 29) for sat in sat_heavy)
    # Weak ties: hub ↔ each light satellite (edge count 1)
    edges.extend((hub, sat, 1) for sat in sat_light)
    # Triangle-ish ties among light satellites to create tie ambiguity
    for i in range(0, len(sat_light) - 1, 2):
        edges.append((sat_light[i], sat_light[i + 1], 1))
    # A few floating dyads that should form their own mini-communities
    edges.append(("pair_a1", "pair_a2", 1))
    edges.append(("pair_b1", "pair_b2", 1))

    projection = _make_projection(edges)

    start = time.time()
    clusters = label_propagation(projection)
    elapsed = time.time() - start

    all_nodes = {hub, *sat_heavy, *sat_light, "pair_a1", "pair_a2", "pair_b1", "pair_b2"}
    _assert_partition(clusters, all_nodes)
    assert elapsed < 1.0, f"pathological graph should converge fast; took {elapsed:.2f}s"
    # Sanity: at least one community should contain the hub and its heavy ties
    hub_cluster = next(c for c in clusters if hub in c)
    for sat in sat_heavy:
        assert sat in hub_cluster, f"{sat} should be in hub's community"


def test_deterministic_under_seed():
    """Same input produces the same partition across runs.

    The async form shuffles node order, but uses a fixed RNG seed so
    results are reproducible.
    """
    edges = [
        ("a", "b", 1),
        ("b", "c", 1),
        ("c", "a", 1),
        ("d", "e", 1),
        ("e", "f", 1),
        ("f", "d", 1),
        ("a", "d", 1),
    ]
    projection = _make_projection(edges)

    first = label_propagation(projection)
    second = label_propagation(projection)

    # Canonicalize (sort within cluster, sort list of clusters)
    def canon(cs: list[list[str]]) -> list[list[str]]:
        return sorted([sorted(c) for c in cs])

    assert canon(first) == canon(second)


@pytest.mark.parametrize("n", [50, 200])
def test_ring_graph_of_varying_sizes(n: int):
    """Rings are edge cases for label propagation."""
    edges = [(f"r{i}", f"r{(i + 1) % n}", 1) for i in range(n)]
    projection = _make_projection(edges)
    start = time.time()
    clusters = label_propagation(projection)
    elapsed = time.time() - start
    _assert_partition(clusters, {f"r{i}" for i in range(n)})
    assert elapsed < 2.0, f"ring of {n} should converge fast; took {elapsed:.2f}s"


# ── Representative member sampling ────────────────────────────────────


def _make_entity(uuid: str, name: str = "", summary: str = "") -> EntityNode:
    """Build a minimal EntityNode for sampling tests."""
    return EntityNode(uuid=uuid, name=name or uuid, group_id="g", summary=summary)


def test_sampling_returns_all_members_when_cluster_smaller_than_k():
    members = [_make_entity(f"e{i}") for i in range(5)]
    sampled = _select_representative_members(members, projection=None, sample_size=10)
    assert sampled == members


def test_sampling_prefers_higher_in_community_degree():
    # hub has 3 in-community neighbors with weight 5 each (total 15)
    # isolated nodes have 0 or 1 connections
    members = [_make_entity(f"e{i}") for i in range(5)]
    projection = _make_projection(
        [
            ("e0", "e1", 5),  # hub e0 connected to e1
            ("e0", "e2", 5),  # hub e0 connected to e2
            ("e0", "e3", 5),  # hub e0 connected to e3
            ("e1", "e4", 1),  # peripheral edge
        ]
    )
    sampled = _select_representative_members(members, projection, sample_size=2)
    assert len(sampled) == 2
    # e0 (degree=15) must be picked first
    assert sampled[0].uuid == "e0"


def test_sampling_falls_back_to_summary_length_without_projection():
    members = [
        _make_entity("short", summary="x"),
        _make_entity("medium", summary="x" * 50),
        _make_entity("long", summary="x" * 200),
    ]
    sampled = _select_representative_members(members, projection=None, sample_size=2)
    # Longest summary wins, then medium
    assert sampled[0].uuid == "long"
    assert sampled[1].uuid == "medium"


def test_sampling_is_deterministic_on_ties():
    # All members have identical degree and summary — ties broken by name desc
    members = [_make_entity(f"e{i}", name=f"e{i}") for i in range(5)]
    projection = _make_projection(
        [
            ("e0", "e1", 1),
            ("e1", "e2", 1),
            ("e2", "e3", 1),
            ("e3", "e4", 1),
            ("e4", "e0", 1),
        ]
    )
    # Every node has identical in-community degree (2) and identical summary ("")
    first = _select_representative_members(members, projection, sample_size=2)
    second = _select_representative_members(members, projection, sample_size=2)
    assert [m.uuid for m in first] == [m.uuid for m in second]


def test_sampling_only_counts_in_community_edges():
    # e0 has many connections outside the community, but only a few inside
    # e1 has fewer total connections but all inside the community
    members = [_make_entity("e0"), _make_entity("e1")]
    projection = _make_projection(
        [
            ("e0", "outsider_a", 100),  # outside the cluster
            ("e0", "outsider_b", 100),
            ("e0", "e1", 1),  # inside: weight 1
            ("e1", "e0", 1),  # same edge, already counted via symmetry in _make_projection
        ]
    )
    # Remove duplicate symmetric insertion bookkeeping by using the projection directly:
    # _make_projection already handles symmetry, so we override:
    projection = {
        "e0": [
            Neighbor(node_uuid="outsider_a", edge_count=100),
            Neighbor(node_uuid="outsider_b", edge_count=100),
            Neighbor(node_uuid="e1", edge_count=1),
        ],
        "e1": [
            Neighbor(node_uuid="e0", edge_count=1),
        ],
    }
    sampled = _select_representative_members(members, projection, sample_size=1)
    # e0 and e1 both have in-community degree 1, tie broken by name desc → "e1"
    assert sampled[0].uuid == "e1"
