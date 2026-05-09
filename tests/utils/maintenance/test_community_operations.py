"""Tests for community summary member sampling.

The `sample_size` parameter on `build_community` (and `build_communities`)
limits the number of members whose summaries feed the binary-merge
summarization tree. This bounds LLM cost on large graphs:

- Without sampling, summary cost grows as O(total_nodes) — every entity's
  summary participates in the merge tree.
- With sampling, cost grows as O(num_communities * sample_size) — only the
  top-K most representative members per community participate.

These tests focus on the `_select_representative_members` helper that
implements the ranking. End-to-end tests of `build_communities` with a
real LLM are out of scope here — see the existing integration tests.
"""

from __future__ import annotations

from graphiti_core.nodes import EntityNode
from graphiti_core.utils.maintenance.community_operations import (
    Neighbor,
    _select_representative_members,
)


def _make_entity(uuid: str, name: str = '', summary: str = '') -> EntityNode:
    """Build a minimal EntityNode for sampling tests."""
    return EntityNode(uuid=uuid, name=name or uuid, group_id='g', summary=summary)


def test_returns_all_members_when_cluster_smaller_than_k():
    members = [_make_entity(f'e{i}') for i in range(5)]
    sampled = _select_representative_members(members, projection=None, sample_size=10)
    assert sampled == members


def test_returns_all_members_when_cluster_equal_to_k():
    members = [_make_entity(f'e{i}') for i in range(5)]
    sampled = _select_representative_members(members, projection=None, sample_size=5)
    assert sampled == members


def test_prefers_higher_in_community_degree():
    """A node with many in-community neighbors outranks isolated nodes."""
    # e0 is a hub: 3 weighted edges within the community.
    # e1 has 1 weighted edge.
    # e2..e4 have no in-community edges in this projection.
    members = [_make_entity(f'e{i}') for i in range(5)]
    projection: dict[str, list[Neighbor]] = {
        'e0': [
            Neighbor(node_uuid='e1', edge_count=5),
            Neighbor(node_uuid='e2', edge_count=5),
            Neighbor(node_uuid='e3', edge_count=5),
        ],
        'e1': [Neighbor(node_uuid='e0', edge_count=5)],
        'e2': [Neighbor(node_uuid='e0', edge_count=5)],
        'e3': [Neighbor(node_uuid='e0', edge_count=5)],
        'e4': [],
    }
    sampled = _select_representative_members(members, projection, sample_size=2)
    assert len(sampled) == 2
    # Hub must be picked first
    assert sampled[0].uuid == 'e0'


def test_falls_back_to_summary_length_without_projection():
    """When no projection is available, longer summaries win."""
    members = [
        _make_entity('short', summary='x'),
        _make_entity('medium', summary='x' * 50),
        _make_entity('long', summary='x' * 200),
    ]
    sampled = _select_representative_members(members, projection=None, sample_size=2)
    assert sampled[0].uuid == 'long'
    assert sampled[1].uuid == 'medium'


def test_falls_back_to_summary_length_with_empty_projection():
    """An empty projection (e.g., from a graph_operations_interface that
    does not expose projections) is treated like no projection at all."""
    members = [
        _make_entity('a', summary='short'),
        _make_entity('b', summary='x' * 100),
    ]
    sampled = _select_representative_members(members, projection={}, sample_size=1)
    assert sampled[0].uuid == 'b'


def test_deterministic_on_ties():
    """Same input produces the same partition across runs."""
    members = [_make_entity(f'e{i}') for i in range(5)]
    projection: dict[str, list[Neighbor]] = {
        'e0': [Neighbor(node_uuid='e1', edge_count=1)],
        'e1': [
            Neighbor(node_uuid='e0', edge_count=1),
            Neighbor(node_uuid='e2', edge_count=1),
        ],
        'e2': [
            Neighbor(node_uuid='e1', edge_count=1),
            Neighbor(node_uuid='e3', edge_count=1),
        ],
        'e3': [
            Neighbor(node_uuid='e2', edge_count=1),
            Neighbor(node_uuid='e4', edge_count=1),
        ],
        'e4': [Neighbor(node_uuid='e3', edge_count=1)],
    }
    first = _select_representative_members(members, projection, sample_size=2)
    second = _select_representative_members(members, projection, sample_size=2)
    assert [m.uuid for m in first] == [m.uuid for m in second]


def test_only_counts_in_community_edges():
    """Edges to entities outside the community must be ignored.

    A node with many out-of-community connections but only a few in-community
    edges should not outrank an in-community-focused node.
    """
    members = [_make_entity('insider'), _make_entity('insider2')]
    projection: dict[str, list[Neighbor]] = {
        'insider': [
            # Many heavy edges to entities NOT in the cluster
            Neighbor(node_uuid='outsider_a', edge_count=100),
            Neighbor(node_uuid='outsider_b', edge_count=100),
            # One light edge inside
            Neighbor(node_uuid='insider2', edge_count=1),
        ],
        'insider2': [
            Neighbor(node_uuid='insider', edge_count=1),
        ],
    }
    sampled = _select_representative_members(members, projection, sample_size=1)
    # Both have in-community degree 1; tie-broken by name desc → 'insider2' wins
    assert sampled[0].uuid == 'insider2'


def test_summary_length_breaks_degree_ties():
    """When two nodes have the same in-community degree, the one with the
    richer summary wins (since richer summaries contribute more to the
    binary merge)."""
    members = [
        _make_entity('a', summary='x' * 10),
        _make_entity('b', summary='x' * 200),
    ]
    projection: dict[str, list[Neighbor]] = {
        'a': [Neighbor(node_uuid='b', edge_count=1)],
        'b': [Neighbor(node_uuid='a', edge_count=1)],
    }
    sampled = _select_representative_members(members, projection, sample_size=1)
    assert sampled[0].uuid == 'b'
