"""Unit tests for flattening gathered update_community results (#836)."""

from datetime import UTC, datetime

import pytest

from graphiti_core.edges import CommunityEdge
from graphiti_core.graphiti import _flatten_community_results
from graphiti_core.nodes import CommunityNode


def utc_now() -> datetime:
    return datetime.now(UTC)


@pytest.fixture
def community_nodes():
    return [
        CommunityNode(
            uuid=f'community-{i}',
            name=f'Community {i}',
            group_id='test',
            summary=f'Summary {i}',
        )
        for i in range(3)
    ]


@pytest.fixture
def community_edges(community_nodes):
    return [
        CommunityEdge(
            uuid=f'edge-{i}',
            source_node_uuid=node.uuid,
            target_node_uuid=community_nodes[(i + 1) % 3].uuid,
            group_id='test',
            created_at=utc_now(),
        )
        for i, node in enumerate(community_nodes)
    ]


def test_flatten_handles_empty_results():
    communities, edges = _flatten_community_results([])

    assert communities == []
    assert edges == []


def test_flatten_merges_per_node_pairs(community_nodes, community_edges):
    # One pair per affected node, as returned by each update_community call.
    results = [
        (community_nodes[:2], community_edges[:1]),
        (community_nodes[2:], community_edges[1:]),
    ]

    communities, edges = _flatten_community_results(results)

    assert communities == community_nodes
    assert edges == community_edges


def test_flatten_preserves_order_within_pairs(community_nodes, community_edges):
    single = [(community_nodes, community_edges)]

    communities, edges = _flatten_community_results(single)

    assert communities == community_nodes
    assert edges == community_edges
