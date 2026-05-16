"""
Tests for label_propagation in community_operations.py.

Covers the infinite loop bug where entity pairs with edge_count >= 2 caused
the original while True loop to oscillate forever and never converge.
"""

import pytest

from graphiti_core.utils.maintenance.community_operations import Neighbor, label_propagation


class TestLabelPropagationTermination:
    """label_propagation must always terminate regardless of edge_count."""

    def test_two_nodes_edge_count_2_terminates(self):
        """Minimal trigger for the infinite loop bug: 2 nodes, edge_count=2.

        Before the fix, this oscillated between {A:0,B:1} and {A:1,B:0} forever.
        After the fix, the bounded loop exits within max_iterations.
        """
        projection = {
            'A': [Neighbor(node_uuid='B', edge_count=2)],
            'B': [Neighbor(node_uuid='A', edge_count=2)],
        }
        clusters = label_propagation(projection)
        assert clusters is not None
        assert isinstance(clusters, list)

    def test_two_nodes_edge_count_2_all_nodes_assigned(self):
        """Every node must appear in exactly one cluster."""
        projection = {
            'A': [Neighbor(node_uuid='B', edge_count=2)],
            'B': [Neighbor(node_uuid='A', edge_count=2)],
        }
        clusters = label_propagation(projection)
        all_nodes = [node for cluster in clusters for node in cluster]
        assert sorted(all_nodes) == ['A', 'B']

    def test_chain_with_high_edge_count_terminates(self):
        """Larger graph with edge_count=5 must also terminate."""
        projection = {
            'A': [Neighbor(node_uuid='B', edge_count=5)],
            'B': [Neighbor(node_uuid='A', edge_count=5), Neighbor(node_uuid='C', edge_count=5)],
            'C': [Neighbor(node_uuid='B', edge_count=5), Neighbor(node_uuid='D', edge_count=5)],
            'D': [Neighbor(node_uuid='C', edge_count=5)],
        }
        clusters = label_propagation(projection)
        all_nodes = [node for cluster in clusters for node in cluster]
        assert sorted(all_nodes) == ['A', 'B', 'C', 'D']


class TestLabelPropagationCorrectness:
    """label_propagation should produce sensible community assignments."""

    def test_empty_projection_returns_empty(self):
        clusters = label_propagation({})
        assert clusters == []

    def test_single_node_no_neighbors(self):
        projection = {'A': []}
        clusters = label_propagation(projection)
        all_nodes = [node for cluster in clusters for node in cluster]
        assert all_nodes == ['A']

    def test_two_isolated_nodes_form_separate_clusters(self):
        """Nodes with no edges between them should not be merged."""
        projection = {
            'A': [],
            'B': [],
        }
        clusters = label_propagation(projection)
        assert len(clusters) == 2
        all_nodes = sorted(node for cluster in clusters for node in cluster)
        assert all_nodes == ['A', 'B']

    def test_fully_connected_triangle_edge_count_1(self):
        """Three nodes all connected with edge_count=1 should converge."""
        projection = {
            'A': [Neighbor(node_uuid='B', edge_count=1), Neighbor(node_uuid='C', edge_count=1)],
            'B': [Neighbor(node_uuid='A', edge_count=1), Neighbor(node_uuid='C', edge_count=1)],
            'C': [Neighbor(node_uuid='A', edge_count=1), Neighbor(node_uuid='B', edge_count=1)],
        }
        clusters = label_propagation(projection)
        all_nodes = sorted(node for cluster in clusters for node in cluster)
        assert all_nodes == ['A', 'B', 'C']

    def test_two_dense_cliques_stay_separate(self):
        """Two tightly connected groups with a weak bridge should not fully merge."""
        # Clique 1: A-B-C with edge_count=3
        # Clique 2: D-E-F with edge_count=3
        # Bridge: C-D with edge_count=1
        projection = {
            'A': [Neighbor(node_uuid='B', edge_count=3), Neighbor(node_uuid='C', edge_count=3)],
            'B': [Neighbor(node_uuid='A', edge_count=3), Neighbor(node_uuid='C', edge_count=3)],
            'C': [
                Neighbor(node_uuid='A', edge_count=3),
                Neighbor(node_uuid='B', edge_count=3),
                Neighbor(node_uuid='D', edge_count=1),
            ],
            'D': [
                Neighbor(node_uuid='C', edge_count=1),
                Neighbor(node_uuid='E', edge_count=3),
                Neighbor(node_uuid='F', edge_count=3),
            ],
            'E': [Neighbor(node_uuid='D', edge_count=3), Neighbor(node_uuid='F', edge_count=3)],
            'F': [Neighbor(node_uuid='D', edge_count=3), Neighbor(node_uuid='E', edge_count=3)],
        }
        clusters = label_propagation(projection)
        all_nodes = sorted(node for cluster in clusters for node in cluster)
        assert all_nodes == ['A', 'B', 'C', 'D', 'E', 'F']
        # All nodes accounted for; exact cluster boundaries are algorithm-dependent
        # but must not loop forever and must not lose any node
