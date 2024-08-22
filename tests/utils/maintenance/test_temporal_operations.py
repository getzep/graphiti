from datetime import datetime, timedelta

import pytest

from core.edges import EntityEdge
from core.nodes import EntityNode
from core.utils.maintenance.temporal_operations import (
	prepare_edges_for_invalidation,
	prepare_invalidation_context,
)


# Helper function to create test data
def create_test_data():
	now = datetime.now()

	# Create nodes
	node1 = EntityNode(uuid='1', name='Node1', labels=['Person'], created_at=now)
	node2 = EntityNode(uuid='2', name='Node2', labels=['Person'], created_at=now)
	node3 = EntityNode(uuid='3', name='Node3', labels=['Person'], created_at=now)

	# Create edges
	existing_edge1 = EntityEdge(
		uuid='e1',
		source_node_uuid='1',
		target_node_uuid='2',
		name='KNOWS',
		fact='Node1 knows Node2',
		created_at=now,
	)
	existing_edge2 = EntityEdge(
		uuid='e2',
		source_node_uuid='2',
		target_node_uuid='3',
		name='LIKES',
		fact='Node2 likes Node3',
		created_at=now,
	)
	new_edge1 = EntityEdge(
		uuid='e3',
		source_node_uuid='1',
		target_node_uuid='3',
		name='WORKS_WITH',
		fact='Node1 works with Node3',
		created_at=now,
	)
	new_edge2 = EntityEdge(
		uuid='e4',
		source_node_uuid='1',
		target_node_uuid='2',
		name='DISLIKES',
		fact='Node1 dislikes Node2',
		created_at=now,
	)

	return {
		'nodes': [node1, node2, node3],
		'existing_edges': [existing_edge1, existing_edge2],
		'new_edges': [new_edge1, new_edge2],
	}


def test_prepare_edges_for_invalidation_basic():
	test_data = create_test_data()

	existing_edges_pending_invalidation, new_edges_with_nodes = prepare_edges_for_invalidation(
		test_data['existing_edges'], test_data['new_edges'], test_data['nodes']
	)

	assert len(existing_edges_pending_invalidation) == 2
	assert len(new_edges_with_nodes) == 2

	# Check if the edges are correctly associated with nodes
	for edge_with_nodes in existing_edges_pending_invalidation + new_edges_with_nodes:
		assert isinstance(edge_with_nodes[0], EntityNode)
		assert isinstance(edge_with_nodes[1], EntityEdge)
		assert isinstance(edge_with_nodes[2], EntityNode)


def test_prepare_edges_for_invalidation_no_existing_edges():
	test_data = create_test_data()

	existing_edges_pending_invalidation, new_edges_with_nodes = prepare_edges_for_invalidation(
		[], test_data['new_edges'], test_data['nodes']
	)

	assert len(existing_edges_pending_invalidation) == 0
	assert len(new_edges_with_nodes) == 2


def test_prepare_edges_for_invalidation_no_new_edges():
	test_data = create_test_data()

	existing_edges_pending_invalidation, new_edges_with_nodes = prepare_edges_for_invalidation(
		test_data['existing_edges'], [], test_data['nodes']
	)

	assert len(existing_edges_pending_invalidation) == 2
	assert len(new_edges_with_nodes) == 0


def test_prepare_edges_for_invalidation_missing_nodes():
	test_data = create_test_data()

	# Remove one node to simulate a missing node scenario
	nodes = test_data['nodes'][:-1]

	existing_edges_pending_invalidation, new_edges_with_nodes = prepare_edges_for_invalidation(
		test_data['existing_edges'], test_data['new_edges'], nodes
	)

	assert len(existing_edges_pending_invalidation) == 1
	assert len(new_edges_with_nodes) == 1


def test_prepare_invalidation_context():
	# Create test data
	now = datetime.now()

	# Create nodes
	node1 = EntityNode(uuid='1', name='Node1', labels=['Person'], created_at=now)
	node2 = EntityNode(uuid='2', name='Node2', labels=['Person'], created_at=now)
	node3 = EntityNode(uuid='3', name='Node3', labels=['Person'], created_at=now)

	# Create edges
	edge1 = EntityEdge(
		uuid='e1',
		source_node_uuid='1',
		target_node_uuid='2',
		name='KNOWS',
		fact='Node1 knows Node2',
		created_at=now,
	)
	edge2 = EntityEdge(
		uuid='e2',
		source_node_uuid='2',
		target_node_uuid='3',
		name='LIKES',
		fact='Node2 likes Node3',
		created_at=now,
	)

	# Create NodeEdgeNodeTriplet objects
	existing_edge = (node1, edge1, node2)
	new_edge = (node2, edge2, node3)

	# Prepare test input
	existing_edges = [existing_edge]
	new_edges = [new_edge]

	# Call the function
	result = prepare_invalidation_context(existing_edges, new_edges)

	# Assert the result
	assert isinstance(result, dict)
	assert 'existing_edges' in result
	assert 'new_edges' in result
	assert len(result['existing_edges']) == 1
	assert len(result['new_edges']) == 1

	# Check the format of the existing edge
	existing_edge_str = result['existing_edges'][0]
	assert edge1.uuid in existing_edge_str
	assert node1.name in existing_edge_str
	assert edge1.name in existing_edge_str
	assert node2.name in existing_edge_str
	assert edge1.created_at.isoformat() in existing_edge_str

	# Check the format of the new edge
	new_edge_str = result['new_edges'][0]
	assert edge2.uuid in new_edge_str
	assert node2.name in new_edge_str
	assert edge2.name in new_edge_str
	assert node3.name in new_edge_str
	assert edge2.created_at.isoformat() in new_edge_str


def test_prepare_invalidation_context_empty_input():
	result = prepare_invalidation_context([], [])
	assert isinstance(result, dict)
	assert 'existing_edges' in result
	assert 'new_edges' in result
	assert len(result['existing_edges']) == 0
	assert len(result['new_edges']) == 0


def test_prepare_invalidation_context_sorting():
	now = datetime.now()

	# Create nodes
	node1 = EntityNode(uuid='1', name='Node1', labels=['Person'], created_at=now)
	node2 = EntityNode(uuid='2', name='Node2', labels=['Person'], created_at=now)

	# Create edges with different timestamps
	edge1 = EntityEdge(
		uuid='e1',
		source_node_uuid='1',
		target_node_uuid='2',
		name='KNOWS',
		fact='Node1 knows Node2',
		created_at=now,
	)
	edge2 = EntityEdge(
		uuid='e2',
		source_node_uuid='2',
		target_node_uuid='1',
		name='LIKES',
		fact='Node2 likes Node1',
		created_at=now + timedelta(hours=1),
	)

	edge_with_nodes1 = (node1, edge1, node2)
	edge_with_nodes2 = (node2, edge2, node1)

	# Prepare test input
	existing_edges = [edge_with_nodes1, edge_with_nodes2]

	# Call the function
	result = prepare_invalidation_context(existing_edges, [])

	# Assert the result
	assert len(result['existing_edges']) == 2
	assert edge2.uuid in result['existing_edges'][0]  # The newer edge should be first
	assert edge1.uuid in result['existing_edges'][1]  # The older edge should be second


# Run the tests
if __name__ == '__main__':
	pytest.main([__file__])
