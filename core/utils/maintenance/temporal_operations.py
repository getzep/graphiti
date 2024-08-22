import logging
from datetime import datetime
from typing import List

from core.edges import EntityEdge
from core.llm_client import LLMClient
from core.nodes import EntityNode, EpisodicNode
from core.prompts import prompt_library

logger = logging.getLogger(__name__)

NodeEdgeNodeTriplet = tuple[EntityNode, EntityEdge, EntityNode]


def extract_node_and_edge_triplets(
	edges: list[EntityEdge], nodes: list[EntityNode]
) -> list[NodeEdgeNodeTriplet]:
	return [extract_node_edge_node_triplet(edge, nodes) for edge in edges]


def extract_node_edge_node_triplet(
	edge: EntityEdge, nodes: list[EntityNode]
) -> NodeEdgeNodeTriplet:
	source_node = next((node for node in nodes if node.uuid == edge.source_node_uuid), None)
	target_node = next((node for node in nodes if node.uuid == edge.target_node_uuid), None)
	return (source_node, edge, target_node)


def prepare_edges_for_invalidation(
	existing_edges: list[EntityEdge],
	new_edges: list[EntityEdge],
	nodes: list[EntityNode],
) -> tuple[list[NodeEdgeNodeTriplet], list[NodeEdgeNodeTriplet]]:
	existing_edges_pending_invalidation = []  # TODO: this is not yet used?
	new_edges_with_nodes = []  # TODO: this is not yet used?

	existing_edges_pending_invalidation = []
	new_edges_with_nodes = []

	for edge_list, result_list in [
		(existing_edges, existing_edges_pending_invalidation),
		(new_edges, new_edges_with_nodes),
	]:
		for edge in edge_list:
			source_node = next((node for node in nodes if node.uuid == edge.source_node_uuid), None)
			target_node = next((node for node in nodes if node.uuid == edge.target_node_uuid), None)

			if source_node and target_node:
				result_list.append((source_node, edge, target_node))

	return existing_edges_pending_invalidation, new_edges_with_nodes


async def invalidate_edges(
	llm_client: LLMClient,
	existing_edges_pending_invalidation: list[NodeEdgeNodeTriplet],
	new_edges: list[NodeEdgeNodeTriplet],
	current_episode: EpisodicNode,
	previous_episodes: list[EpisodicNode],
) -> list[EntityEdge]:
	invalidated_edges = []  # TODO: this is not yet used?

	context = prepare_invalidation_context(
		existing_edges_pending_invalidation,
		new_edges,
		current_episode,
		previous_episodes,
	)
	logger.info(prompt_library.invalidate_edges.v1(context))
	llm_response = await llm_client.generate_response(prompt_library.invalidate_edges.v1(context))
	logger.info(f'invalidate_edges LLM response: {llm_response}')

	edges_to_invalidate = llm_response.get('invalidated_edges', [])
	invalidated_edges = process_edge_invalidation_llm_response(
		edges_to_invalidate, existing_edges_pending_invalidation
	)

	return invalidated_edges


def prepare_invalidation_context(
	existing_edges: list[NodeEdgeNodeTriplet],
	new_edges: list[NodeEdgeNodeTriplet],
	current_episode: EpisodicNode,
	previous_episodes: list[EpisodicNode],
) -> dict:
	return {
		'existing_edges': [
			f'{edge.uuid} | {source_node.name} - {edge.name} - {target_node.name} (Fact: {edge.fact}) ({edge.created_at.isoformat()})'
			for source_node, edge, target_node in sorted(
				existing_edges, key=lambda x: x[1].created_at, reverse=True
			)
		],
		'new_edges': [
			f'{edge.uuid} | {source_node.name} - {edge.name} - {target_node.name} (Fact: {edge.fact}) ({edge.created_at.isoformat()})'
			for source_node, edge, target_node in sorted(
				new_edges, key=lambda x: x[1].created_at, reverse=True
			)
		],
		'current_episode': current_episode.content,
		'previous_episodes': [episode.content for episode in previous_episodes],
	}


def process_edge_invalidation_llm_response(
	edges_to_invalidate: List[dict], existing_edges: List[NodeEdgeNodeTriplet]
) -> List[EntityEdge]:
	invalidated_edges = []
	for edge_to_invalidate in edges_to_invalidate:
		edge_uuid = edge_to_invalidate['edge_uuid']
		edge_to_update = next(
			(edge for _, edge, _ in existing_edges if edge.uuid == edge_uuid),
			None,
		)
		if edge_to_update:
			edge_to_update.expired_at = datetime.now()
			edge_to_update.fact = edge_to_invalidate['fact']
			invalidated_edges.append(edge_to_update)
			logger.info(
				f"Invalidated edge: {edge_to_update.name} (UUID: {edge_to_update.uuid}). Updated Fact: {edge_to_invalidate['fact']}"
			)
	return invalidated_edges
