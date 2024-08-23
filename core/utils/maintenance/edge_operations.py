import logging
from datetime import datetime
from time import time
from typing import List

from core.edges import EntityEdge, EpisodicEdge
from core.llm_client import LLMClient
from core.nodes import EntityNode, EpisodicNode
from core.prompts import prompt_library

logger = logging.getLogger(__name__)


def build_episodic_edges(
	entity_nodes: List[EntityNode],
	episode: EpisodicNode,
	created_at: datetime,
) -> List[EpisodicEdge]:
	edges: List[EpisodicEdge] = []

	for node in entity_nodes:
		edge = EpisodicEdge(
			source_node_uuid=episode.uuid,
			target_node_uuid=node.uuid,
			created_at=created_at,
		)
		edges.append(edge)

	return edges


async def extract_edges(
	llm_client: LLMClient,
	episode: EpisodicNode,
	nodes: list[EntityNode],
	previous_episodes: list[EpisodicNode],
) -> list[EntityEdge]:
	start = time()

	# Prepare context for LLM
	context = {
		'episode_content': episode.content,
		'episode_timestamp': (episode.valid_at.isoformat() if episode.valid_at else None),
		'nodes': [
			{'uuid': node.uuid, 'name': node.name, 'summary': node.summary} for node in nodes
		],
		'previous_episodes': [
			{
				'content': ep.content,
				'timestamp': ep.valid_at.isoformat() if ep.valid_at else None,
			}
			for ep in previous_episodes
		],
	}

	llm_response = await llm_client.generate_response(prompt_library.extract_edges.v2(context))
	edges_data = llm_response.get('edges', [])

	end = time()
	logger.info(f'Extracted new edges: {edges_data} in {(end - start) * 1000} ms')

	# Convert the extracted data into EntityEdge objects
	edges = []
	for edge_data in edges_data:
		if edge_data['target_node_uuid'] and edge_data['source_node_uuid']:
			edge = EntityEdge(
				source_node_uuid=edge_data['source_node_uuid'],
				target_node_uuid=edge_data['target_node_uuid'],
				name=edge_data['relation_type'],
				fact=edge_data['fact'],
				episodes=[episode.uuid],
				created_at=datetime.now(),
				valid_at=None,
				invalid_at=None,
			)
			edges.append(edge)
			logger.info(
				f'Created new edge: {edge.name} from (UUID: {edge.source_node_uuid}) to (UUID: {edge.target_node_uuid})'
			)

	return edges


def create_edge_identifier(
	source_node: EntityNode, edge: EntityEdge, target_node: EntityNode
) -> str:
	return f'{source_node.name}-{edge.name}-{target_node.name}'


async def dedupe_extracted_edges(
	llm_client: LLMClient,
	extracted_edges: list[EntityEdge],
	existing_edges: list[EntityEdge],
) -> list[EntityEdge]:
	# Create edge map
	edge_map = {}
	for edge in extracted_edges:
		edge_map[edge.uuid] = edge

	# Prepare context for LLM
	context = {
		'extracted_edges': [
			{'uuid': edge.uuid, 'name': edge.name, 'fact': edge.fact} for edge in extracted_edges
		],
		'existing_edges': [
			{'uuid': edge.uuid, 'name': edge.name, 'fact': edge.fact} for edge in existing_edges
		],
	}

	llm_response = await llm_client.generate_response(prompt_library.dedupe_edges.v1(context))
	unique_edge_data = llm_response.get('unique_facts', [])
	logger.info(f'Extracted unique edges: {unique_edge_data}')

	# Get full edge data
	edges = []
	for unique_edge in unique_edge_data:
		edge = edge_map[unique_edge['uuid']]
		edges.append(edge)

	return edges


async def dedupe_edge_list(
	llm_client: LLMClient,
	edges: list[EntityEdge],
) -> list[EntityEdge]:
	start = time()

	# Create edge map
	edge_map = {}
	for edge in edges:
		edge_map[edge.uuid] = edge

	# Prepare context for LLM
	context = {'edges': [{'uuid': edge.uuid, 'fact': edge.fact} for edge in edges]}

	llm_response = await llm_client.generate_response(
		prompt_library.dedupe_edges.edge_list(context)
	)
	unique_edges_data = llm_response.get('unique_facts', [])

	end = time()
	logger.info(f'Extracted edge duplicates: {unique_edges_data} in {(end - start) * 1000} ms ')

	# Get full edge data
	unique_edges = []
	for edge_data in unique_edges_data:
		uuid = edge_data['uuid']
		edge = edge_map[uuid]
		edge.fact = edge_data['fact']
		unique_edges.append(edge)

	return unique_edges
