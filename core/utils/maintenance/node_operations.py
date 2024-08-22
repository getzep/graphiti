import logging
from datetime import datetime
from time import time

from core.llm_client import LLMClient
from core.nodes import EntityNode, EpisodicNode
from core.prompts import prompt_library

logger = logging.getLogger(__name__)


async def extract_new_nodes(
	llm_client: LLMClient,
	episode: EpisodicNode,
	relevant_schema: dict[str, any],
	previous_episodes: list[EpisodicNode],
) -> list[EntityNode]:
	# Prepare context for LLM
	existing_nodes = [
		{'name': node_name, 'label': node_info['label'], 'uuid': node_info['uuid']}
		for node_name, node_info in relevant_schema['nodes'].items()
	]

	context = {
		'episode_content': episode.content,
		'episode_timestamp': (episode.valid_at.isoformat() if episode.valid_at else None),
		'existing_nodes': existing_nodes,
		'previous_episodes': [
			{
				'content': ep.content,
				'timestamp': ep.valid_at.isoformat() if ep.valid_at else None,
			}
			for ep in previous_episodes
		],
	}

	llm_response = await llm_client.generate_response(prompt_library.extract_nodes.v1(context))
	new_nodes_data = llm_response.get('new_nodes', [])
	logger.info(f'Extracted new nodes: {new_nodes_data}')
	# Convert the extracted data into EntityNode objects
	new_nodes = []
	for node_data in new_nodes_data:
		# Check if the node already exists
		if not any(existing_node['name'] == node_data['name'] for existing_node in existing_nodes):
			new_node = EntityNode(
				name=node_data['name'],
				labels=node_data['labels'],
				summary=node_data['summary'],
				created_at=datetime.now(),
			)
			new_nodes.append(new_node)
			logger.info(f'Created new node: {new_node.name} (UUID: {new_node.uuid})')
		else:
			logger.info(f"Node {node_data['name']} already exists, skipping creation.")

	return new_nodes


async def extract_nodes(
	llm_client: LLMClient,
	episode: EpisodicNode,
	previous_episodes: list[EpisodicNode],
) -> list[EntityNode]:
	start = time()

	# Prepare context for LLM
	context = {
		'episode_content': episode.content,
		'episode_timestamp': (episode.valid_at.isoformat() if episode.valid_at else None),
		'previous_episodes': [
			{
				'content': ep.content,
				'timestamp': ep.valid_at.isoformat() if ep.valid_at else None,
			}
			for ep in previous_episodes
		],
	}

	llm_response = await llm_client.generate_response(prompt_library.extract_nodes.v3(context))
	new_nodes_data = llm_response.get('new_nodes', [])

	end = time()
	logger.info(f'Extracted new nodes: {new_nodes_data} in {(end - start) * 1000} ms')
	# Convert the extracted data into EntityNode objects
	new_nodes = []
	for node_data in new_nodes_data:
		new_node = EntityNode(
			name=node_data['name'],
			labels=node_data['labels'],
			summary=node_data['summary'],
			created_at=datetime.now(),
		)
		new_nodes.append(new_node)
		logger.info(f'Created new node: {new_node.name} (UUID: {new_node.uuid})')

	return new_nodes


async def dedupe_extracted_nodes(
	llm_client: LLMClient,
	extracted_nodes: list[EntityNode],
	existing_nodes: list[EntityNode],
) -> tuple[list[EntityNode], dict[str, str]]:
	start = time()

	# build existing node map
	node_map = {}
	for node in existing_nodes:
		node_map[node.name] = node

	# Prepare context for LLM
	existing_nodes_context = [
		{'name': node.name, 'summary': node.summary} for node in existing_nodes
	]

	extracted_nodes_context = [
		{'name': node.name, 'summary': node.summary} for node in extracted_nodes
	]

	context = {
		'existing_nodes': existing_nodes_context,
		'extracted_nodes': extracted_nodes_context,
	}

	llm_response = await llm_client.generate_response(prompt_library.dedupe_nodes.v2(context))

	duplicate_data = llm_response.get('duplicates', [])

	end = time()
	logger.info(f'Deduplicated nodes: {duplicate_data} in {(end - start) * 1000} ms')

	uuid_map = {}
	for duplicate in duplicate_data:
		uuid = node_map[duplicate['name']].uuid
		uuid_value = node_map[duplicate['duplicate_of']].uuid
		uuid_map[uuid] = uuid_value

	nodes = []
	for node in extracted_nodes:
		if node.uuid in uuid_map:
			existing_name = uuid_map[node.name]
			existing_node = node_map[existing_name]
			nodes.append(existing_node)
			continue
		nodes.append(node)

	return nodes, uuid_map


async def dedupe_node_list(
	llm_client: LLMClient,
	nodes: list[EntityNode],
) -> tuple[list[EntityNode], dict[str, str]]:
	start = time()

	# build node map
	node_map = {}
	for node in nodes:
		node_map[node.name] = node

	# Prepare context for LLM
	nodes_context = [{'name': node.name, 'summary': node.summary} for node in nodes]

	context = {
		'nodes': nodes_context,
	}

	llm_response = await llm_client.generate_response(
		prompt_library.dedupe_nodes.node_list(context)
	)

	nodes_data = llm_response.get('nodes', [])

	end = time()
	logger.info(f'Deduplicated nodes: {nodes_data} in {(end - start) * 1000} ms')

	# Get full node data
	unique_nodes = []
	uuid_map: dict[str, str] = {}
	for node_data in nodes_data:
		node = node_map[node_data['names'][0]]
		unique_nodes.append(node)

		for name in node_data['names'][1:]:
			uuid = node_map[name].uuid
			uuid_value = node_map[node_data['names'][0]].uuid
			uuid_map[uuid] = uuid_value

	return unique_nodes, uuid_map
