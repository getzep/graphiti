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

import logging
from datetime import datetime
from time import time
from typing import Any

from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.prompts import prompt_library

logger = logging.getLogger(__name__)


async def extract_message_nodes(
    llm_client: LLMClient, episode: EpisodicNode, previous_episodes: list[EpisodicNode]
) -> list[dict[str, Any]]:
    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'episode_timestamp': episode.valid_at.isoformat(),
        'previous_episodes': [
            {
                'content': ep.content,
                'timestamp': ep.valid_at.isoformat(),
            }
            for ep in previous_episodes
        ],
    }

    llm_response = await llm_client.generate_response(prompt_library.extract_nodes.v2(context))
    extracted_node_data = llm_response.get('extracted_nodes', [])
    return extracted_node_data


async def extract_json_nodes(
    llm_client: LLMClient,
    episode: EpisodicNode,
) -> list[dict[str, Any]]:
    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'episode_timestamp': episode.valid_at.isoformat(),
        'source_description': episode.source_description,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.extract_json(context)
    )
    extracted_node_data = llm_response.get('extracted_nodes', [])
    return extracted_node_data


async def extract_nodes(
    llm_client: LLMClient,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
) -> list[EntityNode]:
    start = time()
    extracted_node_data: list[dict[str, Any]] = []
    if episode.source in [EpisodeType.message, EpisodeType.text]:
        extracted_node_data = await extract_message_nodes(llm_client, episode, previous_episodes)
    elif episode.source == EpisodeType.json:
        extracted_node_data = await extract_json_nodes(llm_client, episode)

    end = time()
    logger.info(f'Extracted new nodes: {extracted_node_data} in {(end - start) * 1000} ms')
    # Convert the extracted data into EntityNode objects
    new_nodes = []
    for node_data in extracted_node_data:
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
) -> tuple[list[EntityNode], dict[str, str], list[EntityNode]]:
    start = time()

    # build existing node map
    node_map: dict[str, EntityNode] = {}
    for node in existing_nodes:
        node_map[node.name] = node

    # Temp hack
    new_nodes_map: dict[str, EntityNode] = {}
    for node in extracted_nodes:
        new_nodes_map[node.name] = node

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

    uuid_map: dict[str, str] = {}
    for duplicate in duplicate_data:
        uuid = new_nodes_map[duplicate['name']].uuid
        uuid_value = node_map[duplicate['duplicate_of']].uuid
        uuid_map[uuid] = uuid_value

    nodes: list[EntityNode] = []
    brand_new_nodes: list[EntityNode] = []
    for node in extracted_nodes:
        if node.uuid in uuid_map:
            existing_uuid = uuid_map[node.uuid]
            # TODO(Preston): This is a bit of a hack I implemented because we were getting incorrect uuids for existing nodes,
            # can you revisit the node dedup function and make it somewhat cleaner and add more comments/tests please?
            # find an existing node by the uuid from the nodes_map (each key is name, so we need to iterate by uuid value)
            existing_node = next((v for k, v in node_map.items() if v.uuid == existing_uuid), None)
            if existing_node:
                nodes.append(existing_node)

            continue
        brand_new_nodes.append(node)
        nodes.append(node)

    return nodes, uuid_map, brand_new_nodes


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
