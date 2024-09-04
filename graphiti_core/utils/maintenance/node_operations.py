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

import asyncio
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
) -> tuple[list[EntityNode], dict[str, str]]:
    start = time()

    # build existing node map
    node_map: dict[str, EntityNode] = {}
    for node in existing_nodes:
        node_map[node.uuid] = node

    # Prepare context for LLM
    existing_nodes_context = [
        {'uuid': node.uuid, 'name': node.name, 'summary': node.summary} for node in existing_nodes
    ]

    extracted_nodes_context = [
        {'uuid': node.uuid, 'name': node.name, 'summary': node.summary} for node in extracted_nodes
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
        uuid_value = duplicate['duplicate_of']
        uuid_map[duplicate['uuid']] = uuid_value

    nodes: list[EntityNode] = []
    for node in extracted_nodes:
        if node.uuid in uuid_map:
            existing_uuid = uuid_map[node.uuid]
            existing_node = node_map[existing_uuid]
            nodes.append(existing_node)
        else:
            nodes.append(node)

    return nodes, uuid_map


async def resolve_extracted_nodes(
    llm_client: LLMClient,
    extracted_nodes: list[EntityNode],
    existing_nodes_lists: list[list[EntityNode]],
) -> tuple[list[EntityNode], dict[str, str]]:
    uuid_map: dict[str, str] = {}
    resolved_nodes: list[EntityNode] = []
    results: list[tuple[EntityNode, dict[str, str]]] = list(
        await asyncio.gather(
            *[
                resolve_extracted_node(llm_client, extracted_node, existing_nodes)
                for extracted_node, existing_nodes in zip(extracted_nodes, existing_nodes_lists)
            ]
        )
    )

    for result in results:
        uuid_map.update(result[1])
        resolved_nodes.append(result[0])

    return resolved_nodes, uuid_map


async def resolve_extracted_node(
    llm_client: LLMClient, extracted_node: EntityNode, existing_nodes: list[EntityNode]
) -> tuple[EntityNode, dict[str, str]]:
    start = time()

    # Prepare context for LLM
    existing_nodes_context = [
        {'uuid': node.uuid, 'name': node.name, 'summary': node.summary} for node in existing_nodes
    ]

    extracted_node_context = {
        'uuid': extracted_node.uuid,
        'name': extracted_node.name,
        'summary': extracted_node.summary,
    }

    context = {
        'existing_nodes': existing_nodes_context,
        'extracted_nodes': extracted_node_context,
    }

    llm_response = await llm_client.generate_response(prompt_library.dedupe_nodes.v3(context))

    is_duplicate: bool = llm_response.get('is_duplicate', False)
    uuid: str | None = llm_response.get('uuid', None)
    summary = llm_response.get('summary', '')

    node = extracted_node
    uuid_map: dict[str, str] = {}
    if is_duplicate:
        for existing_node in existing_nodes:
            if existing_node.uuid != uuid:
                continue
            node = existing_node
            node.summary = summary
            uuid_map[extracted_node.uuid] = existing_node.uuid

    end = time()
    logger.info(
        f'Resolved node: {extracted_node.name} is {node.name}, in {(end - start) * 1000} ms'
    )

    return node, uuid_map


async def dedupe_node_list(
    llm_client: LLMClient,
    nodes: list[EntityNode],
) -> tuple[list[EntityNode], dict[str, str]]:
    start = time()

    # build node map
    node_map = {}
    for node in nodes:
        node_map[node.uuid] = node

    # Prepare context for LLM
    nodes_context = [
        {'uuid': node.uuid, 'name': node.name, 'summary': node.summary} for node in nodes
    ]

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
        node = node_map[node_data['uuids'][0]]
        node.summary = node_data['summary']
        unique_nodes.append(node)

        for uuid in node_data['uuids'][1:]:
            uuid_value = node_map[node_data['uuids'][0]].uuid
            uuid_map[uuid] = uuid_value

    return unique_nodes, uuid_map
