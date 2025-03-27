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
from contextlib import suppress
from time import time

import pydantic
from pydantic import BaseModel

from graphiti_core.helpers import MAX_REFLEXION_ITERATIONS, semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_nodes import NodeDuplicate
from graphiti_core.prompts.extract_nodes import (
    EntityClassification,
    ExtractedNodes,
    MissedEntities,
)
from graphiti_core.prompts.summarize_nodes import Summary
from graphiti_core.utils.datetime_utils import utc_now

logger = logging.getLogger(__name__)


async def extract_message_nodes(
    llm_client: LLMClient,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    custom_prompt='',
) -> list[str]:
    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'episode_timestamp': episode.valid_at.isoformat(),
        'previous_episodes': [ep.content for ep in previous_episodes],
        'custom_prompt': custom_prompt,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.extract_message(context), response_model=ExtractedNodes
    )
    extracted_node_names = llm_response.get('extracted_node_names', [])
    return extracted_node_names


async def extract_text_nodes(
    llm_client: LLMClient,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    custom_prompt='',
) -> list[str]:
    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'episode_timestamp': episode.valid_at.isoformat(),
        'previous_episodes': [ep.content for ep in previous_episodes],
        'custom_prompt': custom_prompt,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.extract_text(context), ExtractedNodes
    )
    extracted_node_names = llm_response.get('extracted_node_names', [])
    return extracted_node_names


async def extract_json_nodes(
    llm_client: LLMClient, episode: EpisodicNode, custom_prompt=''
) -> list[str]:
    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'episode_timestamp': episode.valid_at.isoformat(),
        'source_description': episode.source_description,
        'custom_prompt': custom_prompt,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.extract_json(context), ExtractedNodes
    )
    extracted_node_names = llm_response.get('extracted_node_names', [])
    return extracted_node_names


async def extract_nodes_reflexion(
    llm_client: LLMClient,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    node_names: list[str],
) -> list[str]:
    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'previous_episodes': [ep.content for ep in previous_episodes],
        'extracted_entities': node_names,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.reflexion(context), MissedEntities
    )
    missed_entities = llm_response.get('missed_entities', [])

    return missed_entities


async def extract_nodes(
    llm_client: LLMClient,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    entity_types: dict[str, BaseModel] | None = None,
) -> list[EntityNode]:
    start = time()
    extracted_node_names: list[str] = []
    custom_prompt = ''
    entities_missed = True
    reflexion_iterations = 0
    while entities_missed and reflexion_iterations < MAX_REFLEXION_ITERATIONS:
        if episode.source == EpisodeType.message:
            extracted_node_names = await extract_message_nodes(
                llm_client, episode, previous_episodes, custom_prompt
            )
        elif episode.source == EpisodeType.text:
            extracted_node_names = await extract_text_nodes(
                llm_client, episode, previous_episodes, custom_prompt
            )
        elif episode.source == EpisodeType.json:
            extracted_node_names = await extract_json_nodes(llm_client, episode, custom_prompt)

        reflexion_iterations += 1
        if reflexion_iterations < MAX_REFLEXION_ITERATIONS:
            missing_entities = await extract_nodes_reflexion(
                llm_client, episode, previous_episodes, extracted_node_names
            )

            entities_missed = len(missing_entities) != 0

            custom_prompt = 'The following entities were missed in a previous extraction: '
            for entity in missing_entities:
                custom_prompt += f'\n{entity},'

    node_classification_context = {
        'episode_content': episode.content,
        'previous_episodes': [ep.content for ep in previous_episodes],
        'extracted_entities': extracted_node_names,
        'entity_types': {
            type_name: values.model_json_schema().get('description')
            for type_name, values in entity_types.items()
        }
        if entity_types is not None
        else {},
    }

    node_classifications: dict[str, str | None] = {}

    if entity_types is not None:
        try:
            llm_response = await llm_client.generate_response(
                prompt_library.extract_nodes.classify_nodes(node_classification_context),
                response_model=EntityClassification,
            )
            entity_classifications = llm_response.get('entity_classifications', [])
            node_classifications.update(
                {
                    entity_classification.get('name'): entity_classification.get('entity_type')
                    for entity_classification in entity_classifications
                }
            )
        # catch classification errors and continue if we can't classify
        except Exception as e:
            logger.exception(e)

    end = time()
    logger.debug(f'Extracted new nodes: {extracted_node_names} in {(end - start) * 1000} ms')
    # Convert the extracted data into EntityNode objects
    new_nodes = []
    for name in extracted_node_names:
        entity_type = node_classifications.get(name)
        if entity_types is not None and entity_type not in entity_types:
            entity_type = None

        labels = (
            ['Entity']
            if entity_type is None or entity_type == 'None' or entity_type == 'null'
            else ['Entity', entity_type]
        )

        new_node = EntityNode(
            name=name,
            group_id=episode.group_id,
            labels=labels,
            summary='',
            created_at=utc_now(),
        )
        new_nodes.append(new_node)
        logger.debug(f'Created new node: {new_node.name} (UUID: {new_node.uuid})')

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

    llm_response = await llm_client.generate_response(prompt_library.dedupe_nodes.node(context))

    duplicate_data = llm_response.get('duplicates', [])

    end = time()
    logger.debug(f'Deduplicated nodes: {duplicate_data} in {(end - start) * 1000} ms')

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
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_types: dict[str, BaseModel] | None = None,
) -> tuple[list[EntityNode], dict[str, str]]:
    uuid_map: dict[str, str] = {}
    resolved_nodes: list[EntityNode] = []
    results: list[tuple[EntityNode, dict[str, str]]] = list(
        await semaphore_gather(
            *[
                resolve_extracted_node(
                    llm_client,
                    extracted_node,
                    existing_nodes,
                    episode,
                    previous_episodes,
                    entity_types,
                )
                for extracted_node, existing_nodes in zip(extracted_nodes, existing_nodes_lists)
            ]
        )
    )

    for result in results:
        uuid_map.update(result[1])
        resolved_nodes.append(result[0])

    return resolved_nodes, uuid_map


async def resolve_extracted_node(
    llm_client: LLMClient,
    extracted_node: EntityNode,
    existing_nodes: list[EntityNode],
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_types: dict[str, BaseModel] | None = None,
) -> tuple[EntityNode, dict[str, str]]:
    start = time()

    # Prepare context for LLM
    existing_nodes_context = [
        {'uuid': node.uuid, 'name': node.name, 'attributes': node.attributes}
        for node in existing_nodes
    ]

    extracted_node_context = {
        'uuid': extracted_node.uuid,
        'name': extracted_node.name,
        'summary': extracted_node.summary,
    }

    context = {
        'existing_nodes': existing_nodes_context,
        'extracted_nodes': extracted_node_context,
        'episode_content': episode.content if episode is not None else '',
        'previous_episodes': [ep.content for ep in previous_episodes]
        if previous_episodes is not None
        else [],
    }

    summary_context = {
        'node_name': extracted_node.name,
        'node_summary': extracted_node.summary,
        'episode_content': episode.content if episode is not None else '',
        'previous_episodes': [ep.content for ep in previous_episodes]
        if previous_episodes is not None
        else [],
        'attributes': [],
    }

    entity_type_classes: tuple[BaseModel, ...] = tuple()
    if entity_types is not None:  # type: ignore
        entity_type_classes = entity_type_classes + tuple(
            filter(
                lambda x: x is not None,  # type: ignore
                [entity_types.get(entity_type) for entity_type in extracted_node.labels],  # type: ignore
            )
        )

    for entity_type in entity_type_classes:
        for field_name in entity_type.model_fields:
            summary_context.get('attributes', []).append(field_name)  # type: ignore

    entity_attributes_model = pydantic.create_model(  # type: ignore
        'EntityAttributes',
        __base__=entity_type_classes + (Summary,),  # type: ignore
    )

    llm_response, node_attributes_response = await semaphore_gather(
        llm_client.generate_response(
            prompt_library.dedupe_nodes.node(context), response_model=NodeDuplicate
        ),
        llm_client.generate_response(
            prompt_library.summarize_nodes.summarize_context(summary_context),
            response_model=entity_attributes_model,
        ),
    )

    extracted_node.summary = node_attributes_response.get('summary', '')
    node_attributes = {
        key: value if (value != 'None' or key == 'summary') else None
        for key, value in node_attributes_response.items()
    }

    with suppress(KeyError):
        del node_attributes['summary']

    extracted_node.attributes.update(node_attributes)

    is_duplicate: bool = llm_response.get('is_duplicate', False)
    uuid: str | None = llm_response.get('uuid', None)
    name = llm_response.get('name', '')

    node = extracted_node
    uuid_map: dict[str, str] = {}
    if is_duplicate:
        for existing_node in existing_nodes:
            if existing_node.uuid != uuid:
                continue
            summary_response = await llm_client.generate_response(
                prompt_library.summarize_nodes.summarize_pair(
                    {'node_summaries': [extracted_node.summary, existing_node.summary]}
                ),
                response_model=Summary,
            )
            node = existing_node
            node.name = name
            node.summary = summary_response.get('summary', '')

            new_attributes = extracted_node.attributes
            existing_attributes = existing_node.attributes
            for attribute_name, attribute_value in existing_attributes.items():
                if new_attributes.get(attribute_name) is None:
                    new_attributes[attribute_name] = attribute_value
            node.attributes = new_attributes

            uuid_map[extracted_node.uuid] = existing_node.uuid

    end = time()
    logger.debug(
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
    logger.debug(f'Deduplicated nodes: {nodes_data} in {(end - start) * 1000} ms')

    # Get full node data
    unique_nodes = []
    uuid_map: dict[str, str] = {}
    for node_data in nodes_data:
        node_instance: EntityNode | None = node_map.get(node_data['uuids'][0])
        if node_instance is None:
            logger.warning(f'Node {node_data["uuids"][0]} not found in node map')
            continue
        node_instance.summary = node_data['summary']
        unique_nodes.append(node_instance)

        for uuid in node_data['uuids'][1:]:
            uuid_value = node_map[node_data['uuids'][0]].uuid
            uuid_map[uuid] = uuid_value

    return unique_nodes, uuid_map
