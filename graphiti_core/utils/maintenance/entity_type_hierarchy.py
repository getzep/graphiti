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
from typing import Any

from pydantic import BaseModel

from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.classify_entity_subtype import EntitySubtypeClassification

logger = logging.getLogger(__name__)

PARENT_ENTITY_TYPE_ATTR = '__parent_entity_type__'
MAX_ENTITY_TYPE_SUBTYPE_DEPTH = 4


def parent_entity_type(model: type[BaseModel] | None) -> str | None:
    if model is None:
        return None
    return getattr(model, PARENT_ENTITY_TYPE_ATTR, None)


def entity_type_depth(type_name: str, entity_types: dict[str, type[BaseModel]] | None) -> int:
    if not entity_types:
        return 0
    depth = 0
    visited = {type_name}
    current = type_name
    while current in entity_types:
        parent = parent_entity_type(entity_types[current])
        if not parent or parent in visited or parent not in entity_types:
            break
        visited.add(parent)
        depth += 1
        current = parent
    return depth


def most_specific_entity_type_name(
    labels: list[str], entity_types: dict[str, type[BaseModel]] | None
) -> str:
    selected = ''
    selected_depth = -1
    for label in labels:
        if label == 'Entity':
            continue
        depth = entity_type_depth(label, entity_types)
        if depth > selected_depth:
            selected = label
            selected_depth = depth
    return selected


def build_entity_type_hierarchy(
    entity_types: dict[str, type[BaseModel]] | None,
) -> dict[str, list[str]]:
    if not entity_types:
        return {}
    children: dict[str, list[str]] = {}
    for child_name, model in entity_types.items():
        parent = parent_entity_type(model)
        if parent in entity_types:
            children.setdefault(parent, []).append(child_name)
    for names in children.values():
        names.sort()
    return children


def top_level_entity_types(
    entity_types: dict[str, type[BaseModel]] | None,
) -> dict[str, type[BaseModel]] | None:
    if entity_types is None:
        return None
    return {
        name: model
        for name, model in entity_types.items()
        if parent_entity_type(model) not in entity_types
    }


def _type_chain(
    type_name: str,
    entity_types: dict[str, type[BaseModel]],
) -> list[dict[str, str | None]]:
    chain: list[dict[str, str | None]] = []
    visited: set[str] = set()
    current = type_name
    while current in entity_types and current not in visited:
        visited.add(current)
        model = entity_types[current]
        chain.append({'name': current, 'description': model.__doc__})
        parent = parent_entity_type(model)
        if not parent or parent in visited or parent not in entity_types:
            break
        current = parent
    chain.reverse()
    return chain


async def classify_entity_subtypes(
    llm_client: Any,
    nodes: list[EntityNode],
    episodes: list[EpisodicNode],
    node_episode_index_map: dict[str, list[int]],
    entity_types: dict[str, type[BaseModel]] | None,
    custom_extraction_instructions: str | None = None,
    max_coroutines: int | None = None,
) -> None:
    if not entity_types:
        return

    children = build_entity_type_hierarchy(entity_types)
    if not children:
        return

    async def classify_node(node: EntityNode) -> None:
        current = most_specific_entity_type_name(node.labels, entity_types)
        calls = 0
        while calls < MAX_ENTITY_TYPE_SUBTYPE_DEPTH:
            candidates = children.get(current, [])
            if not candidates:
                return

            episode_indices = node_episode_index_map.get(node.uuid, [])
            node_episodes = [
                episodes[index] for index in episode_indices if 0 <= index < len(episodes)
            ]
            if not node_episodes:
                node_episodes = episodes
            context = {
                'entity_name': node.name,
                'chain': _type_chain(current, entity_types),
                'candidates': [
                    {
                        'index': index,
                        'name': name,
                        'description': entity_types[name].__doc__,
                    }
                    for index, name in enumerate(candidates, start=1)
                ],
                'episode_content': [episode.content for episode in node_episodes],
                'custom_extraction_instructions': custom_extraction_instructions,
            }
            calls += 1
            try:
                result = await llm_client.generate_response(
                    prompt_library.classify_entity_subtype.classify(context),
                    response_model=EntitySubtypeClassification,
                    model_size=ModelSize.small,
                    group_id=node.group_id,
                    prompt_name='classify_entity_subtype',
                )
            except Exception as exc:
                logger.info(
                    'Entity subtype classification failed',
                    extra={
                        'node_uuid': node.uuid,
                        'exception_type': type(exc).__name__,
                    },
                )
                return

            subtype_index = (
                result.get('subtype_index')
                if isinstance(result, dict)
                else getattr(result, 'subtype_index', 0)
            )
            if subtype_index == 0:
                logger.debug('No entity subtype selected', extra={'node_uuid': node.uuid})
                return
            if not isinstance(subtype_index, int) or not 1 <= subtype_index <= len(candidates):
                logger.info('Entity subtype index out of range', extra={'node_uuid': node.uuid})
                return

            current = candidates[subtype_index - 1]
            if current not in node.labels:
                node.labels.append(current)

    work = [classify_node(node) for node in nodes]
    if work:
        await semaphore_gather(*work, max_coroutines=max_coroutines)
