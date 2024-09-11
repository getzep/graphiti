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
from typing import List

from graphiti_core.edges import CommunityEdge, EntityEdge, EpisodicEdge
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.utils.maintenance.temporal_operations import (
    extract_edge_dates,
    get_edge_contradictions,
)

logger = logging.getLogger(__name__)


def build_episodic_edges(
    entity_nodes: List[EntityNode],
    episode: EpisodicNode,
    created_at: datetime,
) -> List[EpisodicEdge]:
    edges: List[EpisodicEdge] = [
        EpisodicEdge(
            source_node_uuid=episode.uuid,
            target_node_uuid=node.uuid,
            created_at=created_at,
            group_id=episode.group_id,
        )
        for node in entity_nodes
    ]

    return edges


def build_community_edges(
    entity_nodes: List[EntityNode],
    community_node: CommunityNode,
    created_at: datetime,
) -> List[CommunityEdge]:
    edges: List[CommunityEdge] = [
        CommunityEdge(
            source_node_uuid=community_node.uuid,
            target_node_uuid=node.uuid,
            created_at=created_at,
            group_id=community_node.group_id,
        )
        for node in entity_nodes
    ]

    return edges


async def extract_edges(
    llm_client: LLMClient,
    episode: EpisodicNode,
    nodes: list[EntityNode],
    previous_episodes: list[EpisodicNode],
    group_id: str | None,
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
                group_id=group_id,
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
    edge_map: dict[str, EntityEdge] = {}
    for edge in existing_edges:
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
    duplicate_data = llm_response.get('duplicates', [])
    logger.info(f'Extracted unique edges: {duplicate_data}')

    duplicate_uuid_map: dict[str, str] = {}
    for duplicate in duplicate_data:
        uuid_value = duplicate['duplicate_of']
        duplicate_uuid_map[duplicate['uuid']] = uuid_value

    # Get full edge data
    edges: list[EntityEdge] = []
    for edge in extracted_edges:
        if edge.uuid in duplicate_uuid_map:
            existing_uuid = duplicate_uuid_map[edge.uuid]
            existing_edge = edge_map[existing_uuid]
            edges.append(existing_edge)
        else:
            edges.append(edge)

    return edges


async def resolve_extracted_edges(
    llm_client: LLMClient,
    extracted_edges: list[EntityEdge],
    related_edges_lists: list[list[EntityEdge]],
    existing_edges_lists: list[list[EntityEdge]],
    current_episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
) -> tuple[list[EntityEdge], list[EntityEdge]]:
    # resolve edges with related edges in the graph, extract temporal information, and find invalidation candidates
    results: list[tuple[EntityEdge, list[EntityEdge]]] = list(
        await asyncio.gather(
            *[
                resolve_extracted_edge(
                    llm_client,
                    extracted_edge,
                    related_edges,
                    existing_edges,
                    current_episode,
                    previous_episodes,
                )
                for extracted_edge, related_edges, existing_edges in zip(
                    extracted_edges, related_edges_lists, existing_edges_lists
                )
            ]
        )
    )

    resolved_edges: list[EntityEdge] = []
    invalidated_edges: list[EntityEdge] = []
    for result in results:
        resolved_edge = result[0]
        invalidated_edge_chunk = result[1]

        resolved_edges.append(resolved_edge)
        invalidated_edges.extend(invalidated_edge_chunk)

    return resolved_edges, invalidated_edges


async def resolve_extracted_edge(
    llm_client: LLMClient,
    extracted_edge: EntityEdge,
    related_edges: list[EntityEdge],
    existing_edges: list[EntityEdge],
    current_episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
) -> tuple[EntityEdge, list[EntityEdge]]:
    resolved_edge, (valid_at, invalid_at), invalidation_candidates = await asyncio.gather(
        dedupe_extracted_edge(llm_client, extracted_edge, related_edges),
        extract_edge_dates(llm_client, extracted_edge, current_episode, previous_episodes),
        get_edge_contradictions(llm_client, extracted_edge, existing_edges),
    )

    now = datetime.now()

    resolved_edge.valid_at = valid_at if valid_at is not None else resolved_edge.valid_at
    resolved_edge.invalid_at = invalid_at if invalid_at is not None else resolved_edge.invalid_at
    if invalid_at is not None and resolved_edge.expired_at is None:
        resolved_edge.expired_at = now

    # Determine if the new_edge needs to be expired
    if resolved_edge.expired_at is None:
        invalidation_candidates.sort(key=lambda c: (c.valid_at is None, c.valid_at))
        for candidate in invalidation_candidates:
            if (
                candidate.valid_at is not None and resolved_edge.valid_at is not None
            ) and candidate.valid_at > resolved_edge.valid_at:
                # Expire new edge since we have information about more recent events
                resolved_edge.invalid_at = candidate.valid_at
                resolved_edge.expired_at = now
                break

    # Determine which contradictory edges need to be expired
    invalidated_edges: list[EntityEdge] = []
    for edge in invalidation_candidates:
        # (Edge invalid before new edge becomes valid) or (new edge invalid before edge becomes valid)
        if (
            edge.invalid_at is not None
            and resolved_edge.valid_at is not None
            and edge.invalid_at < resolved_edge.valid_at
        ) or (
            edge.valid_at is not None
            and resolved_edge.invalid_at is not None
            and resolved_edge.invalid_at < edge.valid_at
        ):
            continue
        # New edge invalidates edge
        elif (
            edge.valid_at is not None
            and resolved_edge.valid_at is not None
            and edge.valid_at < resolved_edge.valid_at
        ):
            edge.invalid_at = resolved_edge.valid_at
            edge.expired_at = edge.expired_at if edge.expired_at is not None else now
            invalidated_edges.append(edge)

    return resolved_edge, invalidated_edges


async def dedupe_extracted_edge(
    llm_client: LLMClient, extracted_edge: EntityEdge, related_edges: list[EntityEdge]
) -> EntityEdge:
    start = time()

    # Prepare context for LLM
    related_edges_context = [
        {'uuid': edge.uuid, 'name': edge.name, 'fact': edge.fact} for edge in related_edges
    ]

    extracted_edge_context = {
        'uuid': extracted_edge.uuid,
        'name': extracted_edge.name,
        'fact': extracted_edge.fact,
    }

    context = {
        'related_edges': related_edges_context,
        'extracted_edges': extracted_edge_context,
    }

    llm_response = await llm_client.generate_response(prompt_library.dedupe_edges.v3(context))

    is_duplicate: bool = llm_response.get('is_duplicate', False)
    uuid: str | None = llm_response.get('uuid', None)

    edge = extracted_edge
    if is_duplicate:
        for existing_edge in related_edges:
            if existing_edge.uuid != uuid:
                continue
            edge = existing_edge

    end = time()
    logger.info(
        f'Resolved Edge: {extracted_edge.name} is {edge.name}, in {(end - start) * 1000} ms'
    )

    return edge


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
