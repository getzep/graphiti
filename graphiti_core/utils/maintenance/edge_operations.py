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

from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    create_entity_edge_embeddings,
)
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import MAX_REFLEXION_ITERATIONS, semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate, UniqueFacts
from graphiti_core.prompts.extract_edges import ExtractedEdges, MissingFacts
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import get_edge_invalidation_candidates, get_relevant_edges
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now
from graphiti_core.utils.maintenance.temporal_operations import (
    get_edge_contradictions,
)

logger = logging.getLogger(__name__)


def build_episodic_edges(
    entity_nodes: list[EntityNode],
    episode: EpisodicNode,
    created_at: datetime,
) -> list[EpisodicEdge]:
    episodic_edges: list[EpisodicEdge] = [
        EpisodicEdge(
            source_node_uuid=episode.uuid,
            target_node_uuid=node.uuid,
            created_at=created_at,
            group_id=episode.group_id,
        )
        for node in entity_nodes
    ]

    logger.debug(f'Built episodic edges: {episodic_edges}')

    return episodic_edges


def build_community_edges(
    entity_nodes: list[EntityNode],
    community_node: CommunityNode,
    created_at: datetime,
) -> list[CommunityEdge]:
    edges: list[CommunityEdge] = [
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
    clients: GraphitiClients,
    episode: EpisodicNode,
    nodes: list[EntityNode],
    previous_episodes: list[EpisodicNode],
    group_id: str = '',
) -> list[EntityEdge]:
    start = time()

    extract_edges_max_tokens = 16384
    llm_client = clients.llm_client
    embedder = clients.embedder

    node_uuids_by_name_map = {node.name: node.uuid for node in nodes}

    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'nodes': [node.name for node in nodes],
        'previous_episodes': [ep.content for ep in previous_episodes],
        'reference_time': episode.valid_at,
        'custom_prompt': '',
    }

    facts_missed = True
    reflexion_iterations = 0
    while facts_missed and reflexion_iterations <= MAX_REFLEXION_ITERATIONS:
        llm_response = await llm_client.generate_response(
            prompt_library.extract_edges.edge(context),
            response_model=ExtractedEdges,
            max_tokens=extract_edges_max_tokens,
        )
        edges_data = llm_response.get('edges', [])

        context['extracted_facts'] = [edge_data.get('fact', '') for edge_data in edges_data]

        reflexion_iterations += 1
        if reflexion_iterations < MAX_REFLEXION_ITERATIONS:
            reflexion_response = await llm_client.generate_response(
                prompt_library.extract_edges.reflexion(context),
                response_model=MissingFacts,
                max_tokens=extract_edges_max_tokens,
            )

            missing_facts = reflexion_response.get('missing_facts', [])

            custom_prompt = 'The following facts were missed in a previous extraction: '
            for fact in missing_facts:
                custom_prompt += f'\n{fact},'

            context['custom_prompt'] = custom_prompt

            facts_missed = len(missing_facts) != 0

    end = time()
    logger.debug(f'Extracted new edges: {edges_data} in {(end - start) * 1000} ms')

    if len(edges_data) == 0:
        return []

    # Convert the extracted data into EntityEdge objects
    edges = []
    for edge_data in edges_data:
        # Validate Edge Date information
        valid_at = edge_data.get('valid_at', None)
        invalid_at = edge_data.get('invalid_at', None)
        valid_at_datetime = None
        invalid_at_datetime = None

        if valid_at:
            try:
                valid_at_datetime = ensure_utc(
                    datetime.fromisoformat(valid_at.replace('Z', '+00:00'))
                )
            except ValueError as e:
                logger.warning(f'WARNING: Error parsing valid_at date: {e}. Input: {valid_at}')

        if invalid_at:
            try:
                invalid_at_datetime = ensure_utc(
                    datetime.fromisoformat(invalid_at.replace('Z', '+00:00'))
                )
            except ValueError as e:
                logger.warning(f'WARNING: Error parsing invalid_at date: {e}. Input: {invalid_at}')
        edge = EntityEdge(
            source_node_uuid=node_uuids_by_name_map.get(
                edge_data.get('source_entity_name', ''), ''
            ),
            target_node_uuid=node_uuids_by_name_map.get(
                edge_data.get('target_entity_name', ''), ''
            ),
            name=edge_data.get('relation_type', ''),
            group_id=group_id,
            fact=edge_data.get('fact', ''),
            episodes=[episode.uuid],
            created_at=utc_now(),
            valid_at=valid_at_datetime,
            invalid_at=invalid_at_datetime,
        )
        edges.append(edge)
        logger.debug(
            f'Created new edge: {edge.name} from (UUID: {edge.source_node_uuid}) to (UUID: {edge.target_node_uuid})'
        )

    await create_entity_edge_embeddings(embedder, edges)

    logger.debug(f'Extracted edges: {[(e.name, e.uuid) for e in edges]}')

    return edges


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

    llm_response = await llm_client.generate_response(prompt_library.dedupe_edges.edge(context))
    duplicate_data = llm_response.get('duplicates', [])
    logger.debug(f'Extracted unique edges: {duplicate_data}')

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
            # Add current episode to the episodes list
            existing_edge.episodes += edge.episodes
            edges.append(existing_edge)
        else:
            edges.append(edge)

    return edges


async def resolve_extracted_edges(
    clients: GraphitiClients,
    extracted_edges: list[EntityEdge],
) -> tuple[list[EntityEdge], list[EntityEdge]]:
    driver = clients.driver
    llm_client = clients.llm_client

    search_results: tuple[list[list[EntityEdge]], list[list[EntityEdge]]] = await semaphore_gather(
        get_relevant_edges(driver, extracted_edges, SearchFilters()),
        get_edge_invalidation_candidates(driver, extracted_edges, SearchFilters()),
    )

    related_edges_lists, edge_invalidation_candidates = search_results

    logger.debug(
        f'Related edges lists: {[(e.name, e.uuid) for edges_lst in related_edges_lists for e in edges_lst]}'
    )

    # resolve edges with related edges in the graph and find invalidation candidates
    results: list[tuple[EntityEdge, list[EntityEdge]]] = list(
        await semaphore_gather(
            *[
                resolve_extracted_edge(
                    llm_client,
                    extracted_edge,
                    related_edges,
                    existing_edges,
                )
                for extracted_edge, related_edges, existing_edges in zip(
                    extracted_edges, related_edges_lists, edge_invalidation_candidates, strict=True
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

    logger.debug(f'Resolved edges: {[(e.name, e.uuid) for e in resolved_edges]}')

    return resolved_edges, invalidated_edges


def resolve_edge_contradictions(
    resolved_edge: EntityEdge, invalidation_candidates: list[EntityEdge]
) -> list[EntityEdge]:
    if len(invalidation_candidates) == 0:
        return []

    # Determine which contradictory edges need to be expired
    invalidated_edges: list[EntityEdge] = []
    for edge in invalidation_candidates:
        # (Edge invalid before new edge becomes valid) or (new edge invalid before edge becomes valid)
        if (
            edge.invalid_at is not None
            and resolved_edge.valid_at is not None
            and edge.invalid_at <= resolved_edge.valid_at
        ) or (
            edge.valid_at is not None
            and resolved_edge.invalid_at is not None
            and resolved_edge.invalid_at <= edge.valid_at
        ):
            continue
        # New edge invalidates edge
        elif (
            edge.valid_at is not None
            and resolved_edge.valid_at is not None
            and edge.valid_at < resolved_edge.valid_at
        ):
            edge.invalid_at = resolved_edge.valid_at
            edge.expired_at = edge.expired_at if edge.expired_at is not None else utc_now()
            invalidated_edges.append(edge)

    return invalidated_edges


async def resolve_extracted_edge(
    llm_client: LLMClient,
    extracted_edge: EntityEdge,
    related_edges: list[EntityEdge],
    existing_edges: list[EntityEdge],
) -> tuple[EntityEdge, list[EntityEdge]]:
    resolved_edge, invalidation_candidates = await semaphore_gather(
        dedupe_extracted_edge(llm_client, extracted_edge, related_edges),
        get_edge_contradictions(llm_client, extracted_edge, existing_edges),
    )

    now = utc_now()

    if resolved_edge.invalid_at and not resolved_edge.expired_at:
        resolved_edge.expired_at = now

    # Determine if the new_edge needs to be expired
    if resolved_edge.expired_at is None:
        invalidation_candidates.sort(key=lambda c: (c.valid_at is None, c.valid_at))
        for candidate in invalidation_candidates:
            if (
                candidate.valid_at
                and resolved_edge.valid_at
                and candidate.valid_at.tzinfo
                and resolved_edge.valid_at.tzinfo
                and candidate.valid_at > resolved_edge.valid_at
            ):
                # Expire new edge since we have information about more recent events
                resolved_edge.invalid_at = candidate.valid_at
                resolved_edge.expired_at = now
                break

    # Determine which contradictory edges need to be expired
    invalidated_edges = resolve_edge_contradictions(resolved_edge, invalidation_candidates)

    return resolved_edge, invalidated_edges


async def dedupe_extracted_edge(
    llm_client: LLMClient, extracted_edge: EntityEdge, related_edges: list[EntityEdge]
) -> EntityEdge:
    if len(related_edges) == 0:
        return extracted_edge

    start = time()

    # Prepare context for LLM
    related_edges_context = [
        {'id': edge.uuid, 'fact': edge.fact} for i, edge in enumerate(related_edges)
    ]

    extracted_edge_context = {
        'fact': extracted_edge.fact,
    }

    context = {
        'related_edges': related_edges_context,
        'extracted_edges': extracted_edge_context,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.dedupe_edges.edge(context),
        response_model=EdgeDuplicate,
        model_size=ModelSize.small,
    )

    duplicate_fact_id: int = llm_response.get('duplicate_fact_id', -1)

    edge = (
        related_edges[duplicate_fact_id]
        if 0 <= duplicate_fact_id < len(related_edges)
        else extracted_edge
    )

    end = time()
    logger.debug(
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
        prompt_library.dedupe_edges.edge_list(context), response_model=UniqueFacts
    )
    unique_edges_data = llm_response.get('unique_facts', [])

    end = time()
    logger.debug(f'Extracted edge duplicates: {unique_edges_data} in {(end - start) * 1000} ms ')

    # Get full edge data
    unique_edges = []
    for edge_data in unique_edges_data:
        uuid = edge_data['uuid']
        edge = edge_map[uuid]
        edge.fact = edge_data['fact']
        unique_edges.append(edge)

    return unique_edges
