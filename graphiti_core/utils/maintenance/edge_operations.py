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

from pydantic import BaseModel
from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphDriver
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

logger = logging.getLogger(__name__)


def build_episodic_edges(
    entity_nodes: list[EntityNode],
    episode_uuid: str,
    created_at: datetime,
) -> list[EpisodicEdge]:
    episodic_edges: list[EpisodicEdge] = [
        EpisodicEdge(
            source_node_uuid=episode_uuid,
            target_node_uuid=node.uuid,
            created_at=created_at,
            group_id=node.group_id,
        )
        for node in entity_nodes
    ]

    logger.debug(f'Built episodic edges: {episodic_edges}')

    return episodic_edges


def build_duplicate_of_edges(
    episode: EpisodicNode,
    created_at: datetime,
    duplicate_nodes: list[tuple[EntityNode, EntityNode]],
) -> list[EntityEdge]:
    is_duplicate_of_edges: list[EntityEdge] = []
    for source_node, target_node in duplicate_nodes:
        if source_node.uuid == target_node.uuid:
            continue

        is_duplicate_of_edges.append(
            EntityEdge(
                source_node_uuid=source_node.uuid,
                target_node_uuid=target_node.uuid,
                name='IS_DUPLICATE_OF',
                group_id=episode.group_id,
                fact=f'{source_node.name} is a duplicate of {target_node.name}',
                episodes=[episode.uuid],
                created_at=created_at,
                valid_at=created_at,
            )
        )

    return is_duplicate_of_edges


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
    edge_type_map: dict[tuple[str, str], list[str]],
    group_id: str = '',
    edge_types: dict[str, BaseModel] | None = None,
) -> list[EntityEdge]:
    start = time()

    extract_edges_max_tokens = 16384
    llm_client = clients.llm_client

    edge_type_signature_map: dict[str, tuple[str, str]] = {
        edge_type: signature
        for signature, edge_types in edge_type_map.items()
        for edge_type in edge_types
    }

    edge_types_context = (
        [
            {
                'fact_type_name': type_name,
                'fact_type_signature': edge_type_signature_map.get(type_name, ('Entity', 'Entity')),
                'fact_type_description': type_model.__doc__,
            }
            for type_name, type_model in edge_types.items()
        ]
        if edge_types is not None
        else []
    )

    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'nodes': [
            {'id': idx, 'name': node.name, 'entity_types': node.labels}
            for idx, node in enumerate(nodes)
        ],
        'previous_episodes': [ep.content for ep in previous_episodes],
        'reference_time': episode.valid_at,
        'edge_types': edge_types_context,
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

        source_node_idx = edge_data.get('source_entity_id', -1)
        target_node_idx = edge_data.get('target_entity_id', -1)
        if not (-1 < source_node_idx < len(nodes) and -1 < target_node_idx < len(nodes)):
            logger.warning(
                f'WARNING: source or target node not filled {edge_data.get("edge_name")}. source_node_uuid: {source_node_idx} and target_node_uuid: {target_node_idx} '
            )
            continue
        source_node_uuid = nodes[source_node_idx].uuid
        target_node_uuid = nodes[edge_data.get('target_entity_id')].uuid

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
            source_node_uuid=source_node_uuid,
            target_node_uuid=target_node_uuid,
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

    logger.debug(f'Extracted edges: {[(e.name, e.uuid) for e in edges]}')

    return edges


async def resolve_extracted_edges(
    clients: GraphitiClients,
    extracted_edges: list[EntityEdge],
    episode: EpisodicNode,
    entities: list[EntityNode],
    edge_types: dict[str, BaseModel],
    edge_type_map: dict[tuple[str, str], list[str]],
) -> tuple[list[EntityEdge], list[EntityEdge]]:
    driver = clients.driver
    llm_client = clients.llm_client
    embedder = clients.embedder
    await create_entity_edge_embeddings(embedder, extracted_edges)

    search_results = await semaphore_gather(
        get_relevant_edges(driver, extracted_edges, SearchFilters()),
        get_edge_invalidation_candidates(driver, extracted_edges, SearchFilters(), 0.2),
    )

    related_edges_lists, edge_invalidation_candidates = search_results

    logger.debug(
        f'Related edges lists: {[(e.name, e.uuid) for edges_lst in related_edges_lists for e in edges_lst]}'
    )

    # Build entity hash table
    uuid_entity_map: dict[str, EntityNode] = {entity.uuid: entity for entity in entities}

    # Determine which edge types are relevant for each edge
    edge_types_lst: list[dict[str, BaseModel]] = []
    for extracted_edge in extracted_edges:
        source_node = uuid_entity_map.get(extracted_edge.source_node_uuid)
        target_node = uuid_entity_map.get(extracted_edge.target_node_uuid)
        source_node_labels = (
            source_node.labels + ['Entity'] if source_node is not None else ['Entity']
        )
        target_node_labels = (
            target_node.labels + ['Entity'] if target_node is not None else ['Entity']
        )
        label_tuples = [
            (source_label, target_label)
            for source_label in source_node_labels
            for target_label in target_node_labels
        ]

        extracted_edge_types = {}
        for label_tuple in label_tuples:
            type_names = edge_type_map.get(label_tuple, [])
            for type_name in type_names:
                type_model = edge_types.get(type_name)
                if type_model is None:
                    continue

                extracted_edge_types[type_name] = type_model

        edge_types_lst.append(extracted_edge_types)

    # resolve edges with related edges in the graph and find invalidation candidates
    results: list[tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]] = list(
        await semaphore_gather(
            *[
                resolve_extracted_edge(
                    llm_client,
                    extracted_edge,
                    related_edges,
                    existing_edges,
                    episode,
                    extracted_edge_types,
                )
                for extracted_edge, related_edges, existing_edges, extracted_edge_types in zip(
                    extracted_edges,
                    related_edges_lists,
                    edge_invalidation_candidates,
                    edge_types_lst,
                    strict=True,
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

    await semaphore_gather(
        create_entity_edge_embeddings(embedder, resolved_edges),
        create_entity_edge_embeddings(embedder, invalidated_edges),
    )

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
    episode: EpisodicNode,
    edge_types: dict[str, BaseModel] | None = None,
) -> tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]:
    if len(related_edges) == 0 and len(existing_edges) == 0:
        return extracted_edge, [], []

    start = time()

    # Prepare context for LLM
    related_edges_context = [
        {'id': edge.uuid, 'fact': edge.fact} for i, edge in enumerate(related_edges)
    ]

    invalidation_edge_candidates_context = [
        {'id': i, 'fact': existing_edge.fact} for i, existing_edge in enumerate(existing_edges)
    ]

    edge_types_context = (
        [
            {
                'fact_type_id': i,
                'fact_type_name': type_name,
                'fact_type_description': type_model.__doc__,
            }
            for i, (type_name, type_model) in enumerate(edge_types.items())
        ]
        if edge_types is not None
        else []
    )

    context = {
        'existing_edges': related_edges_context,
        'new_edge': extracted_edge.fact,
        'edge_invalidation_candidates': invalidation_edge_candidates_context,
        'edge_types': edge_types_context,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.dedupe_edges.resolve_edge(context),
        response_model=EdgeDuplicate,
        model_size=ModelSize.small,
    )

    duplicate_fact_ids: list[int] = list(
        filter(lambda i: 0 <= i < len(related_edges), llm_response.get('duplicate_facts', []))
    )

    resolved_edge = extracted_edge
    for duplicate_fact_id in duplicate_fact_ids:
        resolved_edge = related_edges[duplicate_fact_id]
        break

    if duplicate_fact_ids and episode is not None:
        resolved_edge.episodes.append(episode.uuid)

    contradicted_facts: list[int] = llm_response.get('contradicted_facts', [])

    invalidation_candidates: list[EntityEdge] = [existing_edges[i] for i in contradicted_facts]

    fact_type: str = str(llm_response.get('fact_type'))
    if fact_type.upper() != 'DEFAULT' and edge_types is not None:
        resolved_edge.name = fact_type

        edge_attributes_context = {
            'episode_content': episode.content,
            'reference_time': episode.valid_at,
            'fact': resolved_edge.fact,
        }

        edge_model = edge_types.get(fact_type)
        if edge_model is not None and len(edge_model.model_fields) != 0:
            edge_attributes_response = await llm_client.generate_response(
                prompt_library.extract_edges.extract_attributes(edge_attributes_context),
                response_model=edge_model,  # type: ignore
                model_size=ModelSize.small,
            )

            resolved_edge.attributes = edge_attributes_response

    end = time()
    logger.debug(
        f'Resolved Edge: {extracted_edge.name} is {resolved_edge.name}, in {(end - start) * 1000} ms'
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
    invalidated_edges: list[EntityEdge] = resolve_edge_contradictions(
        resolved_edge, invalidation_candidates
    )
    duplicate_edges: list[EntityEdge] = [related_edges[idx] for idx in duplicate_fact_ids]

    return resolved_edge, invalidated_edges, duplicate_edges


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


async def filter_existing_duplicate_of_edges(
    driver: GraphDriver, duplicates_node_tuples: list[tuple[EntityNode, EntityNode]]
) -> list[tuple[EntityNode, EntityNode]]:
    query: LiteralString = """
        UNWIND $duplicate_node_uuids AS duplicate_tuple
        MATCH (n:Entity {uuid: duplicate_tuple[0]})-[r:RELATES_TO {name: 'IS_DUPLICATE_OF'}]->(m:Entity {uuid: duplicate_tuple[1]})
        RETURN DISTINCT
            n.uuid AS source_uuid,
            m.uuid AS target_uuid
    """

    duplicate_nodes_map = {
        (source.uuid, target.uuid): (source, target) for source, target in duplicates_node_tuples
    }

    records, _, _ = await driver.execute_query(
        query,
        duplicate_node_uuids=list(duplicate_nodes_map.keys()),
        routing_='r',
    )

    # Remove duplicates that already have the IS_DUPLICATE_OF edge
    for record in records:
        duplicate_tuple = (record.get('source_uuid'), record.get('target_uuid'))
        if duplicate_nodes_map.get(duplicate_tuple):
            duplicate_nodes_map.pop(duplicate_tuple)

    return list(duplicate_nodes_map.values())
