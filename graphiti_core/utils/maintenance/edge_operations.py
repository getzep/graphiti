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

from graphiti_core.driver.driver import GraphDriver, GraphProvider
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
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate
from graphiti_core.prompts.extract_edges import ExtractedEdges, MissingFacts
from graphiti_core.search.search import search
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact

DEFAULT_EDGE_NAME = 'RELATES_TO'

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
    edge_types: dict[str, type[BaseModel]] | None = None,
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
        'ensure_ascii': clients.ensure_ascii,
    }

    facts_missed = True
    reflexion_iterations = 0
    while facts_missed and reflexion_iterations <= MAX_REFLEXION_ITERATIONS:
        llm_response = await llm_client.generate_response(
            prompt_library.extract_edges.edge(context),
            response_model=ExtractedEdges,
            max_tokens=extract_edges_max_tokens,
        )
        edges_data = ExtractedEdges(**llm_response).edges

        context['extracted_facts'] = [edge_data.fact for edge_data in edges_data]

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
        valid_at = edge_data.valid_at
        invalid_at = edge_data.invalid_at
        valid_at_datetime = None
        invalid_at_datetime = None

        source_node_idx = edge_data.source_entity_id
        target_node_idx = edge_data.target_entity_id
        if not (-1 < source_node_idx < len(nodes) and -1 < target_node_idx < len(nodes)):
            logger.warning(
                f'WARNING: source or target node not filled {edge_data.relation_type}. source_node_uuid: {source_node_idx} and target_node_uuid: {target_node_idx} '
            )
            continue
        source_node_uuid = nodes[source_node_idx].uuid
        target_node_uuid = nodes[edge_data.target_entity_id].uuid

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
            name=edge_data.relation_type,
            group_id=group_id,
            fact=edge_data.fact,
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
    edge_types: dict[str, type[BaseModel]],
    edge_type_map: dict[tuple[str, str], list[str]],
) -> tuple[list[EntityEdge], list[EntityEdge]]:
    driver = clients.driver
    llm_client = clients.llm_client
    embedder = clients.embedder
    await create_entity_edge_embeddings(embedder, extracted_edges)

    valid_edges_list: list[list[EntityEdge]] = await semaphore_gather(
        *[
            EntityEdge.get_between_nodes(driver, edge.source_node_uuid, edge.target_node_uuid)
            for edge in extracted_edges
        ]
    )

    related_edges_results: list[SearchResults] = await semaphore_gather(
        *[
            search(
                clients,
                extracted_edge.fact,
                group_ids=[extracted_edge.group_id],
                config=EDGE_HYBRID_SEARCH_RRF,
                search_filter=SearchFilters(edge_uuids=[edge.uuid for edge in valid_edges]),
            )
            for extracted_edge, valid_edges in zip(extracted_edges, valid_edges_list, strict=True)
        ]
    )

    related_edges_lists: list[list[EntityEdge]] = [result.edges for result in related_edges_results]

    edge_invalidation_candidate_results: list[SearchResults] = await semaphore_gather(
        *[
            search(
                clients,
                extracted_edge.fact,
                group_ids=[extracted_edge.group_id],
                config=EDGE_HYBRID_SEARCH_RRF,
                search_filter=SearchFilters(),
            )
            for extracted_edge in extracted_edges
        ]
    )

    edge_invalidation_candidates: list[list[EntityEdge]] = [
        result.edges for result in edge_invalidation_candidate_results
    ]

    logger.debug(
        f'Related edges lists: {[(e.name, e.uuid) for edges_lst in related_edges_lists for e in edges_lst]}'
    )

    # Build entity hash table
    uuid_entity_map: dict[str, EntityNode] = {entity.uuid: entity for entity in entities}

    # Determine which edge types are relevant for each edge.
    # `edge_types_lst` stores the subset of custom edge definitions whose
    # node signature matches each extracted edge. Anything outside this subset
    # should only stay on the edge if it is a non-custom (LLM generated) label.
    edge_types_lst: list[dict[str, type[BaseModel]]] = []
    custom_type_names = set(edge_types or {})
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

    for extracted_edge, extracted_edge_types in zip(extracted_edges, edge_types_lst, strict=True):
        allowed_type_names = set(extracted_edge_types)
        is_custom_name = extracted_edge.name in custom_type_names
        if not allowed_type_names:
            # No custom types are valid for this node pairing. Keep LLM generated
            # labels, but flip disallowed custom names back to the default.
            if is_custom_name and extracted_edge.name != DEFAULT_EDGE_NAME:
                extracted_edge.name = DEFAULT_EDGE_NAME
            continue
        if is_custom_name and extracted_edge.name not in allowed_type_names:
            # Custom name exists but it is not permitted for this source/target
            # signature, so fall back to the default edge label.
            extracted_edge.name = DEFAULT_EDGE_NAME

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
                    custom_type_names,
                    clients.ensure_ascii,
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
    edge_type_candidates: dict[str, type[BaseModel]] | None = None,
    custom_edge_type_names: set[str] | None = None,
    ensure_ascii: bool = True,
) -> tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]:
    """Resolve an extracted edge against existing graph context.

    Parameters
    ----------
    llm_client : LLMClient
        Client used to invoke the LLM for deduplication and attribute extraction.
    extracted_edge : EntityEdge
        Newly extracted edge whose canonical representation is being resolved.
    related_edges : list[EntityEdge]
        Candidate edges with identical endpoints used for duplicate detection.
    existing_edges : list[EntityEdge]
        Broader set of edges evaluated for contradiction / invalidation.
    episode : EpisodicNode
        Episode providing content context when extracting edge attributes.
    edge_type_candidates : dict[str, type[BaseModel]] | None
        Custom edge types permitted for the current source/target signature.
    custom_edge_type_names : set[str] | None
        Full catalog of registered custom edge names. Used to distinguish
        between disallowed custom types (which fall back to the default label)
        and ad-hoc labels emitted by the LLM.
    ensure_ascii : bool
        Whether prompt payloads should coerce ASCII output.

    Returns
    -------
    tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]
        The resolved edge, any duplicates, and edges to invalidate.
    """
    if len(related_edges) == 0 and len(existing_edges) == 0:
        return extracted_edge, [], []

    # Fast path: if the fact text and endpoints already exist verbatim, reuse the matching edge.
    normalized_fact = _normalize_string_exact(extracted_edge.fact)
    for edge in related_edges:
        if (
            edge.source_node_uuid == extracted_edge.source_node_uuid
            and edge.target_node_uuid == extracted_edge.target_node_uuid
            and _normalize_string_exact(edge.fact) == normalized_fact
        ):
            resolved = edge
            if episode is not None and episode.uuid not in resolved.episodes:
                resolved.episodes.append(episode.uuid)
            return resolved, [], []

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
            for i, (type_name, type_model) in enumerate(edge_type_candidates.items())
        ]
        if edge_type_candidates is not None
        else []
    )

    context = {
        'existing_edges': related_edges_context,
        'new_edge': extracted_edge.fact,
        'edge_invalidation_candidates': invalidation_edge_candidates_context,
        'edge_types': edge_types_context,
        'ensure_ascii': ensure_ascii,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.dedupe_edges.resolve_edge(context),
        response_model=EdgeDuplicate,
        model_size=ModelSize.small,
    )
    response_object = EdgeDuplicate(**llm_response)
    duplicate_facts = response_object.duplicate_facts

    duplicate_fact_ids: list[int] = [i for i in duplicate_facts if 0 <= i < len(related_edges)]

    resolved_edge = extracted_edge
    for duplicate_fact_id in duplicate_fact_ids:
        resolved_edge = related_edges[duplicate_fact_id]
        break

    if duplicate_fact_ids and episode is not None:
        resolved_edge.episodes.append(episode.uuid)

    contradicted_facts: list[int] = response_object.contradicted_facts

    invalidation_candidates: list[EntityEdge] = [
        existing_edges[i] for i in contradicted_facts if 0 <= i < len(existing_edges)
    ]

    fact_type: str = response_object.fact_type
    candidate_type_names = set(edge_type_candidates or {})
    custom_type_names = custom_edge_type_names or set()

    is_default_type = fact_type.upper() == 'DEFAULT'
    is_custom_type = fact_type in custom_type_names
    is_allowed_custom_type = fact_type in candidate_type_names

    if is_allowed_custom_type:
        # The LLM selected a custom type that is allowed for the node pair.
        # Adopt the custom type and, if needed, extract its structured attributes.
        resolved_edge.name = fact_type

        edge_attributes_context = {
            'episode_content': episode.content,
            'reference_time': episode.valid_at,
            'fact': resolved_edge.fact,
            'ensure_ascii': ensure_ascii,
        }

        edge_model = edge_type_candidates.get(fact_type) if edge_type_candidates else None
        if edge_model is not None and len(edge_model.model_fields) != 0:
            edge_attributes_response = await llm_client.generate_response(
                prompt_library.extract_edges.extract_attributes(edge_attributes_context),
                response_model=edge_model,  # type: ignore
                model_size=ModelSize.small,
            )

            resolved_edge.attributes = edge_attributes_response
    elif not is_default_type and is_custom_type:
        # The LLM picked a custom type that is not allowed for this signature.
        # Reset to the default label and drop any structured attributes.
        resolved_edge.name = DEFAULT_EDGE_NAME
        resolved_edge.attributes = {}
    elif not is_default_type:
        # Non-custom labels are allowed to pass through so long as the LLM does
        # not return the sentinel DEFAULT value.
        resolved_edge.name = fact_type
        resolved_edge.attributes = {}

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


async def filter_existing_duplicate_of_edges(
    driver: GraphDriver, duplicates_node_tuples: list[tuple[EntityNode, EntityNode]]
) -> list[tuple[EntityNode, EntityNode]]:
    if not duplicates_node_tuples:
        return []

    duplicate_nodes_map = {
        (source.uuid, target.uuid): (source, target) for source, target in duplicates_node_tuples
    }

    if driver.provider == GraphProvider.NEPTUNE:
        query: LiteralString = """
            UNWIND $duplicate_node_uuids AS duplicate_tuple
            MATCH (n:Entity {uuid: duplicate_tuple.source})-[r:RELATES_TO {name: 'IS_DUPLICATE_OF'}]->(m:Entity {uuid: duplicate_tuple.target})
            RETURN DISTINCT
                n.uuid AS source_uuid,
                m.uuid AS target_uuid
        """

        duplicate_nodes = [
            {'source': source.uuid, 'target': target.uuid}
            for source, target in duplicates_node_tuples
        ]

        records, _, _ = await driver.execute_query(
            query,
            duplicate_node_uuids=duplicate_nodes,
            routing_='r',
        )
    else:
        if driver.provider == GraphProvider.KUZU:
            query = """
                UNWIND $duplicate_node_uuids AS duplicate
                MATCH (n:Entity {uuid: duplicate.src})-[:RELATES_TO]->(e:RelatesToNode_ {name: 'IS_DUPLICATE_OF'})-[:RELATES_TO]->(m:Entity {uuid: duplicate.dst})
                RETURN DISTINCT
                    n.uuid AS source_uuid,
                    m.uuid AS target_uuid
            """
            duplicate_node_uuids = [{'src': src, 'dst': dst} for src, dst in duplicate_nodes_map]
        else:
            query: LiteralString = """
                UNWIND $duplicate_node_uuids AS duplicate_tuple
                MATCH (n:Entity {uuid: duplicate_tuple[0]})-[r:RELATES_TO {name: 'IS_DUPLICATE_OF'}]->(m:Entity {uuid: duplicate_tuple[1]})
                RETURN DISTINCT
                    n.uuid AS source_uuid,
                    m.uuid AS target_uuid
            """
            duplicate_node_uuids = list(duplicate_nodes_map.keys())

        records, _, _ = await driver.execute_query(
            query,
            duplicate_node_uuids=duplicate_node_uuids,
            routing_='r',
        )

    # Remove duplicates that already have the IS_DUPLICATE_OF edge
    for record in records:
        duplicate_tuple = (record.get('source_uuid'), record.get('target_uuid'))
        if duplicate_nodes_map.get(duplicate_tuple):
            duplicate_nodes_map.pop(duplicate_tuple)

    return list(duplicate_nodes_map.values())
