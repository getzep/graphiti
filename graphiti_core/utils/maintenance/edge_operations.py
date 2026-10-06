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
from graphiti_core.graphiti_types import (
    GraphitiClients,
    generate_prompt_response,
    uses_prompt_routing,
)
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate
from graphiti_core.prompts.extract_edges import Edge as ExtractedEdge
from graphiti_core.prompts.extract_edges import (
    EdgeTimestamps,
    ExtractedEdges,
)
from graphiti_core.search.search import search
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now
from graphiti_core.utils.maintenance.attribute_utils import apply_capped_attributes
from graphiti_core.utils.maintenance.dangling_endpoints import (
    materialize_dangling_endpoint_node,
    prune_unreferenced_materialized_nodes,
)
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact
from graphiti_core.utils.maintenance.temporal_edge_utils import (
    apply_extracted_timestamps,
    normalize_future_temporal_bounds,
)
from graphiti_core.utils.text_utils import concatenate_episodes

logger = logging.getLogger(__name__)

DEFAULT_EDGE_NAME = 'RELATES_TO'


def build_episodic_edges(
    entity_nodes: list[EntityNode],
    episode_uuid: str | list[str],
    created_at: datetime,
    node_episode_index_map: dict[str, list[int]] | None = None,
) -> list[EpisodicEdge]:
    """Build episodic (MENTIONED_IN) edges between entity nodes and episodes.

    Parameters
    ----------
    entity_nodes : list[EntityNode]
        Nodes to connect to episodes.
    episode_uuid : str | list[str]
        A single episode UUID or a list of episode UUIDs.
    created_at : datetime
        Timestamp for the edges.
    node_episode_index_map : dict[str, list[int]] | None
        Optional mapping from node UUID to 0-indexed episode positions.
        When provided with a list of episode_uuids, each node is connected
        only to its attributed episodes. When None, every node is connected
        to all episodes.
    """
    episode_uuids = [episode_uuid] if isinstance(episode_uuid, str) else episode_uuid

    episodic_edges: list[EpisodicEdge] = []
    for node in entity_nodes:
        if node_episode_index_map and node.uuid in node_episode_index_map:
            indices = node_episode_index_map[node.uuid]
        else:
            indices = list(range(len(episode_uuids)))

        for idx in indices:
            if 0 <= idx < len(episode_uuids):
                episodic_edges.append(
                    EpisodicEdge(
                        source_node_uuid=episode_uuids[idx],
                        target_node_uuid=node.uuid,
                        created_at=created_at,
                        group_id=node.group_id,
                    )
                )

    logger.debug(f'Built {len(episodic_edges)} episodic edges')

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
    episode: EpisodicNode | list[EpisodicNode],
    nodes: list[EntityNode],
    previous_episodes: list[EpisodicNode],
    edge_type_map: dict[tuple[str, str], list[str]],
    group_id: str = '',
    edge_types: dict[str, type[BaseModel]] | None = None,
    strict_edge_types: bool = False,
    custom_extraction_instructions: str | None = None,
) -> tuple[list[EntityEdge], list[EntityNode]]:
    """Extract edges from one or more episodes.

    Parameters
    ----------
    episode : EpisodicNode | list[EpisodicNode]
        A single episode or a list of episodes to extract edges from.
        When a list is provided, their contents are concatenated for extraction
        and edges are linked to all episode UUIDs.

    Returns
    -------
    tuple[list[EntityEdge], list[EntityNode]]
        Extracted edges and any endpoint nodes materialized onto ``nodes`` for
        dangling source/target names that survived edge filters.
    """
    episodes = episode if isinstance(episode, list) else [episode]
    primary_episode = episodes[0]

    start = time()
    initial_node_uuids = {node.uuid for node in nodes}

    extract_edges_max_tokens = 16384
    llm_client = clients.llm_client

    # Build mapping from edge type name to list of valid signatures
    edge_type_signatures_map: dict[str, list[tuple[str, str]]] = {}
    for signature, edge_type_names in edge_type_map.items():
        for edge_type in edge_type_names:
            if edge_type not in edge_type_signatures_map:
                edge_type_signatures_map[edge_type] = []
            edge_type_signatures_map[edge_type].append(signature)

    edge_types_context = (
        [
            {
                'fact_type_name': type_name,
                'fact_type_signatures': edge_type_signatures_map.get(
                    type_name, [('Entity', 'Entity')]
                ),
                'fact_type_description': type_model.__doc__,
            }
            for type_name, type_model in edge_types.items()
        ]
        if edge_types is not None
        else []
    )
    if strict_edge_types and not edge_types_context:
        logger.debug('No edge types available for strict edge extraction')
        return [], []

    allowed_edge_type_names = set(edge_types or {}) if strict_edge_types else None

    # Build normalized name-to-node mapping for validation / materialization
    name_to_node: dict[str, EntityNode] = {
        _normalize_string_exact(node.name): node for node in nodes
    }

    # Build episode attribution instructions for multi-episode extraction
    episode_attribution = ''
    if len(episodes) > 1:
        episode_attribution = (
            '\n8. **Episode Attribution**: The CURRENT_MESSAGE contains multiple episodes labeled '
            '[Episode 0], [Episode 1], etc. Each episode header includes a timestamp indicating '
            'when that episode occurred. Use the per-episode timestamp to resolve relative time '
            'mentions within each episode rather than relying solely on REFERENCE_TIME. '
            'For each extracted fact, set `episode_indices` '
            'to the 0-based list of episode numbers that the fact was derived from. '
            'A fact sourced from Episodes 0 and 1 should have `episode_indices: [0, 1]`.'
        )

    # Prepare context for LLM
    # Use the latest episode's timestamp as the primary reference time
    latest_episode = max(episodes, key=lambda ep: ep.valid_at)
    context = {
        'episode_content': concatenate_episodes(episodes),
        'nodes': [{'name': node.name, 'entity_types': node.labels} for node in nodes],
        'previous_episodes': [
            {
                'content': ep.content,
                'timestamp': ep.valid_at.isoformat() if ep.valid_at else None,
            }
            for ep in previous_episodes
        ],
        'reference_time': latest_episode.valid_at,
        'edge_types': edge_types_context,
        'strict_edge_types': strict_edge_types,
        'custom_extraction_instructions': (custom_extraction_instructions or '')
        + episode_attribution,
    }

    llm_response = await generate_prompt_response(
        llm_client,
        'extract_edges.edge',
        prompt_library.extract_edges.edge,
        context,
        clients=clients,
        response_model=ExtractedEdges,
        max_tokens=extract_edges_max_tokens,
        group_id=group_id or primary_episode.group_id,
    )
    all_edges_data = ExtractedEdges(**llm_response).edges

    edge_group_id = group_id or primary_episode.group_id

    # Validate entity names; materialize dangling endpoints only for edges we keep
    edges_data: list[ExtractedEdge] = []
    for edge_data in all_edges_data:
        source_name = edge_data.source_entity_name.strip()
        target_name = edge_data.target_entity_name.strip()
        if not source_name or not target_name:
            logger.warning('Skipping edge with empty source or target entity name')
            continue

        if not edge_data.fact.strip():
            continue

        if (
            allowed_edge_type_names is not None
            and edge_data.relation_type not in allowed_edge_type_names
        ):
            logger.debug(
                'Skipping edge with relation type "%s" not in strict edge ontology',
                edge_data.relation_type,
            )
            continue

        source_node = name_to_node.get(_normalize_string_exact(source_name))
        if source_node is None:
            source_node = materialize_dangling_endpoint_node(
                source_name,
                group_id=edge_group_id,
                name_to_node=name_to_node,
                nodes=nodes,
            )

        target_node = name_to_node.get(_normalize_string_exact(target_name))
        if target_node is None:
            target_node = materialize_dangling_endpoint_node(
                target_name,
                group_id=edge_group_id,
                name_to_node=name_to_node,
                nodes=nodes,
            )

        # Keep canonical names on the edge payload for the conversion loop below
        edge_data.source_entity_name = source_node.name
        edge_data.target_entity_name = target_node.name
        edges_data.append(edge_data)

    end = time()
    logger.debug(f'Extracted {len(edges_data)} new edges in {(end - start) * 1000:.0f} ms')

    if len(edges_data) == 0:
        materialized_nodes = prune_unreferenced_materialized_nodes(
            nodes,
            initial_uuids=initial_node_uuids,
            referenced_uuids=set(),
        )
        return [], materialized_nodes

    # Convert the extracted data into EntityEdge objects
    edges = []
    for edge_data in edges_data:
        # Validate Edge Date information
        valid_at = edge_data.valid_at
        invalid_at = edge_data.invalid_at
        valid_at_datetime = None
        invalid_at_datetime = None

        # Names already validated / rewritten to canonical node names above
        source_node = name_to_node.get(_normalize_string_exact(edge_data.source_entity_name))
        target_node = name_to_node.get(_normalize_string_exact(edge_data.target_entity_name))

        if source_node is None or target_node is None:
            logger.warning('Could not find source or target node for extracted edge')
            continue

        source_node_uuid = source_node.uuid
        target_node_uuid = target_node.uuid

        if valid_at:
            try:
                valid_at_datetime = ensure_utc(
                    datetime.fromisoformat(valid_at.replace('Z', '+00:00'))
                )
            except ValueError:
                logger.warning('Error parsing valid_at date, skipping')

        if invalid_at:
            try:
                invalid_at_datetime = ensure_utc(
                    datetime.fromisoformat(invalid_at.replace('Z', '+00:00'))
                )
            except ValueError as e:
                logger.warning(f'WARNING: Error parsing invalid_at date: {e}. Input: {invalid_at}')

        # Map episode_indices (0-indexed) to episode UUIDs.
        # Clamp indices to valid range and fall back to all episodes if empty.
        edge_episode_uuids = []
        for idx in edge_data.episode_indices:
            if 0 <= idx < len(episodes):
                edge_episode_uuids.append(episodes[idx].uuid)
        if not edge_episode_uuids:
            edge_episode_uuids = [ep.uuid for ep in episodes]

        edge_reference_time = (
            episodes[edge_data.episode_indices[0]].valid_at
            if edge_data.episode_indices and 0 <= edge_data.episode_indices[0] < len(episodes)
            else primary_episode.valid_at
        )
        valid_at_datetime, invalid_at_datetime = normalize_future_temporal_bounds(
            edge_data.fact,
            edge_reference_time,
            valid_at_datetime,
            invalid_at_datetime,
        )

        edge = EntityEdge(
            source_node_uuid=source_node_uuid,
            target_node_uuid=target_node_uuid,
            name=edge_data.relation_type,
            group_id=group_id or primary_episode.group_id,
            fact=edge_data.fact,
            episodes=edge_episode_uuids,
            created_at=utc_now(),
            valid_at=valid_at_datetime,
            invalid_at=invalid_at_datetime,
            reference_time=edge_reference_time,
        )
        edges.append(edge)
        logger.debug(
            f'Created new edge {edge.uuid} from {edge.source_node_uuid} to {edge.target_node_uuid}'
        )

    logger.debug(f'Extracted edges: {[e.uuid for e in edges]}')

    referenced_uuids = {edge.source_node_uuid for edge in edges} | {
        edge.target_node_uuid for edge in edges
    }
    materialized_nodes = prune_unreferenced_materialized_nodes(
        nodes,
        initial_uuids=initial_node_uuids,
        referenced_uuids=referenced_uuids,
    )
    if materialized_nodes:
        logger.warning(
            'Materialized %d dangling edge endpoint node(s) for group_id=%s',
            len(materialized_nodes),
            edge_group_id,
        )

    return edges, materialized_nodes


def _pair_search_filter(edge: EntityEdge) -> SearchFilters:
    """Filter restricting an edge search to the extracted edge's endpoints.

    src IN {a, b} AND tgt IN {a, b} covers both orientations in one query.
    The filter can reduce the candidate set before ranking. It also
    admits self-loops on either endpoint when a != b -- callers drop those
    with _same_pair_edges.
    """
    endpoints = [edge.source_node_uuid, edge.target_node_uuid]
    return SearchFilters(
        edge_source_node_uuids=endpoints,
        edge_target_node_uuids=endpoints,
    )


def _same_pair_edges(extracted_edge: EntityEdge, candidates: list[EntityEdge]) -> list[EntityEdge]:
    """Keep only edges between the extracted edge's endpoints, either
    orientation (drops the self-loops a {a,b}x{a,b} endpoint filter admits)."""
    pair = {
        (extracted_edge.source_node_uuid, extracted_edge.target_node_uuid),
        (extracted_edge.target_node_uuid, extracted_edge.source_node_uuid),
    }
    return [edge for edge in candidates if (edge.source_node_uuid, edge.target_node_uuid) in pair]


async def resolve_extracted_edges(
    clients: GraphitiClients,
    extracted_edges: list[EntityEdge],
    episode: EpisodicNode,
    entities: list[EntityNode],
    edge_types: dict[str, type[BaseModel]],
    edge_type_map: dict[tuple[str, str], list[str]],
    strict_edge_types: bool = False,
    invalidated_by: dict[str, list[str]] | None = None,
) -> tuple[list[EntityEdge], list[EntityEdge], list[EntityEdge]]:
    """Resolve extracted edges against existing graph context.

    An edge whose name is a custom edge type must join the entity types that the
    type's signature declares in ``edge_type_map``. A mismatched edge is dropped
    when ``strict_edge_types`` is set and renamed to ``DEFAULT_EDGE_NAME`` otherwise.

    When `invalidated_by` is given, it is filled with the uuid of every
    invalidated edge mapped to the uuids of the resolved edges that invalidated it.

    Returns
    -------
    tuple[list[EntityEdge], list[EntityEdge], list[EntityEdge]]
        A tuple of (resolved_edges, invalidated_edges, new_edges) where:
        - resolved_edges: All edges after resolution (may include existing edges if duplicates found)
        - invalidated_edges: Edges that were invalidated/contradicted by new information
        - new_edges: Only edges that are new to the graph (not duplicates of existing edges)
    """
    driver = clients.driver
    llm_client = clients.llm_client
    embedder = clients.embedder

    # Build entity hash table
    uuid_entity_map: dict[str, EntityNode] = {entity.uuid: entity for entity in entities}

    # Collect all node UUIDs referenced by edges that are not in the entities list
    referenced_node_uuids = set()
    for extracted_edge in extracted_edges:
        if extracted_edge.source_node_uuid not in uuid_entity_map:
            referenced_node_uuids.add(extracted_edge.source_node_uuid)
        if extracted_edge.target_node_uuid not in uuid_entity_map:
            referenced_node_uuids.add(extracted_edge.target_node_uuid)

    # Fetch missing nodes from the database
    if referenced_node_uuids:
        # Limit the lookup to the edge group.
        edge_group_id = extracted_edges[0].group_id
        missing_nodes = await EntityNode.get_by_uuids(
            driver, list(referenced_node_uuids), group_id=edge_group_id
        )
        for node in missing_nodes:
            uuid_entity_map[node.uuid] = node

    # Determine which edge types are relevant for each edge based on node signatures.
    # `edge_types_lst` stores the subset of custom edge definitions whose
    # node signature matches each extracted edge.
    edge_types_lst: list[dict[str, type[BaseModel]]] = []
    for extracted_edge in extracted_edges:
        edge_types_lst.append(
            _edge_types_for_endpoints(extracted_edge, uuid_entity_map, edge_types, edge_type_map)
        )

    extracted_edges, edge_types_lst = enforce_edge_type_signatures(
        extracted_edges, edge_types_lst, edge_type_map, strict_edge_types
    )

    # Fast path: deduplicate same-direction and same-type reverse-direction matches
    seen: dict[tuple[str, str, str], EntityEdge] = {}
    deduplicated_edges: list[EntityEdge] = []
    deduplicated_edge_types: list[dict[str, type[BaseModel]]] = []

    for edge, matching_edge_types in zip(extracted_edges, edge_types_lst, strict=True):
        key = (
            edge.source_node_uuid,
            edge.target_node_uuid,
            _normalize_string_exact(edge.fact),
        )
        if key in seen:
            continue

        reverse_key = (
            edge.target_node_uuid,
            edge.source_node_uuid,
            _normalize_string_exact(edge.fact),
        )
        reverse_edge = seen.get(reverse_key)
        if reverse_edge is not None and reverse_edge.name == edge.name:
            continue

        seen[key] = edge
        deduplicated_edges.append(edge)
        deduplicated_edge_types.append(matching_edge_types)

    extracted_edges = deduplicated_edges
    edge_types_lst = deduplicated_edge_types

    await create_entity_edge_embeddings(embedder, extracted_edges)

    # Duplicate candidates have the same endpoints in either direction.
    # Prefiltering the search by endpoints limits the search work. The post-filter
    # removes self-loops and protects against indexes that return unrelated edges.
    # Reuse each computed fact embedding to avoid another embedder request.
    related_edges_results: list[SearchResults] = await semaphore_gather(
        *[
            search(
                clients,
                extracted_edge.fact,
                group_ids=[extracted_edge.group_id],
                config=EDGE_HYBRID_SEARCH_RRF,
                search_filter=_pair_search_filter(extracted_edge),
                query_vector=extracted_edge.fact_embedding,
            )
            for extracted_edge in extracted_edges
        ],
        max_coroutines=getattr(clients, 'max_coroutines', None),
    )
    related_edges_lists: list[list[EntityEdge]] = [
        [candidate for candidate in _same_pair_edges(extracted_edge, result.edges)]
        for extracted_edge, result in zip(extracted_edges, related_edges_results, strict=True)
    ]

    edge_invalidation_candidate_results: list[SearchResults] = await semaphore_gather(
        *[
            search(
                clients,
                extracted_edge.fact,
                group_ids=[extracted_edge.group_id],
                config=EDGE_HYBRID_SEARCH_RRF,
                search_filter=SearchFilters(),
                query_vector=extracted_edge.fact_embedding,
            )
            for extracted_edge in extracted_edges
        ],
        max_coroutines=getattr(clients, 'max_coroutines', None),
    )

    resolver_kwargs = {'clients': clients} if uses_prompt_routing(clients) else {}
    edge_invalidation_candidates: list[list[EntityEdge]] = []
    for _extracted_edge, related_edges, invalidation_result in zip(
        extracted_edges,
        related_edges_lists,
        edge_invalidation_candidate_results,
        strict=True,
    ):
        related_uuids = {edge.uuid for edge in related_edges}
        deduplicated = [
            edge for edge in invalidation_result.edges if edge.uuid not in related_uuids
        ]
        edge_invalidation_candidates.append(deduplicated)

    logger.debug(
        f'Related edges: {[e.uuid for edges_lst in related_edges_lists for e in edges_lst]}'
    )

    dedupe_results: list[tuple[EntityEdge, list[EntityEdge], list[EntityEdge] | None]] = list(
        await semaphore_gather(
            *[
                _dedupe_extracted_edge(
                    llm_client,
                    extracted_edge,
                    related_edges,
                    existing_edges,
                    episode,
                    extracted_edge_types,
                    **resolver_kwargs,
                )
                for extracted_edge, related_edges, existing_edges, extracted_edge_types in zip(
                    extracted_edges,
                    related_edges_lists,
                    edge_invalidation_candidates,
                    edge_types_lst,
                    strict=True,
                )
            ],
            max_coroutines=getattr(clients, 'max_coroutines', None),
        )
    )

    resolved_edges: list[EntityEdge] = []
    invalidated_edges: list[EntityEdge] = []
    new_edges: list[EntityEdge] = []
    for extracted_edge, (resolved_edge, _duplicates, _candidates) in zip(
        extracted_edges, dedupe_results, strict=True
    ):
        resolved_edges.append(resolved_edge)

        # Track edges that are new (not duplicates of existing edges)
        # An edge is new if the resolved edge UUID matches the extracted edge UUID
        if resolved_edge.uuid == extracted_edge.uuid:
            new_edges.append(resolved_edge)

    logger.debug(f'Resolved edges: {[e.uuid for e in resolved_edges]}')
    logger.debug(f'New edges (non-duplicates): {[e.uuid for e in new_edges]}')

    for resolved_edge, _duplicates, candidates in dedupe_results:
        if candidates is None:
            continue
        invalidated = _apply_temporal_invalidation(resolved_edge, candidates)
        invalidated_edges.extend(invalidated)
        if invalidated_by is not None:
            for edge in invalidated:
                invalidated_by.setdefault(edge.uuid, []).append(resolved_edge.uuid)

    await semaphore_gather(
        create_entity_edge_embeddings(embedder, resolved_edges),
        create_entity_edge_embeddings(embedder, invalidated_edges),
        max_coroutines=getattr(clients, 'max_coroutines', None),
    )

    return resolved_edges, invalidated_edges, new_edges


def _edge_types_for_endpoints(
    extracted_edge: EntityEdge,
    uuid_entity_map: dict[str, EntityNode],
    edge_types: dict[str, type[BaseModel]],
    edge_type_map: dict[tuple[str, str], list[str]],
) -> dict[str, type[BaseModel]]:
    """Return the custom edge types whose signature matches the edge's endpoint labels."""
    source_node = uuid_entity_map.get(extracted_edge.source_node_uuid)
    target_node = uuid_entity_map.get(extracted_edge.target_node_uuid)
    source_node_labels = source_node.labels + ['Entity'] if source_node is not None else ['Entity']
    target_node_labels = target_node.labels + ['Entity'] if target_node is not None else ['Entity']

    matching_edge_types: dict[str, type[BaseModel]] = {}
    for source_label in source_node_labels:
        for target_label in target_node_labels:
            for type_name in edge_type_map.get((source_label, target_label), []):
                type_model = edge_types.get(type_name)
                if type_model is not None:
                    matching_edge_types[type_name] = type_model
    return matching_edge_types


def enforce_edge_type_signatures(
    extracted_edges: list[EntityEdge],
    edge_types_lst: list[dict[str, type[BaseModel]]],
    edge_type_map: dict[tuple[str, str], list[str]],
    strict_edge_types: bool,
) -> tuple[list[EntityEdge], list[dict[str, type[BaseModel]]]]:
    """Apply the custom edge type signatures to the extracted edges.

    ``edge_types_lst[i]`` holds the custom edge types whose signature matches the
    endpoint labels of ``extracted_edges[i]``. An edge named after an edge type
    that declares at least one signature in ``edge_type_map``, but is not in its
    matching set, violates the signature. Such an edge is dropped when
    ``strict_edge_types`` is set, and renamed to ``DEFAULT_EDGE_NAME`` otherwise.
    Edges named after a type with no declared signature pass through unchanged,
    because such a type applies between any entity types.
    """
    signed_edge_types = {name for names in edge_type_map.values() for name in names}
    kept_edges: list[EntityEdge] = []
    kept_edge_types: list[dict[str, type[BaseModel]]] = []
    for extracted_edge, matching_edge_types in zip(extracted_edges, edge_types_lst, strict=True):
        if (
            extracted_edge.name in signed_edge_types
            and extracted_edge.name not in matching_edge_types
        ):
            if strict_edge_types:
                logger.info(
                    'Dropping edge %s: name %s does not match the endpoint signature',
                    extracted_edge.uuid,
                    extracted_edge.name,
                )
                continue
            logger.info(
                'Renaming edge %s from %s to %s: name does not match the endpoint signature',
                extracted_edge.uuid,
                extracted_edge.name,
                DEFAULT_EDGE_NAME,
            )
            extracted_edge.name = DEFAULT_EDGE_NAME
            extracted_edge.attributes = {}
        kept_edges.append(extracted_edge)
        kept_edge_types.append(matching_edge_types)
    return kept_edges, kept_edge_types


def _apply_temporal_invalidation(
    resolved_edge: EntityEdge,
    invalidation_candidates: list[EntityEdge],
) -> list[EntityEdge]:
    """Expire the resolved edge and any candidate it contradicts.

    Every check here compares dates, so the caller must have dated the edge first.
    Returns the candidate edges this edge invalidated.
    """
    now = utc_now()

    # The edge arrived with an end date, so it is already historical.
    if resolved_edge.invalid_at and not resolved_edge.expired_at:
        resolved_edge.expired_at = now

    # A candidate that starts later supersedes this edge, so expire this one instead.
    if resolved_edge.expired_at is None:
        invalidation_candidates.sort(key=lambda c: (c.valid_at is None, ensure_utc(c.valid_at)))
        for candidate in invalidation_candidates:
            candidate_valid_at_utc = ensure_utc(candidate.valid_at)
            resolved_edge_valid_at_utc = ensure_utc(resolved_edge.valid_at)
            if (
                candidate_valid_at_utc is not None
                and resolved_edge_valid_at_utc is not None
                and candidate_valid_at_utc > resolved_edge_valid_at_utc
            ):
                resolved_edge.invalid_at = candidate.valid_at
                resolved_edge.expired_at = now
                break

    return resolve_edge_contradictions(resolved_edge, invalidation_candidates)


def resolve_edge_contradictions(
    resolved_edge: EntityEdge, invalidation_candidates: list[EntityEdge]
) -> list[EntityEdge]:
    if len(invalidation_candidates) == 0:
        return []

    # Determine which contradictory edges need to be expired
    invalidated_edges: list[EntityEdge] = []
    for edge in invalidation_candidates:
        # (Edge invalid before new edge becomes valid) or (new edge invalid before edge becomes valid)
        edge_invalid_at_utc = ensure_utc(edge.invalid_at)
        resolved_edge_valid_at_utc = ensure_utc(resolved_edge.valid_at)
        edge_valid_at_utc = ensure_utc(edge.valid_at)
        resolved_edge_invalid_at_utc = ensure_utc(resolved_edge.invalid_at)

        if (
            edge_invalid_at_utc is not None
            and resolved_edge_valid_at_utc is not None
            and edge_invalid_at_utc <= resolved_edge_valid_at_utc
        ) or (
            edge_valid_at_utc is not None
            and resolved_edge_invalid_at_utc is not None
            and resolved_edge_invalid_at_utc <= edge_valid_at_utc
        ):
            continue
        # New edge invalidates edge
        elif (
            edge_valid_at_utc is not None
            and resolved_edge_valid_at_utc is not None
            and edge_valid_at_utc < resolved_edge_valid_at_utc
        ):
            edge.invalid_at = resolved_edge.valid_at
            edge.expired_at = edge.expired_at if edge.expired_at is not None else utc_now()
            invalidated_edges.append(edge)

    return invalidated_edges


async def _extract_edge_timestamps(
    llm_client: LLMClient,
    edge: EntityEdge,
    episode: EpisodicNode | None,
    *,
    clients: GraphitiClients | None = None,
) -> None:
    """Extract valid_at / invalid_at timestamps for an edge via a lightweight LLM call.

    Modifies the edge in place. Skips if the edge already has timestamps set
    (e.g., from the extraction prompt in the separate-extraction path) or if
    no reference time is available.
    clients: optional bundle that selects prompt overrides and model routes.
    """
    if edge.valid_at is not None or edge.invalid_at is not None:
        return

    reference_time = ensure_utc(
        edge.reference_time or (episode.valid_at if episode is not None else None)
    )
    if reference_time is None:
        return

    context = {
        'fact': edge.fact,
        'reference_time': reference_time.isoformat(),
    }
    try:
        llm_response = await generate_prompt_response(
            llm_client,
            'extract_edges.extract_timestamps',
            prompt_library.extract_edges.extract_timestamps,
            context,
            clients=clients,
            response_model=EdgeTimestamps,
            model_size=ModelSize.small,
        )
        timestamps = EdgeTimestamps(**llm_response)
        apply_extracted_timestamps(
            edge,
            timestamps.valid_at,
            timestamps.invalid_at,
            reference_time,
        )
    except Exception:
        logger.warning('Failed to extract timestamps for edge %s', edge.uuid, exc_info=True)


async def resolve_extracted_edge(
    llm_client: LLMClient,
    extracted_edge: EntityEdge,
    related_edges: list[EntityEdge],
    existing_edges: list[EntityEdge],
    episode: EpisodicNode,
    edge_type_candidates: dict[str, type[BaseModel]] | None = None,
    *,
    clients: GraphitiClients | None = None,
) -> tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]:
    """Fully resolve one edge on its own: dedupe it, date it, then apply contradictions.

    This wrapper resolves one edge and applies its temporal contradictions.
    """
    resolver_kwargs = {'clients': clients} if clients is not None else {}
    resolved_edge, duplicate_edges, candidates = await _dedupe_extracted_edge(
        llm_client,
        extracted_edge,
        related_edges,
        existing_edges,
        episode,
        edge_type_candidates,
        **resolver_kwargs,
    )
    if candidates is None:
        return resolved_edge, [], duplicate_edges
    invalidated_edges = _apply_temporal_invalidation(resolved_edge, candidates)
    return resolved_edge, invalidated_edges, duplicate_edges


async def _dedupe_extracted_edge(
    llm_client: LLMClient,
    extracted_edge: EntityEdge,
    related_edges: list[EntityEdge],
    existing_edges: list[EntityEdge],
    episode: EpisodicNode,
    edge_type_candidates: dict[str, type[BaseModel]] | None = None,
    *,
    clients: GraphitiClients | None = None,
) -> tuple[EntityEdge, list[EntityEdge], list[EntityEdge] | None]:
    """Deduplicate an extracted edge and gather the edges it might contradict.

    Stops before contradiction so the caller can apply temporal invalidation after
    the edge is dated.
    Returns (resolved_edge, duplicate_edges, invalidation_candidates). Candidates is
    None when the edge took an early return and must skip contradiction entirely;
    an empty list would instead mean "no candidates, but still run the checks".

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
    clients: optional bundle that selects prompt overrides and model routes.

    Returns
    -------
    tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]
        The resolved edge, any duplicates, and edges to invalidate.
    """
    resolver_kwargs = {'clients': clients} if clients is not None else {}

    # Nothing to compare against, so there is no duplicate and nothing to contradict.
    if len(related_edges) == 0 and len(existing_edges) == 0:
        # Still extract custom attributes and timestamps even when no dedup needed
        edge_model = edge_type_candidates.get(extracted_edge.name) if edge_type_candidates else None
        if edge_model is not None and len(edge_model.model_fields) != 0:
            edge_attributes_context = {
                'fact': extracted_edge.fact,
                'reference_time': episode.valid_at if episode is not None else None,
                'existing_attributes': extracted_edge.attributes,
            }
            edge_attributes_response = await generate_prompt_response(
                llm_client,
                'extract_edges.extract_attributes',
                prompt_library.extract_edges.extract_attributes,
                edge_attributes_context,
                clients=clients,
                response_model=edge_model,  # type: ignore
                model_size=ModelSize.small,
                attribute_extraction=True,
            )
            merged, _ = apply_capped_attributes(
                edge_attributes_response,
                edge_model,
                extracted_edge.attributes,
                merge_mode='replace',
                prompt_name='extract_edges.extract_attributes',
                entity_uuid=extracted_edge.uuid,
                group_id=extracted_edge.group_id,
            )
            extracted_edge.attributes = merged

        await _extract_edge_timestamps(llm_client, extracted_edge, episode, **resolver_kwargs)

        return extracted_edge, [], None

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
            return resolved, [], None

    start = time()

    # Prepare context for LLM with continuous indexing
    related_edges_context = [{'idx': i, 'fact': edge.fact} for i, edge in enumerate(related_edges)]

    # Invalidation candidates start where duplicate candidates end
    invalidation_idx_offset = len(related_edges)
    invalidation_edge_candidates_context = [
        {'idx': invalidation_idx_offset + i, 'fact': existing_edge.fact}
        for i, existing_edge in enumerate(existing_edges)
    ]

    context = {
        'existing_edges': related_edges_context,
        'new_edge': extracted_edge.fact,
        'edge_invalidation_candidates': invalidation_edge_candidates_context,
    }

    if related_edges or existing_edges:
        logger.debug(
            'Resolving edge: sent %d EXISTING FACTS%s and %d INVALIDATION CANDIDATES%s',
            len(related_edges),
            f' (idx 0-{len(related_edges) - 1})' if related_edges else '',
            len(existing_edges),
            f' (idx {invalidation_idx_offset}-{invalidation_idx_offset + len(existing_edges) - 1})'
            if existing_edges
            else '',
        )

    llm_response = await generate_prompt_response(
        llm_client,
        'dedupe_edges.resolve_edge',
        prompt_library.dedupe_edges.resolve_edge,
        context,
        clients=clients,
        response_model=EdgeDuplicate,
        model_size=ModelSize.small,
    )
    response_object = EdgeDuplicate(**llm_response)
    duplicate_facts = response_object.duplicate_facts

    # Validate duplicate_facts are in valid range for EXISTING FACTS
    invalid_duplicates = [i for i in duplicate_facts if i < 0 or i >= len(related_edges)]
    if invalid_duplicates:
        logger.warning(
            'LLM returned invalid duplicate_facts idx values %s (valid range: 0-%d for EXISTING FACTS)',
            invalid_duplicates,
            len(related_edges) - 1,
        )

    duplicate_fact_ids: list[int] = [i for i in duplicate_facts if 0 <= i < len(related_edges)]

    resolved_edge = extracted_edge
    for duplicate_fact_id in duplicate_fact_ids:
        resolved_edge = related_edges[duplicate_fact_id]
        break

    if duplicate_fact_ids and episode is not None:
        resolved_edge.episodes.append(episode.uuid)

    # Process contradicted facts (continuous indexing across both lists)
    contradicted_facts: list[int] = response_object.contradicted_facts
    invalidation_candidates: list[EntityEdge] = []

    # Only process contradictions if there are edges to check against
    if related_edges or existing_edges:
        max_valid_idx = len(related_edges) + len(existing_edges) - 1
        invalid_contradictions = [i for i in contradicted_facts if i < 0 or i > max_valid_idx]
        if invalid_contradictions:
            logger.warning(
                'LLM returned invalid contradicted_facts idx values %s (valid range: 0-%d)',
                invalid_contradictions,
                max_valid_idx,
            )

        # Split contradicted facts into those from related_edges vs existing_edges based on offset
        for idx in contradicted_facts:
            if 0 <= idx < len(related_edges):
                # From EXISTING FACTS (duplicate candidates)
                invalidation_candidates.append(related_edges[idx])
            elif invalidation_idx_offset <= idx <= max_valid_idx:
                # From FACT INVALIDATION CANDIDATES (adjust index by offset)
                invalidation_candidates.append(existing_edges[idx - invalidation_idx_offset])

    # Only extract structured attributes if the edge's relation_type matches an allowed custom type
    # AND the edge model exists for this node pair signature
    edge_model = edge_type_candidates.get(resolved_edge.name) if edge_type_candidates else None
    if edge_model is not None and len(edge_model.model_fields) != 0:
        edge_attributes_context = {
            'fact': resolved_edge.fact,
            'reference_time': episode.valid_at if episode is not None else None,
            'existing_attributes': resolved_edge.attributes,
        }

        edge_attributes_response = await generate_prompt_response(
            llm_client,
            'extract_edges.extract_attributes',
            prompt_library.extract_edges.extract_attributes,
            edge_attributes_context,
            clients=clients,
            response_model=edge_model,  # type: ignore
            model_size=ModelSize.small,
            attribute_extraction=True,
        )
        merged, _ = apply_capped_attributes(
            edge_attributes_response,
            edge_model,
            resolved_edge.attributes,
            merge_mode='replace',
            prompt_name='extract_edges.extract_attributes',
            entity_uuid=resolved_edge.uuid,
            group_id=resolved_edge.group_id,
        )
        resolved_edge.attributes = merged
    elif edge_type_candidates is not None:
        # Schema map was supplied but this relation has no model. Clear
        # attributes that belong to a prior schema. Do not wipe when the
        # caller omitted the map: add_triplet puts caller attributes on the
        # edge and never passes edge_type_candidates.
        resolved_edge.attributes = {}

    # Extract timestamps for new edges (duplicated edges retain their existing timestamps)
    if resolved_edge.uuid == extracted_edge.uuid:
        await _extract_edge_timestamps(llm_client, resolved_edge, episode, **resolver_kwargs)

    end = time()
    logger.debug(
        f'Resolved Edge: {extracted_edge.uuid} -> {resolved_edge.uuid}, in {(end - start) * 1000} ms'
    )

    duplicate_edges: list[EntityEdge] = [related_edges[idx] for idx in duplicate_fact_ids]

    return resolved_edge, duplicate_edges, invalidation_candidates


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
