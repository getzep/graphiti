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
from itertools import zip_longest
from time import time
from uuid import uuid4

from pydantic import BaseModel

from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti_types import GraphitiClients, generate_prompt_response
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.extract_edges import BatchEdgeTimestamps
from graphiti_core.prompts.extract_nodes_and_edges import (
    CombinedExtraction,
    CombinedExtractionHyperedge,
    CombinedFact,
    CombinedFactHyperedge,
    EntityEndpointPair,
)
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.dangling_endpoints import materialize_dangling_endpoint_node
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact
from graphiti_core.utils.maintenance.hyperedge import group_hyperedges, untag_invalid_groups
from graphiti_core.utils.maintenance.node_operations import (
    _build_entity_types_context,
    _collapse_exact_duplicate_extracted_nodes,
    _filter_entity_types_context,
)
from graphiti_core.utils.maintenance.temporal_edge_utils import apply_extracted_timestamps
from graphiti_core.utils.text_utils import concatenate_episodes, concatenate_timeline

logger = logging.getLogger(__name__)


def _iter_fact_endpoint_pairs(
    edge_data: CombinedFact | CombinedFactHyperedge,
) -> list[EntityEndpointPair]:
    """Normalize pairwise and hyperedge facts into endpoint pairs.

    A hyperedge fact already carries pair objects, so they are returned as-is; only the
    pairwise shape needs wrapping. Endpoint names are stripped by the caller.
    """
    if isinstance(edge_data, CombinedFactHyperedge):
        return list(edge_data.entity_endpoints)
    return [
        EntityEndpointPair(
            source_entity_name=edge_data.source_entity_name,
            target_entity_name=edge_data.target_entity_name,
            relation_type=edge_data.relation_type,
        )
    ]


def _episode_uuids_for_fact(
    edge_data: CombinedFact | CombinedFactHyperedge,
    episodes: list[EpisodicNode],
) -> list[str]:
    episode_uuids = [
        episodes[idx].uuid for idx in edge_data.episode_indices if 0 <= idx < len(episodes)
    ]
    return episode_uuids or [episode.uuid for episode in episodes]


def _fact_group_key(
    edge_data: CombinedFactHyperedge,
    episode_uuids: list[str],
) -> tuple[str, str, tuple[str, ...]]:
    if edge_data.fact_group_id is None:
        model_group = 'fact'
    else:
        model_group = f'{type(edge_data.fact_group_id).__name__}:{edge_data.fact_group_id}'
    return (
        model_group,
        _normalize_string_exact(edge_data.fact),
        tuple(sorted(set(episode_uuids))),
    )


def _provisional_hyperedge_uuids(
    edges: list[CombinedFactHyperedge],
    episodes: list[EpisodicNode],
) -> dict[tuple[str, str, tuple[str, ...]], str]:
    """Assign response-local UUIDs before pairwise edge materialization."""
    pairs_by_group: dict[
        tuple[str, str, tuple[str, ...]],
        list[EntityEndpointPair],
    ] = {}
    for edge_data in edges:
        key = _fact_group_key(edge_data, _episode_uuids_for_fact(edge_data, episodes))
        pairs_by_group.setdefault(key, []).extend(edge_data.entity_endpoints)

    provisional: dict[tuple[str, str, tuple[str, ...]], str] = {}
    for key, pairs in pairs_by_group.items():
        endpoints = {
            _normalize_string_exact(name)
            for pair in pairs
            for name in (pair.source_entity_name, pair.target_entity_name)
            if name.strip()
        }
        if len(pairs) >= 2 and len(endpoints) >= 3:
            provisional[key] = str(uuid4())
    return provisional


async def extract_nodes_and_edges(
    clients: GraphitiClients,
    episode: EpisodicNode | list[EpisodicNode],
    previous_episodes: list[EpisodicNode],
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    edge_type_map: dict[tuple[str, str], list[str]] | None = None,
    edge_types: dict[str, type[BaseModel]] | None = None,
    strict_edge_types: bool = False,
    custom_extraction_instructions: str | None = None,
    enable_hyperedges: bool = False,
    timeline: list[EpisodicNode] | None = None,
) -> tuple[list[EntityNode], list[EntityEdge], dict[str, list[int]]]:
    """Extract entity nodes and relationship facts in a single LLM call.

    This combined extraction produces better results than separate node+edge
    extraction because the model can see both tasks simultaneously, ensuring
    every entity has at least one connecting fact and reducing orphaned nodes.

    Parameters
    ----------
    clients : GraphitiClients
        LLM and embedder clients.
    episode : EpisodicNode | list[EpisodicNode]
        A single episode or a list of episodes to extract from.
    previous_episodes : list[EpisodicNode]
        Prior episodes for context (not extracted from).
    entity_types : dict | None
        Custom entity type definitions.
    excluded_entity_types : list[str] | None
        Entity types to exclude from extraction.
    edge_type_map : dict | None
        Mapping of (source_type, target_type) tuples to lists of edge type names.
    edge_types : dict | None
        Custom edge type definitions (Pydantic models keyed by type name).
    strict_edge_types : bool
        Whether to discard edges that do not use one of the provided edge_types.
    custom_extraction_instructions : str | None
        Additional extraction instructions.
    enable_hyperedges : bool
        When True, use the hyperedge extraction prompt/schema where each fact may
        include multiple typed (source, target) endpoint projections. Multi-endpoint
        facts are flattened to pairwise EntityEdge records that share fact text.
    timeline : list[EpisodicNode] | None
        Ordered span containing the episodes plus interleaved context-only turns.
        When None, the episodes render alone.

    Returns
    -------
    tuple[list[EntityNode], list[EntityEdge], dict[str, list[int]]]
        A tuple of (nodes, edges, node_episode_index_map) where
        node_episode_index_map maps node UUID to 0-indexed episode positions.
    """
    episodes = episode if isinstance(episode, list) else [episode]
    primary_episode = episodes[0]

    start = time()
    llm_client = clients.llm_client

    # Build entity types context
    entity_types_context = _filter_entity_types_context(
        _build_entity_types_context(entity_types), excluded_entity_types
    )
    if not entity_types_context:
        logger.debug('No entity types available for combined extraction after exclusions')
        return [], [], {}

    # Build edge types context (same format as separate extraction path)
    edge_types_context: list[dict] = []
    if edge_types and edge_type_map:
        edge_type_signatures_map: dict[str, list] = {}
        for signature, type_names in edge_type_map.items():
            for type_name in type_names:
                if type_name not in edge_type_signatures_map:
                    edge_type_signatures_map[type_name] = []
                edge_type_signatures_map[type_name].append(signature)

        edge_types_context = [
            {
                'fact_type_name': type_name,
                'fact_type_signatures': edge_type_signatures_map.get(
                    type_name, [('Entity', 'Entity')]
                ),
                'fact_type_description': type_model.__doc__,
            }
            for type_name, type_model in edge_types.items()
        ]

    if strict_edge_types and not edge_types_context:
        logger.debug('No edge types available for strict combined extraction')
        return [], [], {}

    allowed_edge_type_names = set(edge_types or {}) if strict_edge_types else None

    # Build context for the combined prompt
    context = {
        'episode_content': (
            concatenate_timeline(episodes, timeline) if timeline else concatenate_episodes(episodes)
        ),
        'previous_episodes': [
            {
                'content': ep.content,
                'timestamp': ep.valid_at.isoformat() if ep.valid_at else None,
            }
            for ep in previous_episodes
        ],
        'custom_extraction_instructions': custom_extraction_instructions or '',
        'entity_types': entity_types_context,
        'edge_types': edge_types_context,
        'strict_edge_types': strict_edge_types,
    }

    # Single LLM call for combined extraction
    llm_response = await generate_prompt_response(
        llm_client,
        'extract_nodes_and_edges.extract_message',
        prompt_library.extract_nodes_and_edges.extract_message,
        context,
        clients=clients,
        response_model=CombinedExtraction,
        group_id=primary_episode.group_id,
    )
    response_object: CombinedExtraction | CombinedExtractionHyperedge
    if enable_hyperedges:
        response_object = CombinedExtractionHyperedge(**llm_response)
    else:
        response_object = CombinedExtraction(**llm_response)

    end = time()
    logger.debug(
        f'Combined extraction: {len(response_object.extracted_entities)} entities, '
        f'{len(response_object.edges)} edges in {(end - start) * 1000:.0f} ms'
        f'{" (hyperedges)" if enable_hyperedges else ""}'
    )

    # --- Process nodes ---

    # Filter empty names
    filtered_entities = [e for e in response_object.extracted_entities if e.name.strip()]

    # Convert CombinedEntity objects to EntityNode objects (no episode attribution yet —
    # that is derived from edges below).
    extracted_nodes: list[EntityNode] = []
    excluded_entity_names: set[str] = set()
    for entity in filtered_entities:
        type_id = entity.entity_type_id
        if 0 <= type_id < len(entity_types_context):
            entity_type_name = entity_types_context[type_id].get('entity_type_name')
        else:
            entity_type_name = 'Entity'

        if excluded_entity_types and entity_type_name in excluded_entity_types:
            excluded_entity_names.add(_normalize_string_exact(entity.name))
            logger.debug(f'Excluding entity of type "{entity_type_name}"')
            continue

        labels: list[str] = list({'Entity', str(entity_type_name)})
        new_node = EntityNode(
            name=entity.name,
            group_id=primary_episode.group_id,
            labels=labels,
            summary='',
            created_at=utc_now(),
        )
        extracted_nodes.append(new_node)

    # Collapse exact-duplicate nodes (same normalized name).
    # Temporarily use an empty map — real attribution comes from edges below.
    node_episode_index_map: dict[str, list[int]] = {}
    extracted_nodes = _collapse_exact_duplicate_extracted_nodes(
        extracted_nodes, node_episode_index_map
    )

    # --- Process edges ---

    # Build normalized name-to-node map so case/whitespace differences don't drop edges
    name_to_node: dict[str, EntityNode] = {
        _normalize_string_exact(node.name): node for node in extracted_nodes
    }

    extracted_edges: list[EntityEdge] = []
    materialized_endpoint_uuids: set[str] = set()
    provisional_hyperedge_uuids = (
        _provisional_hyperedge_uuids(response_object.edges, episodes)
        if isinstance(response_object, CombinedExtractionHyperedge)
        else {}
    )
    for edge_data in response_object.edges:
        if not edge_data.fact.strip():
            logger.debug('Skipping edge with empty fact')
            continue

        # Map episode_indices (0-indexed) to episode UUIDs
        edge_episode_uuids = _episode_uuids_for_fact(edge_data, episodes)

        # Use the first attributed episode's timestamp as the reference time
        edge_reference_time = (
            episodes[edge_data.episode_indices[0]].valid_at
            if edge_data.episode_indices and 0 <= edge_data.episode_indices[0] < len(episodes)
            else primary_episode.valid_at
        )
        provisional_hyperedge_uuid = (
            provisional_hyperedge_uuids.get(_fact_group_key(edge_data, edge_episode_uuids))
            if isinstance(edge_data, CombinedFactHyperedge)
            else None
        )

        endpoint_pairs = _iter_fact_endpoint_pairs(edge_data)
        fact_edges: list[EntityEdge] = []
        for pair in endpoint_pairs:
            source_name = pair.source_entity_name.strip()
            target_name = pair.target_entity_name.strip()
            if not source_name or not target_name:
                logger.debug('Skipping edge endpoint with empty source or target name')
                continue

            if (
                allowed_edge_type_names is not None
                and pair.relation_type not in allowed_edge_type_names
            ):
                logger.debug(
                    'Skipping edge with relation type "%s" not in strict edge ontology',
                    pair.relation_type,
                )
                continue

            source_key = _normalize_string_exact(source_name)
            target_key = _normalize_string_exact(target_name)
            if source_key in excluded_entity_names or target_key in excluded_entity_names:
                logger.debug('Skipping edge with excluded entity endpoint')
                continue

            # Resolve endpoints; materialize any dangling name instead of dropping the edge
            source_node = name_to_node.get(source_key)
            if source_node is None:
                before = len(extracted_nodes)
                source_node = materialize_dangling_endpoint_node(
                    source_name,
                    group_id=primary_episode.group_id,
                    name_to_node=name_to_node,
                    nodes=extracted_nodes,
                )
                if len(extracted_nodes) > before:
                    materialized_endpoint_uuids.add(source_node.uuid)

            target_node = name_to_node.get(target_key)
            if target_node is None:
                before = len(extracted_nodes)
                target_node = materialize_dangling_endpoint_node(
                    target_name,
                    group_id=primary_episode.group_id,
                    name_to_node=name_to_node,
                    nodes=extracted_nodes,
                )
                if len(extracted_nodes) > before:
                    materialized_endpoint_uuids.add(target_node.uuid)

            fact_edges.append(
                EntityEdge(
                    source_node_uuid=source_node.uuid,
                    target_node_uuid=target_node.uuid,
                    name=pair.relation_type,
                    group_id=primary_episode.group_id,
                    fact=edge_data.fact,
                    episodes=edge_episode_uuids,
                    created_at=utc_now(),
                    reference_time=edge_reference_time,
                    hyperedge_uuid=provisional_hyperedge_uuid,
                )
            )

        extracted_edges.extend(fact_edges)

    untag_invalid_groups(extracted_edges, set(provisional_hyperedge_uuids.values()))

    # --- Extract timestamps once per fact (hyperedge groups dated once, others 1:1) ---
    if extracted_edges:
        hyperedge_groups = group_hyperedges(extracted_edges)
        # One representative per fact: each hyperedge's first member, then every untagged
        # (non-hyperedge) edge. The batch dates each representative; apply then copies a
        # hyperedge's dates onto its members.
        representatives = [group.representative for group in hyperedge_groups] + [
            edge for edge in extracted_edges if not edge.hyperedge_uuid
        ]
        facts_with_ref = [
            {
                'fact': edge.fact,
                'reference_time': (
                    edge.reference_time.isoformat() if edge.reference_time else 'unknown'
                ),
            }
            for edge in representatives
        ]
        try:
            ts_response = await generate_prompt_response(
                llm_client,
                'extract_edges.extract_timestamps_batch',
                prompt_library.extract_edges.extract_timestamps_batch,
                {'facts': facts_with_ref},
                clients=clients,
                response_model=BatchEdgeTimestamps,
                model_size=ModelSize.small,
            )
            batch_result = BatchEdgeTimestamps(**ts_response)
            if len(batch_result.timestamps) != len(representatives):
                logger.warning(
                    'Batch timestamp count mismatch: got %d timestamps for %d facts',
                    len(batch_result.timestamps),
                    len(representatives),
                )
            # Truncating extra rows keeps every zip_longest pair on a real fact;
            # a short batch pads with None so those facts are still stamped.
            batch_rows = batch_result.timestamps[: len(representatives)]
            for edge, timestamps in zip_longest(representatives, batch_rows):
                if timestamps is not None:
                    apply_extracted_timestamps(
                        edge,
                        timestamps.valid_at,
                        timestamps.invalid_at,
                        edge.reference_time,
                    )
            for group in hyperedge_groups:
                group.apply()
        except Exception:
            logger.warning(
                'Failed to extract batch timestamps for %d facts',
                len(representatives),
                exc_info=True,
            )

    # --- Derive node episode attribution from edges and drop orphans ---
    # Each node inherits the episode indices of every edge it participates in.
    # Nodes with no connecting edges are dropped — they have no retrievable facts.
    episode_uuid_to_idx = {ep.uuid: i for i, ep in enumerate(episodes)}
    connected_node_uuids: set[str] = set()
    for edge in extracted_edges:
        connected_node_uuids.add(edge.source_node_uuid)
        connected_node_uuids.add(edge.target_node_uuid)

    orphan_count = sum(1 for n in extracted_nodes if n.uuid not in connected_node_uuids)
    if orphan_count:
        logger.debug(
            'Dropping %d orphan node(s) with no connecting edges',
            orphan_count,
        )
    extracted_nodes = [n for n in extracted_nodes if n.uuid in connected_node_uuids]
    surviving_materialized = len(materialized_endpoint_uuids & connected_node_uuids)
    if surviving_materialized:
        logger.warning(
            'Materialized %d dangling edge endpoint node(s) for group_id=%s',
            surviving_materialized,
            primary_episode.group_id,
        )

    for edge in extracted_edges:
        for node_uuid in (edge.source_node_uuid, edge.target_node_uuid):
            edge_episode_positions = [
                episode_uuid_to_idx[ep_uuid]
                for ep_uuid in edge.episodes
                if ep_uuid in episode_uuid_to_idx
            ]
            existing = node_episode_index_map.get(node_uuid, [])
            merged = sorted(set(existing + edge_episode_positions))
            node_episode_index_map[node_uuid] = merged

    logger.debug(
        f'Combined extraction final: {len(extracted_nodes)} nodes, '
        f'{len(extracted_edges)} edges (from {len(response_object.edges)} raw)'
    )

    return extracted_nodes, extracted_edges, node_episode_index_map
