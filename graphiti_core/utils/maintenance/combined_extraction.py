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

from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.extract_edges import BatchEdgeTimestamps
from graphiti_core.prompts.extract_nodes_and_edges import CombinedExtraction
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact
from graphiti_core.utils.maintenance.node_operations import (
    _build_entity_types_context,
    _collapse_exact_duplicate_extracted_nodes,
)
from graphiti_core.utils.text_utils import concatenate_episodes

logger = logging.getLogger(__name__)


async def extract_nodes_and_edges(
    clients: GraphitiClients,
    episode: EpisodicNode | list[EpisodicNode],
    previous_episodes: list[EpisodicNode],
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    edge_type_map: dict[tuple[str, str], list[str]] | None = None,
    edge_types: dict[str, type[BaseModel]] | None = None,
    custom_extraction_instructions: str | None = None,
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
    custom_extraction_instructions : str | None
        Additional extraction instructions.

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
    entity_types_context = _build_entity_types_context(entity_types)

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

    # Build context for the combined prompt
    context = {
        'episode_content': concatenate_episodes(episodes),
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
    }

    # Single LLM call for combined extraction
    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes_and_edges.extract_message(context),
        response_model=CombinedExtraction,
        group_id=primary_episode.group_id,
        prompt_name='extract_nodes_and_edges.extract_message',
    )
    response_object = CombinedExtraction(**llm_response)

    end = time()
    logger.debug(
        f'Combined extraction: {len(response_object.extracted_entities)} entities, '
        f'{len(response_object.edges)} edges in {(end - start) * 1000:.0f} ms'
    )

    # --- Process nodes ---

    # Filter empty names
    filtered_entities = [e for e in response_object.extracted_entities if e.name.strip()]

    # Convert CombinedEntity objects to EntityNode objects (no episode attribution yet —
    # that is derived from edges below).
    extracted_nodes: list[EntityNode] = []
    for entity in filtered_entities:
        type_id = entity.entity_type_id
        if 0 <= type_id < len(entity_types_context):
            entity_type_name = entity_types_context[type_id].get('entity_type_name')
        else:
            entity_type_name = 'Entity'

        if excluded_entity_types and entity_type_name in excluded_entity_types:
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
    for edge_data in response_object.edges:
        # Validate source and target exist in extracted nodes (case-insensitive)
        source_node = name_to_node.get(_normalize_string_exact(edge_data.source_entity_name))
        target_node = name_to_node.get(_normalize_string_exact(edge_data.target_entity_name))

        if source_node is None:
            logger.debug(
                f'Skipping edge: source "{edge_data.source_entity_name}" not in extracted nodes'
            )
            continue
        if target_node is None:
            logger.debug(
                f'Skipping edge: target "{edge_data.target_entity_name}" not in extracted nodes'
            )
            continue

        if not edge_data.fact.strip():
            logger.debug('Skipping edge with empty fact')
            continue

        # Map episode_indices (0-indexed) to episode UUIDs
        edge_episode_uuids: list[str] = []
        for idx in edge_data.episode_indices:
            if 0 <= idx < len(episodes):
                edge_episode_uuids.append(episodes[idx].uuid)
        if not edge_episode_uuids:
            edge_episode_uuids = [ep.uuid for ep in episodes]

        # Use the first attributed episode's timestamp as the reference time
        edge_reference_time = (
            episodes[edge_data.episode_indices[0]].valid_at
            if edge_data.episode_indices and 0 <= edge_data.episode_indices[0] < len(episodes)
            else primary_episode.valid_at
        )

        edge = EntityEdge(
            source_node_uuid=source_node.uuid,
            target_node_uuid=target_node.uuid,
            name=edge_data.relation_type,
            group_id=primary_episode.group_id,
            fact=edge_data.fact,
            episodes=edge_episode_uuids,
            created_at=utc_now(),
            reference_time=edge_reference_time,
        )
        extracted_edges.append(edge)

    # --- Extract timestamps for all edges in a single batch LLM call ---
    if extracted_edges:
        facts_with_ref = [
            {
                'fact': edge.fact,
                'reference_time': (
                    edge.reference_time.isoformat() if edge.reference_time else 'unknown'
                ),
            }
            for edge in extracted_edges
        ]
        try:
            ts_response = await llm_client.generate_response(
                prompt_library.extract_edges.extract_timestamps_batch({'facts': facts_with_ref}),
                response_model=BatchEdgeTimestamps,
                model_size=ModelSize.small,
                prompt_name='extract_edges.extract_timestamps_batch',
            )
            batch_timestamps = BatchEdgeTimestamps(**ts_response)
            if len(batch_timestamps.timestamps) != len(extracted_edges):
                logger.warning(
                    'Batch timestamp count mismatch: got %d timestamps for %d edges',
                    len(batch_timestamps.timestamps),
                    len(extracted_edges),
                )
            for edge, ts in zip(extracted_edges, batch_timestamps.timestamps, strict=False):
                if ts.valid_at:
                    try:
                        edge.valid_at = ensure_utc(
                            datetime.fromisoformat(ts.valid_at.replace('Z', '+00:00'))
                        )
                    except ValueError:
                        logger.debug(f'Error parsing valid_at: {ts.valid_at}')
                if ts.invalid_at:
                    try:
                        edge.invalid_at = ensure_utc(
                            datetime.fromisoformat(ts.invalid_at.replace('Z', '+00:00'))
                        )
                    except ValueError:
                        logger.debug(f'Error parsing invalid_at: {ts.invalid_at}')
        except Exception:
            logger.warning(
                'Failed to extract batch timestamps for %d edges',
                len(extracted_edges),
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
