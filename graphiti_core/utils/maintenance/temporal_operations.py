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
from typing import List

from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.prompts import prompt_library

logger = logging.getLogger(__name__)

NodeEdgeNodeTriplet = tuple[EntityNode, EntityEdge, EntityNode]


def extract_node_and_edge_triplets(
    edges: list[EntityEdge], nodes: list[EntityNode]
) -> list[NodeEdgeNodeTriplet]:
    return [extract_node_edge_node_triplet(edge, nodes) for edge in edges]


def extract_node_edge_node_triplet(
    edge: EntityEdge, nodes: list[EntityNode]
) -> NodeEdgeNodeTriplet:
    source_node = next((node for node in nodes if node.uuid == edge.source_node_uuid), None)
    target_node = next((node for node in nodes if node.uuid == edge.target_node_uuid), None)
    if not source_node or not target_node:
        raise ValueError(f'Source or target node not found for edge {edge.uuid}')
    return (source_node, edge, target_node)


def prepare_edges_for_invalidation(
    existing_edges: list[EntityEdge],
    new_edges: list[EntityEdge],
    nodes: list[EntityNode],
) -> tuple[list[NodeEdgeNodeTriplet], list[NodeEdgeNodeTriplet]]:
    existing_edges_pending_invalidation: list[NodeEdgeNodeTriplet] = []
    new_edges_with_nodes: list[NodeEdgeNodeTriplet] = []

    for edge_list, result_list in [
        (existing_edges, existing_edges_pending_invalidation),
        (new_edges, new_edges_with_nodes),
    ]:
        for edge in edge_list:
            source_node = next((node for node in nodes if node.uuid == edge.source_node_uuid), None)
            target_node = next((node for node in nodes if node.uuid == edge.target_node_uuid), None)

            if source_node and target_node:
                result_list.append((source_node, edge, target_node))

    return existing_edges_pending_invalidation, new_edges_with_nodes


async def invalidate_edges(
    llm_client: LLMClient,
    existing_edges_pending_invalidation: list[NodeEdgeNodeTriplet],
    new_edges: list[NodeEdgeNodeTriplet],
    current_episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
) -> list[EntityEdge]:
    invalidated_edges = []  # TODO: this is not yet used?

    context = prepare_invalidation_context(
        existing_edges_pending_invalidation,
        new_edges,
        current_episode,
        previous_episodes,
    )
    llm_response = await llm_client.generate_response(prompt_library.invalidate_edges.v1(context))

    edges_to_invalidate = llm_response.get('invalidated_edges', [])
    invalidated_edges = process_edge_invalidation_llm_response(
        edges_to_invalidate, existing_edges_pending_invalidation
    )

    return invalidated_edges


def extract_date_strings_from_edge(edge: EntityEdge) -> str:
    start = edge.valid_at
    end = edge.invalid_at
    date_string = f'Start Date: {start.isoformat()}' if start else ''
    if end:
        date_string += f' (End Date: {end.isoformat()})'
    return date_string


def prepare_invalidation_context(
    existing_edges: list[NodeEdgeNodeTriplet],
    new_edges: list[NodeEdgeNodeTriplet],
    current_episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
) -> dict:
    return {
        'existing_edges': [
            f'{edge.uuid} | {source_node.name} - {edge.name} - {target_node.name} (Fact: {edge.fact}) {extract_date_strings_from_edge(edge)}'
            for source_node, edge, target_node in sorted(
                existing_edges, key=lambda x: (x[1].created_at), reverse=True
            )
        ],
        'new_edges': [
            f'{edge.uuid} | {source_node.name} - {edge.name} - {target_node.name} (Fact: {edge.fact}) {extract_date_strings_from_edge(edge)}'
            for source_node, edge, target_node in sorted(
                new_edges, key=lambda x: (x[1].created_at), reverse=True
            )
        ],
        'current_episode': current_episode.content,
        'previous_episodes': [episode.content for episode in previous_episodes],
    }


def process_edge_invalidation_llm_response(
    edges_to_invalidate: List[dict], existing_edges: List[NodeEdgeNodeTriplet]
) -> List[EntityEdge]:
    invalidated_edges = []
    for edge_to_invalidate in edges_to_invalidate:
        edge_uuid = edge_to_invalidate['edge_uuid']
        edge_to_update = next(
            (edge for _, edge, _ in existing_edges if edge.uuid == edge_uuid),
            None,
        )
        if edge_to_update:
            edge_to_update.expired_at = datetime.now()
            edge_to_update.fact = edge_to_invalidate['fact']
            invalidated_edges.append(edge_to_update)
            logger.info(
                f"Invalidated edge: {edge_to_update.name} (UUID: {edge_to_update.uuid}). Updated Fact: {edge_to_invalidate['fact']}"
            )
    return invalidated_edges


async def extract_edge_dates(
    llm_client: LLMClient,
    edge: EntityEdge,
    current_episode: EpisodicNode,
    previous_episodes: List[EpisodicNode],
) -> tuple[datetime | None, datetime | None]:
    context = {
        'edge_name': edge.name,
        'edge_fact': edge.fact,
        'current_episode': current_episode.content,
        'previous_episodes': [ep.content for ep in previous_episodes],
        'reference_timestamp': current_episode.valid_at.isoformat(),
    }
    llm_response = await llm_client.generate_response(prompt_library.extract_edge_dates.v1(context))

    valid_at = llm_response.get('valid_at')
    invalid_at = llm_response.get('invalid_at')
    explanation = llm_response.get('explanation', '')

    valid_at_datetime = None
    invalid_at_datetime = None

    if valid_at and valid_at != '':
        try:
            valid_at_datetime = datetime.fromisoformat(valid_at.replace('Z', '+00:00'))
        except ValueError as e:
            logger.error(f'Error parsing valid_at date: {e}. Input: {valid_at}')

    if invalid_at and invalid_at != '':
        try:
            invalid_at_datetime = datetime.fromisoformat(invalid_at.replace('Z', '+00:00'))
        except ValueError as e:
            logger.error(f'Error parsing invalid_at date: {e}. Input: {invalid_at}')

    logger.info(f'Edge date extraction explanation: {explanation}')

    return valid_at_datetime, invalid_at_datetime
