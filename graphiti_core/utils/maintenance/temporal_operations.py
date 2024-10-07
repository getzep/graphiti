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
from typing import List

from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EpisodicNode
from graphiti_core.prompts import prompt_library

logger = logging.getLogger(__name__)


async def extract_edge_dates(
    llm_client: LLMClient,
    edge: EntityEdge,
    current_episode: EpisodicNode,
    previous_episodes: List[EpisodicNode],
) -> tuple[datetime | None, datetime | None]:
    context = {
        'edge_fact': edge.fact,
        'current_episode': current_episode.content,
        'previous_episodes': [ep.content for ep in previous_episodes],
        'reference_timestamp': current_episode.valid_at.isoformat(),
    }
    llm_response = await llm_client.generate_response(prompt_library.extract_edge_dates.v1(context))

    valid_at = llm_response.get('valid_at')
    invalid_at = llm_response.get('invalid_at')

    valid_at_datetime = None
    invalid_at_datetime = None

    if valid_at:
        try:
            valid_at_datetime = datetime.fromisoformat(valid_at.replace('Z', '+00:00'))
        except ValueError as e:
            logger.error(f'Error parsing valid_at date: {e}. Input: {valid_at}')

    if invalid_at:
        try:
            invalid_at_datetime = datetime.fromisoformat(invalid_at.replace('Z', '+00:00'))
        except ValueError as e:
            logger.error(f'Error parsing invalid_at date: {e}. Input: {invalid_at}')

    return valid_at_datetime, invalid_at_datetime


async def get_edge_contradictions(
    llm_client: LLMClient, new_edge: EntityEdge, existing_edges: list[EntityEdge]
) -> list[EntityEdge]:
    start = time()
    existing_edge_map = {edge.uuid: edge for edge in existing_edges}

    new_edge_context = {'uuid': new_edge.uuid, 'name': new_edge.name, 'fact': new_edge.fact}
    existing_edge_context = [
        {'uuid': existing_edge.uuid, 'name': existing_edge.name, 'fact': existing_edge.fact}
        for existing_edge in existing_edges
    ]

    context = {'new_edge': new_edge_context, 'existing_edges': existing_edge_context}

    llm_response = await llm_client.generate_response(prompt_library.invalidate_edges.v2(context))

    contradicted_edge_data = llm_response.get('invalidated_edges', [])

    contradicted_edges: list[EntityEdge] = []
    for edge_data in contradicted_edge_data:
        if edge_data['uuid'] in existing_edge_map:
            contradicted_edge = existing_edge_map[edge_data['uuid']]
            contradicted_edge.fact = edge_data['fact']
            contradicted_edges.append(contradicted_edge)

    end = time()
    logger.info(
        f'Found invalidated edge candidates from {new_edge.fact}, in {(end - start) * 1000} ms'
    )

    return contradicted_edges
