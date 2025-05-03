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

from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.extract_edge_dates import EdgeDates
from graphiti_core.prompts.invalidate_edges import InvalidatedEdges
from graphiti_core.utils.datetime_utils import ensure_utc

logger = logging.getLogger(__name__)


async def extract_edge_dates(
    llm_client: LLMClient,
    edge: EntityEdge,
    current_episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
) -> tuple[datetime | None, datetime | None]:
    context = {
        'edge_fact': edge.fact,
        'current_episode': current_episode.content,
        'previous_episodes': [ep.content for ep in previous_episodes],
        'reference_timestamp': current_episode.valid_at.isoformat(),
    }
    llm_response = await llm_client.generate_response(
        prompt_library.extract_edge_dates.v1(context), response_model=EdgeDates
    )

    valid_at = llm_response.get('valid_at')
    invalid_at = llm_response.get('invalid_at')

    valid_at_datetime = None
    invalid_at_datetime = None

    if valid_at:
        try:
            valid_at_datetime = ensure_utc(datetime.fromisoformat(valid_at.replace('Z', '+00:00')))
        except ValueError as e:
            logger.warning(f'WARNING: Error parsing valid_at date: {e}. Input: {valid_at}')

    if invalid_at:
        try:
            invalid_at_datetime = ensure_utc(
                datetime.fromisoformat(invalid_at.replace('Z', '+00:00'))
            )
        except ValueError as e:
            logger.warning(f'WARNING: Error parsing invalid_at date: {e}. Input: {invalid_at}')

    return valid_at_datetime, invalid_at_datetime


async def get_edge_contradictions(
    llm_client: LLMClient, new_edge: EntityEdge, existing_edges: list[EntityEdge]
) -> list[EntityEdge]:
    start = time()

    new_edge_context = {'fact': new_edge.fact}
    existing_edge_context = [
        {'id': i, 'fact': existing_edge.fact} for i, existing_edge in enumerate(existing_edges)
    ]

    context = {'new_edge': new_edge_context, 'existing_edges': existing_edge_context}

    llm_response = await llm_client.generate_response(
        prompt_library.invalidate_edges.v2(context),
        response_model=InvalidatedEdges,
        model_size=ModelSize.small,
    )

    contradicted_facts: list[int] = llm_response.get('contradicted_facts', [])

    contradicted_edges: list[EntityEdge] = [existing_edges[i] for i in contradicted_facts]

    end = time()
    logger.debug(
        f'Found invalidated edge candidates from {new_edge.fact}, in {(end - start) * 1000} ms'
    )

    return contradicted_edges
