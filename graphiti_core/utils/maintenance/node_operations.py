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
from collections.abc import Awaitable, Callable
from time import time
from typing import Any

from pydantic import BaseModel

from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import (
    EntityNode,
    EpisodeType,
    EpisodicNode,
    create_entity_node_embeddings,
)
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_nodes import NodeDuplicate, NodeResolutions
from graphiti_core.prompts.extract_nodes import (
    ExtractedEntities,
    ExtractedEntity,
    SummarizedEntities,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import node_similarity_search
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.dedup_helpers import (
    DedupCandidateIndexes,
    DedupResolutionState,
    _build_candidate_indexes,
    _normalize_string_exact,
    _promote_resolved_node,
    _resolve_with_similarity,
)
from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS, truncate_at_sentence

logger = logging.getLogger(__name__)

# Maximum number of nodes to summarize in a single LLM call
MAX_NODES = 30
NODE_DEDUP_CANDIDATE_LIMIT = 15
NODE_DEDUP_COSINE_MIN_SCORE = 0.6

NodeSummaryFilter = Callable[[EntityNode], Awaitable[bool]]


async def extract_nodes(
    clients: GraphitiClients,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    custom_extraction_instructions: str | None = None,
) -> list[EntityNode]:
    """Extract entity nodes from an episode."""
    start = time()
    llm_client = clients.llm_client

    # Build entity types context
    entity_types_context = _build_entity_types_context(entity_types)

    # Build base context
    context = {
        'episode_content': episode.content,
        'episode_timestamp': episode.valid_at.isoformat(),
        'previous_episodes': [ep.content for ep in previous_episodes],
        'custom_extraction_instructions': custom_extraction_instructions or '',
        'entity_types': entity_types_context,
        'source_description': episode.source_description,
    }

    # Extract entities
    extracted_entities = await _extract_nodes_single(llm_client, episode, context)

    # Filter empty names
    filtered_entities = [e for e in extracted_entities if e.name.strip()]

    end = time()
    logger.debug(f'Extracted {len(filtered_entities)} entities in {(end - start) * 1000:.0f} ms')

    # Convert to EntityNode objects
    extracted_nodes = _create_entity_nodes(
        filtered_entities, entity_types_context, excluded_entity_types, episode
    )
    extracted_nodes = _collapse_exact_duplicate_extracted_nodes(extracted_nodes)

    logger.debug(f'Extracted nodes: {[n.uuid for n in extracted_nodes]}')
    return extracted_nodes


def _build_entity_types_context(
    entity_types: dict[str, type[BaseModel]] | None,
) -> list[dict]:
    """Build entity types context with ID mappings."""
    entity_types_context = [
        {
            'entity_type_id': 0,
            'entity_type_name': 'Entity',
            'entity_type_description': (
                'A specific, identifiable entity that does not fit any of the other listed '
                'types. Must still be a concrete, meaningful thing — specific enough to be '
                'uniquely identifiable. GOOD: a named entity not covered by the other types. '
                'BAD: "luck", "ideas", "tomorrow", "things", "them", "everybody", '
                '"a sense of wonder", "great times". '
                'When in doubt, do not extract the entity.'
            ),
        }
    ]

    if entity_types is not None:
        entity_types_context += [
            {
                'entity_type_id': i + 1,
                'entity_type_name': type_name,
                'entity_type_description': type_model.__doc__,
            }
            for i, (type_name, type_model) in enumerate(entity_types.items())
        ]

    return entity_types_context


def _get_entity_type_description(
    labels: list[str], entity_types: dict[str, type[BaseModel]] | None
) -> str:
    type_name = next((item for item in labels if item != 'Entity'), '')
    type_model = entity_types.get(type_name) if entity_types is not None else None
    return (type_model.__doc__ if type_model is not None else None) or 'Default Entity Type'


async def _extract_nodes_single(
    llm_client: LLMClient,
    episode: EpisodicNode,
    context: dict,
) -> list[ExtractedEntity]:
    """Extract entities using a single LLM call."""
    llm_response = await _call_extraction_llm(llm_client, episode, context)
    response_object = ExtractedEntities(**llm_response)
    return response_object.extracted_entities


async def _call_extraction_llm(
    llm_client: LLMClient,
    episode: EpisodicNode,
    context: dict,
) -> dict:
    """Call the appropriate extraction prompt based on episode type."""
    if episode.source == EpisodeType.message:
        prompt = prompt_library.extract_nodes.extract_message(context)
        prompt_name = 'extract_nodes.extract_message'
    elif episode.source == EpisodeType.text:
        prompt = prompt_library.extract_nodes.extract_text(context)
        prompt_name = 'extract_nodes.extract_text'
    elif episode.source == EpisodeType.json:
        prompt = prompt_library.extract_nodes.extract_json(context)
        prompt_name = 'extract_nodes.extract_json'
    else:
        # Fallback to text extraction
        prompt = prompt_library.extract_nodes.extract_text(context)
        prompt_name = 'extract_nodes.extract_text'

    return await llm_client.generate_response(
        prompt,
        response_model=ExtractedEntities,
        group_id=episode.group_id,
        prompt_name=prompt_name,
    )


def _create_entity_nodes(
    extracted_entities: list[ExtractedEntity],
    entity_types_context: list[dict],
    excluded_entity_types: list[str] | None,
    episode: EpisodicNode,
) -> list[EntityNode]:
    """Convert ExtractedEntity objects to EntityNode objects."""
    extracted_nodes = []

    for extracted_entity in extracted_entities:
        type_id = extracted_entity.entity_type_id
        if 0 <= type_id < len(entity_types_context):
            entity_type_name = entity_types_context[type_id].get('entity_type_name')
        else:
            entity_type_name = 'Entity'

        # Check if this entity type should be excluded
        if excluded_entity_types and entity_type_name in excluded_entity_types:
            logger.debug(f'Excluding entity of type "{entity_type_name}"')
            continue

        labels: list[str] = list({'Entity', str(entity_type_name)})

        new_node = EntityNode(
            name=extracted_entity.name,
            group_id=episode.group_id,
            labels=labels,
            summary='',
            created_at=utc_now(),
        )
        extracted_nodes.append(new_node)
        logger.debug(f'Created new node: {new_node.uuid}')

    return extracted_nodes


def _collapse_exact_duplicate_extracted_nodes(
    extracted_nodes: list[EntityNode],
) -> list[EntityNode]:
    """Collapse same-message duplicates with the same normalized name.

    This is intentionally narrow: it only merges exact normalized-name duplicates that the
    extraction prompt should already have emitted once. When duplicates disagree on specificity,
    keep the more specific node (for example, `Person` over bare `Entity`).
    """
    if len(extracted_nodes) < 2:
        return extracted_nodes

    canonical_by_name: dict[str, EntityNode] = {}
    ordered_names: list[str] = []

    for node in extracted_nodes:
        normalized_name = _normalize_string_exact(node.name)
        existing = canonical_by_name.get(normalized_name)
        if existing is None:
            canonical_by_name[normalized_name] = node
            ordered_names.append(normalized_name)
            continue

        existing_specific_labels = {label for label in existing.labels if label != 'Entity'}
        node_specific_labels = {label for label in node.labels if label != 'Entity'}
        if len(node_specific_labels) > len(existing_specific_labels) or (
            len(node_specific_labels) == len(existing_specific_labels)
            and len(node.name.strip()) > len(existing.name.strip())
        ):
            canonical_by_name[normalized_name] = node

    return [canonical_by_name[name] for name in ordered_names]


def _merge_candidate_nodes(
    candidate_nodes: list[EntityNode],
    existing_nodes_override: list[EntityNode] | None,
) -> list[EntityNode]:
    """Deduplicate candidate nodes while preserving search order and overrides."""
    merged_candidates = list(candidate_nodes)
    if existing_nodes_override is not None:
        merged_candidates.extend(existing_nodes_override)

    seen_candidate_uuids: set[str] = set()
    ordered_candidates: list[EntityNode] = []
    for candidate in merged_candidates:
        if candidate.uuid in seen_candidate_uuids:
            continue
        seen_candidate_uuids.add(candidate.uuid)
        ordered_candidates.append(candidate)

    return ordered_candidates


async def _collect_candidate_nodes(
    clients: GraphitiClients,
    extracted_nodes: list[EntityNode],
    existing_nodes_override: list[EntityNode] | None,
) -> list[list[EntityNode]]:
    """Search per extracted name and return ordered candidates for each extracted node."""
    search_results = await _semantic_candidate_search(clients, extracted_nodes)

    return [_merge_candidate_nodes(result, existing_nodes_override) for result in search_results]


async def _semantic_candidate_search(
    clients: GraphitiClients,
    extracted_nodes: list[EntityNode],
) -> list[list[EntityNode]]:
    """Run direct cosine similarity search per extracted node without reranking."""
    if not extracted_nodes:
        return []

    queries = [node.name.replace('\n', ' ') for node in extracted_nodes]
    try:
        query_vectors = await clients.embedder.create_batch(queries)
    except NotImplementedError:
        query_vectors = list(
            await semaphore_gather(
                *[clients.embedder.create(input_data=[query]) for query in queries]
            )
        )

    return list(
        await semaphore_gather(
            *[
                node_similarity_search(
                    clients.driver,
                    query_vector,
                    SearchFilters(),
                    [node.group_id],
                    NODE_DEDUP_CANDIDATE_LIMIT,
                    NODE_DEDUP_COSINE_MIN_SCORE,
                )
                for node, query_vector in zip(extracted_nodes, query_vectors, strict=True)
            ]
        )
    )


def _commit_resolution(
    state: DedupResolutionState,
    resolved_node: EntityNode | None,
    uuid_map: dict[str, str],
    duplicate_pairs: list[tuple[EntityNode, EntityNode]],
    index: int,
) -> None:
    """Commit a single-node resolution result into the batch-level state."""
    if resolved_node is not None:
        state.resolved_nodes[index] = resolved_node
    state.uuid_map.update(uuid_map)
    state.duplicate_pairs.extend(duplicate_pairs)


async def _resolve_with_llm(
    llm_client: LLMClient,
    extracted_nodes: list[EntityNode],
    indexes: DedupCandidateIndexes,
    state: DedupResolutionState,
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
    entity_types: dict[str, type[BaseModel]] | None,
) -> None:
    """Escalate unresolved nodes to the dedupe prompt so the LLM can select or reject duplicates.

    The guardrails below defensively ignore malformed or duplicate LLM responses so the
    ingestion workflow remains deterministic even when the model misbehaves.
    """
    if not state.unresolved_indices:
        return

    entity_types_dict: dict[str, type[BaseModel]] = entity_types if entity_types is not None else {}

    llm_extracted_nodes = [extracted_nodes[i] for i in state.unresolved_indices]

    extracted_nodes_context = [
        {
            'id': i,
            'name': node.name,
            'entity_type': node.labels,
            'entity_type_description': _get_entity_type_description(node.labels, entity_types_dict),
        }
        for i, node in enumerate(llm_extracted_nodes)
    ]

    sent_ids = [ctx['id'] for ctx in extracted_nodes_context]
    logger.debug(
        'Sending %d entities to LLM for deduplication with IDs 0-%d (actual IDs sent: %s)',
        len(llm_extracted_nodes),
        len(llm_extracted_nodes) - 1,
        sent_ids if len(sent_ids) < 20 else f'{sent_ids[:10]}...{sent_ids[-10:]}',
    )
    if llm_extracted_nodes:
        sample_size = min(3, len(extracted_nodes_context))
        logger.debug(
            'First %d entity IDs: %s',
            sample_size,
            [ctx['id'] for ctx in extracted_nodes_context[:sample_size]],
        )
        if len(extracted_nodes_context) > 3:
            logger.debug(
                'Last %d entity IDs: %s',
                sample_size,
                [ctx['id'] for ctx in extracted_nodes_context[-sample_size:]],
            )

    existing_nodes_context = [
        {
            **candidate.attributes,
            'candidate_id': i,
            'name': candidate.name,
            'entity_types': candidate.labels,
            'summary': candidate.summary[:120] if candidate.summary else '',
        }
        for i, candidate in enumerate(indexes.existing_nodes)
    ]

    # Build candidate_id -> node mapping for resolving duplicates by ID
    candidates_by_id: dict[int, EntityNode] = {
        i: node for i, node in enumerate(indexes.existing_nodes)
    }

    context = {
        'extracted_nodes': extracted_nodes_context,
        'existing_nodes': existing_nodes_context,
        'episode_content': episode.content if episode is not None else '',
        'previous_episodes': (
            [ep.content for ep in previous_episodes] if previous_episodes is not None else []
        ),
    }

    llm_response = await llm_client.generate_response(
        prompt_library.dedupe_nodes.nodes(context),
        response_model=NodeResolutions,
        prompt_name='dedupe_nodes.nodes',
    )

    node_resolutions: list[NodeDuplicate] = NodeResolutions(**llm_response).entity_resolutions

    valid_relative_range = range(len(state.unresolved_indices))
    processed_relative_ids: set[int] = set()

    received_ids = {r.id for r in node_resolutions}
    expected_ids = set(valid_relative_range)
    missing_ids = expected_ids - received_ids
    extra_ids = received_ids - expected_ids

    logger.debug(
        'Received %d resolutions for %d entities',
        len(node_resolutions),
        len(state.unresolved_indices),
    )

    if missing_ids:
        logger.warning('LLM did not return resolutions for IDs: %s', sorted(missing_ids))

    if extra_ids:
        logger.warning(
            'LLM returned invalid IDs outside valid range 0-%d: %s (all returned IDs: %s)',
            len(state.unresolved_indices) - 1,
            sorted(extra_ids),
            sorted(received_ids),
        )

    for resolution in node_resolutions:
        relative_id: int = resolution.id
        duplicate_candidate_id: int = resolution.duplicate_candidate_id

        if relative_id not in valid_relative_range:
            logger.warning(
                'Skipping invalid LLM dedupe id %d (valid range: 0-%d, received %d resolutions)',
                relative_id,
                len(state.unresolved_indices) - 1,
                len(node_resolutions),
            )
            continue

        if relative_id in processed_relative_ids:
            logger.warning('Duplicate LLM dedupe id %s received; ignoring.', relative_id)
            continue
        processed_relative_ids.add(relative_id)

        original_index = state.unresolved_indices[relative_id]
        extracted_node = extracted_nodes[original_index]

        resolved_node: EntityNode
        if duplicate_candidate_id < 0:
            resolved_node = extracted_node
        elif duplicate_candidate_id in candidates_by_id:
            resolved_node = _promote_resolved_node(
                extracted_node, candidates_by_id[duplicate_candidate_id]
            )
        else:
            logger.warning(
                'Invalid duplicate_candidate_id %d for extracted node %s; treating as no duplicate.',
                duplicate_candidate_id,
                extracted_node.uuid,
            )
            resolved_node = extracted_node

        state.resolved_nodes[original_index] = resolved_node
        state.uuid_map[extracted_node.uuid] = resolved_node.uuid
        if resolved_node.uuid != extracted_node.uuid:
            state.duplicate_pairs.append((extracted_node, resolved_node))


async def resolve_extracted_nodes(
    clients: GraphitiClients,
    extracted_nodes: list[EntityNode],
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_types: dict[str, type[BaseModel]] | None = None,
    existing_nodes_override: list[EntityNode] | None = None,
) -> tuple[list[EntityNode], dict[str, str], list[tuple[EntityNode, EntityNode]]]:
    """Resolve nodes with semantic retrieval first, then deterministic and LLM dedup."""
    llm_client = clients.llm_client
    candidate_nodes_by_extracted = await _collect_candidate_nodes(
        clients,
        extracted_nodes,
        existing_nodes_override,
    )

    state = DedupResolutionState(
        resolved_nodes=[None] * len(extracted_nodes),
        uuid_map={},
        unresolved_indices=[],
    )

    for idx, (node, candidates) in enumerate(
        zip(extracted_nodes, candidate_nodes_by_extracted, strict=True)
    ):
        if not candidates:
            continue

        indexes = _build_candidate_indexes(candidates)
        local_state = DedupResolutionState(
            resolved_nodes=[None], uuid_map={}, unresolved_indices=[]
        )
        _resolve_with_similarity([node], indexes, local_state)
        if local_state.resolved_nodes[0] is not None:
            _commit_resolution(
                state,
                local_state.resolved_nodes[0],
                local_state.uuid_map,
                local_state.duplicate_pairs,
                idx,
            )
            continue

        state.unresolved_indices.append(idx)

    if state.unresolved_indices:
        llm_candidate_nodes = _merge_candidate_nodes(
            [
                candidate
                for idx in state.unresolved_indices
                for candidate in candidate_nodes_by_extracted[idx]
            ],
            None,
        )
        await _resolve_with_llm(
            llm_client,
            extracted_nodes,
            _build_candidate_indexes(llm_candidate_nodes),
            state,
            episode,
            previous_episodes,
            entity_types,
        )

    if not state.unresolved_indices and not any(candidate_nodes_by_extracted):
        logger.debug('No semantic dedup candidates found; keeping all extracted nodes as new')

    for idx, node in enumerate(extracted_nodes):
        if state.resolved_nodes[idx] is None:
            state.resolved_nodes[idx] = node
            state.uuid_map[node.uuid] = node.uuid

    logger.debug(
        'Resolved nodes: %s',
        [node.uuid for node in state.resolved_nodes if node is not None],
    )

    return (
        [node for node in state.resolved_nodes if node is not None],
        state.uuid_map,
        state.duplicate_pairs,
    )


def _build_edges_by_node(edges: list[EntityEdge] | None) -> dict[str, list[EntityEdge]]:
    """Build a dictionary mapping node UUIDs to their connected edges."""
    edges_by_node: dict[str, list[EntityEdge]] = {}
    if not edges:
        return edges_by_node
    for edge in edges:
        if edge.source_node_uuid not in edges_by_node:
            edges_by_node[edge.source_node_uuid] = []
        if edge.target_node_uuid not in edges_by_node:
            edges_by_node[edge.target_node_uuid] = []
        edges_by_node[edge.source_node_uuid].append(edge)
        edges_by_node[edge.target_node_uuid].append(edge)
    return edges_by_node


async def extract_attributes_from_nodes(
    clients: GraphitiClients,
    nodes: list[EntityNode],
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_types: dict[str, type[BaseModel]] | None = None,
    should_summarize_node: NodeSummaryFilter | None = None,
    edges: list[EntityEdge] | None = None,
    skip_fact_appending: bool = False,
) -> list[EntityNode]:
    llm_client = clients.llm_client
    embedder = clients.embedder

    # Pre-build edges lookup for O(E + N) instead of O(N * E)
    edges_by_node = _build_edges_by_node(edges)

    # Extract attributes in parallel (per-entity calls)
    attribute_results: list[dict[str, Any]] = await semaphore_gather(
        *[
            _extract_entity_attributes(
                llm_client,
                node,
                episode,
                previous_episodes,
                (
                    entity_types.get(next((item for item in node.labels if item != 'Entity'), ''))
                    if entity_types is not None
                    else None
                ),
            )
            for node in nodes
        ]
    )

    # Apply attributes to nodes
    for node, attributes in zip(nodes, attribute_results, strict=True):
        node.attributes.update(attributes)

    # Extract summaries in batch
    await _extract_entity_summaries_batch(
        llm_client,
        nodes,
        episode,
        previous_episodes,
        should_summarize_node,
        edges_by_node,
        skip_fact_appending=skip_fact_appending,
    )

    await create_entity_node_embeddings(embedder, nodes)

    return nodes


async def _extract_entity_attributes(
    llm_client: LLMClient,
    node: EntityNode,
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
    entity_type: type[BaseModel] | None,
) -> dict[str, Any]:
    if entity_type is None or len(entity_type.model_fields) == 0:
        return {}

    attributes_context = _build_episode_context(
        # should not include summary
        node_data={
            'name': node.name,
            'entity_types': node.labels,
            'attributes': node.attributes,
        },
        episode=episode,
        previous_episodes=previous_episodes,
    )

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.extract_attributes(attributes_context),
        response_model=entity_type,
        model_size=ModelSize.small,
        group_id=node.group_id,
        prompt_name='extract_nodes.extract_attributes',
    )

    # validate response
    entity_type(**llm_response)

    return llm_response


async def _extract_entity_summaries_batch(
    llm_client: LLMClient,
    nodes: list[EntityNode],
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
    should_summarize_node: NodeSummaryFilter | None,
    edges_by_node: dict[str, list[EntityEdge]],
    *,
    skip_fact_appending: bool = False,
) -> None:
    """Extract summaries for multiple entities in batched LLM calls.

    When skip_fact_appending is False (default), nodes with short summaries get edge
    facts appended directly without an LLM call.  Nodes needing summarization are
    partitioned into flights of MAX_NODES and processed with separate LLM calls.

    When skip_fact_appending is True, the raw fact-append shortcut is bypassed and all
    nodes are routed through LLM summarization using an episode-based prompt that
    matches the async graph summary worker.
    """
    # Determine which nodes need LLM summarization vs direct edge fact appending
    nodes_needing_llm: list[EntityNode] = []

    for node in nodes:
        # Check if node should be summarized at all
        if should_summarize_node is not None and not await should_summarize_node(node):
            continue

        if skip_fact_appending:
            # Always route through LLM — no raw fact concatenation.
            if episode is not None or node.summary:
                nodes_needing_llm.append(node)
            continue

        node_edges = edges_by_node.get(node.uuid, [])

        # Build summary with edge facts appended
        summary_with_edges = node.summary
        if node_edges:
            edge_facts = '\n'.join(edge.fact for edge in node_edges if edge.fact)
            summary_with_edges = f'{summary_with_edges}\n{edge_facts}'.strip()

        # If summary is close to the persisted limit, use it directly (append edge facts, no LLM call)
        if summary_with_edges and len(summary_with_edges) <= MAX_SUMMARY_CHARS * 2:
            node.summary = summary_with_edges
            continue

        # Skip if no summary content and no episode to generate from
        if not summary_with_edges and episode is None:
            continue

        # This node needs LLM summarization
        nodes_needing_llm.append(node)

    # If no nodes need LLM summarization, return early
    if not nodes_needing_llm:
        return

    # Partition nodes into flights of MAX_NODES
    node_flights = [
        nodes_needing_llm[i : i + MAX_NODES] for i in range(0, len(nodes_needing_llm), MAX_NODES)
    ]

    # Process flights in parallel
    await semaphore_gather(
        *[
            _process_summary_flight(
                llm_client,
                flight,
                episode,
                previous_episodes,
                use_episode_prompt=skip_fact_appending,
            )
            for flight in node_flights
        ]
    )


async def _process_summary_flight(
    llm_client: LLMClient,
    nodes: list[EntityNode],
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
    *,
    use_episode_prompt: bool = False,
) -> None:
    """Process a single flight of nodes for batch summarization."""
    # Build context for batch summarization
    entities_context = [
        {
            'name': node.name,
            'summary': node.summary,
            'entity_types': node.labels,
            'attributes': node.attributes,
        }
        for node in nodes
    ]

    batch_context = {
        'entities': entities_context,
        'episode_content': episode.content if episode is not None else '',
        'previous_episodes': (
            [ep.content for ep in previous_episodes] if previous_episodes is not None else []
        ),
    }

    # Get group_id from the first node (all nodes in a batch should have same group_id)
    group_id = nodes[0].group_id if nodes else None

    if use_episode_prompt:
        prompt = prompt_library.extract_nodes.extract_entity_summaries_from_episodes(batch_context)
        prompt_name = 'extract_nodes.extract_entity_summaries_from_episodes'
    else:
        prompt = prompt_library.extract_nodes.extract_summaries_batch(batch_context)
        prompt_name = 'extract_nodes.extract_summaries_batch'

    llm_response = await llm_client.generate_response(
        prompt,
        response_model=SummarizedEntities,
        model_size=ModelSize.small,
        group_id=group_id,
        prompt_name=prompt_name,
    )

    # Build case-insensitive name -> nodes mapping (handles duplicates)
    name_to_nodes: dict[str, list[EntityNode]] = {}
    for node in nodes:
        key = node.name.lower()
        if key not in name_to_nodes:
            name_to_nodes[key] = []
        name_to_nodes[key].append(node)

    # Apply summaries from LLM response
    summaries_response = SummarizedEntities(**llm_response)
    for summarized_entity in summaries_response.summaries:
        matching_nodes = name_to_nodes.get(summarized_entity.name.lower(), [])
        if matching_nodes:
            truncated_summary = truncate_at_sentence(summarized_entity.summary, MAX_SUMMARY_CHARS)
            for node in matching_nodes:
                node.summary = truncated_summary
        else:
            logger.warning(
                'LLM returned summary for unknown entity (first 30 chars): %.30s',
                summarized_entity.name,
            )


def _build_episode_context(
    node_data: dict[str, Any],
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
) -> dict[str, Any]:
    return {
        'node': node_data,
        'episode_content': episode.content if episode is not None else '',
        'previous_episodes': (
            [ep.content for ep in previous_episodes] if previous_episodes is not None else []
        ),
    }
