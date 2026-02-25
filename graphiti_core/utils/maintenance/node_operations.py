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
from collections import defaultdict
from collections.abc import Awaitable, Callable
from time import time
from typing import Any

from pydantic import BaseModel, ValidationError

from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import MAX_REFLEXION_ITERATIONS, semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import (
    EntityNode,
    EpisodeType,
    EpisodicNode,
    create_entity_node_embeddings,
)
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_nodes import (
    NodeDuplicate,
    NodeDuplicateWithScores,
    NodeResolutions,
    NodeResolutionsWithScores,
)
from graphiti_core.prompts.extract_nodes import (
    CandidateTypeScore,
    EntityReclassification,
    EntitySummaries,
    EntitySummary,
    EntitySummaryItem,
    EntityValidationResult,
    ExtractedEntities,
    ExtractedEntitiesWithScores,
    ExtractedEntity,
    ExtractedEntityWithScores,
    MissedEntities,
    RefinedEntityTypes,
    ResolvedEntityTypes,
    TopTypeCandidate,
    TypeScore,
)
from graphiti_core.search.search import search
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.dedup_helpers import (
    DedupCandidateIndexes,
    DedupResolutionState,
    _build_candidate_indexes,
    _resolve_with_similarity,
)
from graphiti_core.utils.maintenance.edge_operations import (
    filter_existing_duplicate_of_edges,
)
from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS, truncate_at_sentence

logger = logging.getLogger(__name__)

NodeSummaryFilter = Callable[[EntityNode], Awaitable[bool]]

# Threshold for deduplication similarity score
# If the best match score is below this threshold, treat as no duplicate
DEDUP_SIMILARITY_THRESHOLD = 0.7

# EasyOps: Batch size for summary extraction
# Process multiple entities in a single LLM call to reduce API calls
SUMMARY_BATCH_SIZE = 10


async def extract_nodes_reflexion(
    llm_client: LLMClient,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    node_names: list[str],
    group_id: str | None = None,
    entity_types_context: list[dict] | None = None,
) -> MissedEntities:
    """Perform reflexion on extracted entities - both positive and negative.

    Returns MissedEntities with:
    - missed_entities: entities that should have been extracted
    - entities_to_remove: entities that should NOT have been extracted
    - entities_to_reclassify: entities with wrong type classification
    """
    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'previous_episodes': [ep.content for ep in previous_episodes],
        'extracted_entities': node_names,
        'entity_types': entity_types_context or [],
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.reflexion(context),
        MissedEntities,
        group_id=group_id,
        prompt_name='extract_nodes.reflexion',
    )

    return MissedEntities(**llm_response)


async def filter_extracted_nodes(
    llm_client: LLMClient,
    episode: EpisodicNode,
    nodes: list[EntityNode],
    group_id: str | None = None,
    entity_types_context: list[dict] | None = None,
) -> tuple[list[str], list[EntityReclassification]]:
    """Filter out entities using a two-step validation approach.

    This approach reduces LLM attention issues by:
    - Step 1: Validate each entity against ONLY its assigned type definition (focused attention)
    - Step 2: For invalid entities only, provide ALL type definitions for reclassification

    The episode content is NOT passed - only entity name, summary, and type definition are used.
    This reduces context size and helps LLM focus on the validation task.

    Args:
        llm_client: LLM client for generating responses
        episode: The source episode (used for group_id only)
        nodes: List of EntityNode objects (with summary if available)
        group_id: Optional group ID
        entity_types_context: List of entity type definitions from schema
            Each dict should have: entity_type_id, entity_type_name, entity_type_description

    Returns:
        Tuple of (entities_to_remove, entities_to_reclassify)
        - entities_to_remove: List of entity names that should be removed
        - entities_to_reclassify: List of EntityReclassification objects for type correction
    """
    if not nodes:
        return [], []

    # Build type definition lookup
    type_definitions: dict[str, str] = {}
    if entity_types_context:
        for et in entity_types_context:
            type_name = et.get('entity_type_name', '')
            type_desc = et.get('entity_type_description', '')
            if type_name and type_desc:
                type_definitions[type_name] = type_desc

    # Step 1: Validate each entity against its assigned type
    # Build entity info with type definition for validation
    entities_for_validation = []
    for node in nodes:
        specific_type = next((l for l in node.labels if l != 'Entity'), None) or 'Entity'
        type_def = type_definitions.get(specific_type, '')

        entities_for_validation.append({
            'name': node.name,
            'summary': node.summary or '',
            'assigned_type': specific_type,
            'type_definition': type_def,
        })

    # Batch processing: 5 entities per batch, parallel execution
    BATCH_SIZE = 5
    batches = [
        entities_for_validation[i : i + BATCH_SIZE]
        for i in range(0, len(entities_for_validation), BATCH_SIZE)
    ]

    async def validate_batch(batch: list[dict]) -> list[tuple[str, bool, str]]:
        """Validate a batch of entities. Returns list of (name, is_valid, reason)."""
        context = {'entities': batch}
        llm_response = await llm_client.generate_response(
            prompt_library.extract_nodes.validate_entity_types(context),
            EntityValidationResult,
            group_id=group_id,
            prompt_name='extract_nodes.validate_entity_types',
        )
        result = EntityValidationResult(**llm_response)
        return [(v.name, v.is_valid, v.reason) for v in result.validations]

    # Run validation batches in parallel
    validation_results: list[list[tuple[str, bool, str]]] = await semaphore_gather(
        *[validate_batch(batch) for batch in batches]
    )

    # Collect invalid entities
    invalid_entities: list[dict] = []  # {name, summary, assigned_type, reason}
    for batch_results in validation_results:
        for name, is_valid, reason in batch_results:
            if not is_valid:
                # Find the original entity info
                entity_info = next(
                    (e for e in entities_for_validation if e['name'] == name), None
                )
                if entity_info:
                    invalid_entities.append({
                        'name': name,
                        'summary': entity_info['summary'],
                        'assigned_type': entity_info['assigned_type'],
                        'validation_reason': reason,
                    })
                    logger.info(
                        f'[filter_step1] Entity "{name}" failed validation for type '
                        f'"{entity_info["assigned_type"]}": {reason}'
                    )

    if not invalid_entities:
        logger.info('[filter] All entities passed type validation')
        return [], []

    # EasyOps: Step 2 (reclassify) is disabled - invalid entities are directly removed
    # Collect all invalid entity names for removal
    entities_to_remove: list[str] = [entity['name'] for entity in invalid_entities]
    logger.info(f'[filter] Removing {len(entities_to_remove)} invalid entities: {entities_to_remove}')

    # ========== BEGIN COMMENTED OUT - Step 2: Reclassify ==========
    # # Step 2: Reclassify invalid entities using the production-validated extract_text prompt
    # # For each invalid entity, use its name+summary as "text" and let LLM re-classify
    # entities_to_reclassify: list[EntityReclassification] = []
    #
    # async def reclassify_entity(entity: dict) -> tuple[str, str | None, str]:
    #     """Reclassify a single entity using extract_text prompt.
    #
    #     Returns: (name, new_type or None, reasoning)
    #     """
    #     # EasyOps fix: Use full episode content for reclassification context
    #     # instead of just name+summary, to preserve critical context
    #     reclassify_context = {
    #         'episode_content': episode.content,  # Full episode context
    #         'entity_types': entity_types_context,  # Full schema types
    #         # NOTE: Do NOT include validation_reason here - it may contain type suggestions
    #         # that would bias the LLM. Let extract_text make an independent classification.
    #         'custom_prompt': f"Focus on the entity '{entity['name']}' (description: {entity['summary']}). "
    #                        f"It was previously classified as '{entity['assigned_type']}' "
    #                        f"but that classification may be incorrect. "
    #                        f"Re-read the full episode content and determine the correct entity type "
    #                        f"from the available types based on how '{entity['name']}' is described in context.",
    #     }
    #
    #     llm_response = await llm_client.generate_response(
    #         prompt_library.extract_nodes.extract_text(reclassify_context),
    #         ExtractedEntities,
    #         group_id=group_id,
    #         prompt_name='extract_nodes.extract_text_reclassify',
    #     )
    #
    #     result = ExtractedEntities(**llm_response)
    #
    #     if not result.extracted_entities:
    #         # LLM didn't extract anything - entity should be removed
    #         return entity['name'], None, 'No valid entity type found during reclassification'
    #
    #     # Get the first extracted entity's classification
    #     extracted = result.extracted_entities[0]
    #     type_id = extracted.entity_type_id
    #     reasoning = extracted.reasoning if hasattr(extracted, 'reasoning') else ''
    #
    #     # Map type_id back to type name
    #     if 0 <= type_id < len(entity_types_context):
    #         new_type_name = entity_types_context[type_id].get('entity_type_name', 'Entity')
    #     else:
    #         new_type_name = 'Entity'
    #
    #     # If still Entity or same as failed type, remove it
    #     if new_type_name == 'Entity' or new_type_name == entity['assigned_type']:
    #         return entity['name'], None, f'Reclassified as {new_type_name}, still invalid. {reasoning}'
    #
    #     return entity['name'], new_type_name, reasoning
    #
    # # Run reclassification in parallel
    # reclassify_results = await semaphore_gather(
    #     *[reclassify_entity(entity) for entity in invalid_entities]
    # )
    #
    # # Process results
    # for name, new_type, reasoning in reclassify_results:
    #     if new_type is None:
    #         entities_to_remove.append(name)
    #         logger.info(f'[filter_step2] Removing "{name}": {reasoning}')
    #     else:
    #         entities_to_reclassify.append(
    #             EntityReclassification(name=name, new_type=new_type, reason=reasoning)
    #         )
    #         logger.info(f'[filter_step2] Reclassifying "{name}" to "{new_type}": {reasoning}')
    # ========== END COMMENTED OUT ==========

    return entities_to_remove, []  # No reclassifications when Step 2 is disabled


async def extract_nodes(
    clients: GraphitiClients,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
) -> list[EntityNode]:
    """Extract entities from episode content with type scoring.

    Uses scoring-based prompts that require LLM to evaluate all type options,
    ensuring deliberate classification decisions.

    Type confidence threshold: 0.6
    - If max score < 0.6, entity is classified as 'Entity' (default type)
    """
    TYPE_CONFIDENCE_THRESHOLD = 0.6

    start = time()
    perf_logger = logging.getLogger('graphiti.performance')
    llm_client = clients.llm_client
    llm_response = {}
    custom_prompt = ''
    entities_missed = True
    reflexion_iterations = 0
    llm_call_count = 0

    entity_types_context = [
        {
            'entity_type_id': 0,
            'entity_type_name': 'Entity',
            'entity_type_description': 'Default entity classification. Use this entity type if the entity is not one of the other listed types.',
        }
    ]

    entity_types_context += (
        [
            {
                'entity_type_id': i + 1,
                'entity_type_name': type_name,
                'entity_type_description': type_model.__doc__,
            }
            for i, (type_name, type_model) in enumerate(entity_types.items())
        ]
        if entity_types is not None
        else []
    )

    context = {
        'episode_content': episode.content,
        'episode_timestamp': episode.valid_at.isoformat(),
        'previous_episodes': [ep.content for ep in previous_episodes],
        'custom_prompt': custom_prompt,
        'entity_types': entity_types_context,
        'source_description': episode.source_description,
    }

    while entities_missed and reflexion_iterations <= MAX_REFLEXION_ITERATIONS:
        llm_start = time()
        # Use scoring versions of prompts for deliberate classification
        if episode.source == EpisodeType.message:
            llm_response = await llm_client.generate_response(
                prompt_library.extract_nodes.extract_message_with_scores(context),
                response_model=ExtractedEntitiesWithScores,
                group_id=episode.group_id,
                prompt_name='extract_nodes.extract_message_with_scores',
            )
        elif episode.source == EpisodeType.text:
            llm_response = await llm_client.generate_response(
                prompt_library.extract_nodes.extract_text_with_scores(context),
                response_model=ExtractedEntitiesWithScores,
                group_id=episode.group_id,
                prompt_name='extract_nodes.extract_text_with_scores',
            )
        elif episode.source == EpisodeType.json:
            # JSON extraction uses original prompt (scoring not needed for structured data)
            llm_response = await llm_client.generate_response(
                prompt_library.extract_nodes.extract_json(context),
                response_model=ExtractedEntities,
                group_id=episode.group_id,
                prompt_name='extract_nodes.extract_json',
            )
        elif episode.source == EpisodeType.document:
            # Document extraction uses same prompt as text
            llm_response = await llm_client.generate_response(
                prompt_library.extract_nodes.extract_text_with_scores(context),
                response_model=ExtractedEntitiesWithScores,
                group_id=episode.group_id,
                prompt_name='extract_nodes.extract_text_with_scores',
            )
        else:
            # Fallback to text extraction for unknown types
            llm_response = await llm_client.generate_response(
                prompt_library.extract_nodes.extract_text_with_scores(context),
                response_model=ExtractedEntitiesWithScores,
                group_id=episode.group_id,
                prompt_name='extract_nodes.extract_text_with_scores',
            )
        llm_call_count += 1
        perf_logger.info(f'[PERF]     └─ extract_nodes LLM call #{llm_call_count}: {(time() - llm_start)*1000:.0f}ms')

        # Handle both scored and non-scored responses
        if episode.source == EpisodeType.json:
            response_object = ExtractedEntities(**llm_response)
            extracted_entities = response_object.extracted_entities
        else:
            response_object = ExtractedEntitiesWithScores(**llm_response)
            extracted_entities = response_object.extracted_entities

        reflexion_iterations += 1
        if reflexion_iterations < MAX_REFLEXION_ITERATIONS:
            llm_start = time()
            reflexion_result = await extract_nodes_reflexion(
                llm_client,
                episode,
                previous_episodes,
                [entity.name for entity in extracted_entities],
                episode.group_id,
                entity_types_context,
            )
            llm_call_count += 1
            perf_logger.info(f'[PERF]     └─ extract_nodes reflexion #{llm_call_count}: {(time() - llm_start)*1000:.0f}ms')

            # Handle negative reflexion: remove entities that shouldn't have been extracted
            if reflexion_result.entities_to_remove:
                entities_to_remove_set = set(reflexion_result.entities_to_remove)
                original_count = len(extracted_entities)
                extracted_entities = [
                    e for e in extracted_entities if e.name not in entities_to_remove_set
                ]
                logger.info(
                    f'Negative reflexion removed {original_count - len(extracted_entities)} entities: '
                    f'{reflexion_result.entities_to_remove}'
                )

            # Handle reclassification: update entity types
            if reflexion_result.entities_to_reclassify:
                reclassify_map = {r.name: r.new_type for r in reflexion_result.entities_to_reclassify}
                # Find the type_id for each new_type
                type_name_to_id = {
                    et.get('entity_type_name'): i
                    for i, et in enumerate(entity_types_context)
                }
                for entity in extracted_entities:
                    if entity.name in reclassify_map:
                        new_type = reclassify_map[entity.name]
                        new_type_id = type_name_to_id.get(new_type)
                        if new_type_id is not None:
                            # Update the entity_type_id for ExtractedEntityWithScores
                            if hasattr(entity, 'final_type_id'):
                                entity.final_type_id = new_type_id
                            elif hasattr(entity, 'entity_type_id'):
                                entity.entity_type_id = new_type_id
                            logger.info(f'Reclassified entity "{entity.name}" to type "{new_type}"')

            # Handle positive reflexion: add missed entities to next extraction
            entities_missed = len(reflexion_result.missed_entities) != 0

            custom_prompt = 'Make sure that the following entities are extracted: '
            for entity in reflexion_result.missed_entities:
                custom_prompt += f'\n{entity},'

    filtered_extracted_entities = [entity for entity in extracted_entities if entity.name.strip()]
    end = time()
    logger.debug(f'Extracted new nodes: {filtered_extracted_entities} in {(end - start) * 1000} ms')
    perf_logger.info(f'[PERF]   └─ extract_nodes TOTAL: {(end - start)*1000:.0f}ms, entities={len(filtered_extracted_entities)}, llm_calls={llm_call_count}')

    # Threshold for ambiguous type resolution (multiple high-confidence candidates)
    AMBIGUOUS_SCORE_THRESHOLD = 0.7
    # Minimum score gap between top 1 and top 2 to skip second-pass
    # If top1 - top2 >= this value, we trust the first-pass result
    AMBIGUOUS_SCORE_GAP_THRESHOLD = 0.15

    # First pass: Convert extracted data to EntityNode objects
    # Track entities that need second-pass resolution (multiple candidates >= 0.7 AND close scores)
    extracted_nodes = []
    ambiguous_entities: list[tuple[int, ExtractedEntityWithScores, list[TopTypeCandidate]]] = []

    for idx, extracted_entity in enumerate(filtered_extracted_entities):
        # Handle scored entities (top 3 candidates)
        if isinstance(extracted_entity, ExtractedEntityWithScores):
            # Build type_scores dict for storage from top_candidates
            type_scores_dict = {}
            max_score = 0.0
            high_score_candidates: list[TopTypeCandidate] = []

            for tc in extracted_entity.top_candidates:
                if 0 <= tc.type_id < len(entity_types_context):
                    type_name = entity_types_context[tc.type_id].get('entity_type_name', 'Entity')
                    type_scores_dict[type_name] = {
                        'score': tc.score,
                        'reasoning': tc.reasoning,
                    }
                    if tc.score > max_score:
                        max_score = tc.score
                    # Track high-score candidates for potential second-pass resolution
                    if tc.score >= AMBIGUOUS_SCORE_THRESHOLD:
                        high_score_candidates.append(tc)

            # Apply threshold: if max score < 0.6, use Entity type
            if max_score < TYPE_CONFIDENCE_THRESHOLD:
                entity_type_name = 'Entity'
                type_confidence = max_score
                logger.info(
                    f'Entity "{extracted_entity.name}" max score {max_score:.2f} < threshold {TYPE_CONFIDENCE_THRESHOLD}, '
                    f'defaulting to Entity type'
                )
                reasoning = f'Low confidence ({max_score:.2f}), defaulted to Entity'
            else:
                type_id = extracted_entity.final_type_id
                if 0 <= type_id < len(entity_types_context):
                    entity_type_name = entity_types_context[type_id].get('entity_type_name', 'Entity')
                else:
                    entity_type_name = 'Entity'
                type_confidence = max_score

                # Check if this entity needs second-pass resolution
                # Only trigger if: (1) multiple high-score candidates AND (2) scores are close
                if len(high_score_candidates) > 1:
                    # Sort by score descending to check gap
                    sorted_candidates = sorted(high_score_candidates, key=lambda x: x.score, reverse=True)
                    top1_score = sorted_candidates[0].score
                    top2_score = sorted_candidates[1].score
                    score_gap = top1_score - top2_score

                    if score_gap < AMBIGUOUS_SCORE_GAP_THRESHOLD:
                        # Scores are close - needs second-pass resolution
                        ambiguous_entities.append((idx, extracted_entity, high_score_candidates))
                        logger.debug(
                            f'Entity "{extracted_entity.name}" has {len(high_score_candidates)} high-score candidates '
                            f'with close scores (gap={score_gap:.2f} < {AMBIGUOUS_SCORE_GAP_THRESHOLD}), '
                            f'marking for second-pass resolution'
                        )
                    else:
                        # Clear winner - skip second-pass
                        logger.debug(
                            f'Entity "{extracted_entity.name}" has clear winner (gap={score_gap:.2f} >= {AMBIGUOUS_SCORE_GAP_THRESHOLD}), '
                            f'skipping second-pass'
                        )

                # Get reasoning from the top candidate
                if extracted_entity.top_candidates:
                    top_candidate = extracted_entity.top_candidates[0]
                    reasoning = top_candidate.reasoning
                else:
                    reasoning = None
        else:
            # Handle non-scored entities (JSON extraction)
            type_id = extracted_entity.entity_type_id
            if 0 <= type_id < len(entity_types_context):
                entity_type_name = entity_types_context[type_id].get('entity_type_name', 'Entity')
            else:
                entity_type_name = 'Entity'

            reasoning = getattr(extracted_entity, 'reasoning', None)
            type_scores_dict = None
            type_confidence = None

        # Check if this entity type should be excluded
        if excluded_entity_types and entity_type_name in excluded_entity_types:
            logger.debug(f'Excluding entity "{extracted_entity.name}" of type "{entity_type_name}"')
            continue

        labels: list[str] = list({'Entity', str(entity_type_name)})

        new_node = EntityNode(
            name=extracted_entity.name,
            group_id=episode.group_id,
            labels=labels,
            summary='',
            created_at=utc_now(),
            reasoning=reasoning,
            type_scores=type_scores_dict,
            type_confidence=type_confidence,
        )
        extracted_nodes.append(new_node)
        logger.debug(f'Created new node: {new_node.name} (UUID: {new_node.uuid}, confidence: {type_confidence})')

    # Second pass: Resolve ambiguous entities with multiple high-confidence candidates
    if ambiguous_entities:
        llm_start = time()
        logger.info(f'Starting second-pass resolution for {len(ambiguous_entities)} ambiguous entities')

        # Build context for batch resolution
        ambiguous_entities_context = []
        for idx, extracted_entity, high_candidates in ambiguous_entities:
            candidate_type_ids = [tc.type_id for tc in high_candidates]
            ambiguous_entities_context.append({
                'name': extracted_entity.name,
                'candidate_type_ids': candidate_type_ids,
            })

        # Build candidate types context (only include types that are candidates)
        all_candidate_type_ids = set()
        for _, _, high_candidates in ambiguous_entities:
            for tc in high_candidates:
                all_candidate_type_ids.add(tc.type_id)

        candidate_types_context = [
            entity_types_context[type_id]
            for type_id in sorted(all_candidate_type_ids)
            if 0 <= type_id < len(entity_types_context)
        ]

        resolution_context = {
            'episode_content': episode.content,
            'ambiguous_entities': ambiguous_entities_context,
            'candidate_types': candidate_types_context,
        }

        try:
            resolution_response = await llm_client.generate_response(
                prompt_library.extract_nodes.resolve_ambiguous_types(resolution_context),
                response_model=ResolvedEntityTypes,
                group_id=episode.group_id,
                prompt_name='extract_nodes.resolve_ambiguous_types',
            )
            llm_call_count += 1
            perf_logger.info(f'[PERF]     └─ resolve_ambiguous_types: {(time() - llm_start)*1000:.0f}ms, entities={len(ambiguous_entities)}')

            resolutions = ResolvedEntityTypes(**resolution_response)

            # Build resolution map with candidate_scores
            resolution_map: dict[str, tuple[int, str, list[CandidateTypeScore]]] = {}
            for r in resolutions.resolutions:
                resolution_map[r.name] = (r.chosen_type_id, r.reasoning, r.candidate_scores)

            for idx, extracted_entity, high_candidates in ambiguous_entities:
                if extracted_entity.name in resolution_map:
                    chosen_type_id, resolution_reasoning, candidate_scores = resolution_map[extracted_entity.name]

                    # Find the corresponding node and update it
                    for node in extracted_nodes:
                        if node.name == extracted_entity.name:
                            if 0 <= chosen_type_id < len(entity_types_context):
                                new_type_name = entity_types_context[chosen_type_id].get('entity_type_name', 'Entity')
                                node.labels = list({'Entity', new_type_name})
                                node.reasoning = f'[Second-pass resolution] {resolution_reasoning}'

                                # Update type_scores: preserve pass1, add pass2 scores
                                if node.type_scores is None:
                                    node.type_scores = {}

                                # Move current scores to pass1_* and update with pass2 scores
                                for cs in candidate_scores:
                                    if 0 <= cs.type_id < len(entity_types_context):
                                        type_name = entity_types_context[cs.type_id].get('entity_type_name', 'Entity')
                                        if type_name in node.type_scores:
                                            # Preserve pass1 data
                                            pass1_data = node.type_scores[type_name]
                                            node.type_scores[type_name] = {
                                                'score': cs.score,
                                                'reasoning': cs.reasoning,
                                                'pass1_score': pass1_data.get('score'),
                                                'pass1_reasoning': pass1_data.get('reasoning'),
                                            }
                                        else:
                                            # No pass1 data for this type
                                            node.type_scores[type_name] = {
                                                'score': cs.score,
                                                'reasoning': cs.reasoning,
                                            }

                                # Update type_confidence with pass2 chosen type's score
                                chosen_score = None
                                for cs in candidate_scores:
                                    if cs.type_id == chosen_type_id:
                                        chosen_score = cs.score
                                        break
                                if chosen_score is not None:
                                    node.type_confidence = chosen_score

                                logger.info(
                                    f'Resolved ambiguous entity "{node.name}" to type "{new_type_name}" '
                                    f'(pass2 confidence: {chosen_score}) via second-pass with scoring'
                                )
                            break
        except Exception as e:
            logger.warning(f'Second-pass resolution failed: {e}, using first-pass results')

    logger.debug(f'Extracted nodes: {[(n.name, n.uuid) for n in extracted_nodes]}')

    return extracted_nodes


async def _collect_candidate_nodes(
    clients: GraphitiClients,
    extracted_nodes: list[EntityNode],
    existing_nodes_override: list[EntityNode] | None,
) -> list[EntityNode]:
    """Search per extracted name and return unique candidates with overrides honored in order.

    EasyOps customization: Filter candidates by same entity type to improve deduplication
    accuracy for entities with different names but same semantic meaning (e.g., synonyms).
    """
    perf_logger = logging.getLogger('graphiti.performance')

    # Pre-batch embeddings for all node names to avoid per-search embedding calls
    step_start = time()
    node_names = [node.name.replace('\n', ' ') for node in extracted_nodes]
    if node_names:
        query_embeddings = await clients.embedder.create_batch(node_names)
    else:
        query_embeddings = []
    perf_logger.info(f'[PERF]       └─ batch_search_embeddings: {(time() - step_start)*1000:.0f}ms, count={len(node_names)}')

    step_start = time()

    def _get_specific_label(labels: list[str]) -> str | None:
        """Get the most specific label (non-'Entity') from labels list."""
        for label in labels:
            if label != 'Entity':
                return label
        return None

    search_results: list[SearchResults] = await semaphore_gather(
        *[
            search(
                clients=clients,
                query=node.name,
                group_ids=[node.group_id],
                # EasyOps: Filter by same entity type to find potential duplicates
                search_filter=SearchFilters(
                    node_labels=[_get_specific_label(node.labels)]
                    if _get_specific_label(node.labels)
                    else None
                ),
                config=NODE_HYBRID_SEARCH_RRF,
                query_vector=query_embeddings[i] if i < len(query_embeddings) else None,
            )
            for i, node in enumerate(extracted_nodes)
        ]
    )
    perf_logger.info(f'[PERF]       └─ parallel_node_search: {(time() - step_start)*1000:.0f}ms')

    candidate_nodes: list[EntityNode] = [node for result in search_results for node in result.nodes]

    if existing_nodes_override is not None:
        candidate_nodes.extend(existing_nodes_override)

    seen_candidate_uuids: set[str] = set()
    ordered_candidates: list[EntityNode] = []
    for candidate in candidate_nodes:
        if candidate.uuid in seen_candidate_uuids:
            continue
        seen_candidate_uuids.add(candidate.uuid)
        ordered_candidates.append(candidate)

    return ordered_candidates


# Configuration for batched LLM deduplication
DEDUP_BATCH_SIZE = 5  # Number of entities per LLM request
DEDUP_PARALLELISM = 10  # Maximum concurrent LLM requests


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

    Uses scoring-based deduplication that requires LLM to score each candidate match.
    Only merges entities when similarity_score >= DEDUP_SIMILARITY_THRESHOLD (0.7).

    EasyOps customization: Process entities in batches to avoid LLM output truncation.
    Each batch contains up to DEDUP_BATCH_SIZE entities, processed with DEDUP_PARALLELISM concurrency.

    The guardrails below defensively ignore malformed or duplicate LLM responses so the
    ingestion workflow remains deterministic even when the model misbehaves.
    """
    if not state.unresolved_indices:
        return

    entity_types_dict: dict[str, type[BaseModel]] = entity_types if entity_types is not None else {}

    llm_extracted_nodes = [extracted_nodes[i] for i in state.unresolved_indices]

    # Build entity type definitions separately to avoid repetition
    entity_type_definitions: dict[str, str] = {}
    for node in llm_extracted_nodes:
        for label in node.labels:
            if label != 'Entity' and label not in entity_type_definitions:
                type_model = entity_types_dict.get(label)
                if type_model and type_model.__doc__:
                    entity_type_definitions[label] = type_model.__doc__

    # Build existing nodes context (shared across all batches)
    existing_nodes_context = [
        {
            **{
                'idx': i,
                'name': candidate.name,
                'entity_types': candidate.labels,
            },
            **candidate.attributes,
        }
        for i, candidate in enumerate(indexes.existing_nodes)
    ]

    # Split entities into batches
    batches: list[list[tuple[int, EntityNode]]] = []
    current_batch: list[tuple[int, EntityNode]] = []
    for global_idx, node in enumerate(llm_extracted_nodes):
        current_batch.append((global_idx, node))
        if len(current_batch) >= DEDUP_BATCH_SIZE:
            batches.append(current_batch)
            current_batch = []
    if current_batch:
        batches.append(current_batch)

    logger.info(
        'Deduplicating %d entities in %d batches (batch_size=%d, parallelism=%d)',
        len(llm_extracted_nodes),
        len(batches),
        DEDUP_BATCH_SIZE,
        DEDUP_PARALLELISM,
    )

    async def process_batch(
        batch: list[tuple[int, EntityNode]],
        batch_idx: int,
    ) -> list[tuple[int, NodeDuplicateWithScores]]:
        """Process a single batch and return results with global indices."""
        # Build batch-specific extracted nodes context with local IDs (0 to batch_size-1)
        batch_extracted_context = [
            {
                'id': local_idx,
                'name': node.name,
                'entity_type': node.labels,
            }
            for local_idx, (_, node) in enumerate(batch)
        ]

        context = {
            'extracted_nodes': batch_extracted_context,
            'existing_nodes': existing_nodes_context,
            'entity_type_definitions': entity_type_definitions,
            'episode_content': episode.content if episode is not None else '',
            'previous_episodes': (
                [ep.content for ep in previous_episodes] if previous_episodes is not None else []
            ),
        }

        try:
            llm_response = await llm_client.generate_response(
                prompt_library.dedupe_nodes.nodes_with_scores(context),
                response_model=NodeResolutionsWithScores,
                prompt_name='dedupe_nodes.nodes_with_scores',
            )
            batch_resolutions = NodeResolutionsWithScores(**llm_response).entity_resolutions

            # Map local IDs back to global indices
            results: list[tuple[int, NodeDuplicateWithScores]] = []
            for resolution in batch_resolutions:
                local_id = resolution.id
                if 0 <= local_id < len(batch):
                    global_idx = batch[local_id][0]
                    results.append((global_idx, resolution))
                else:
                    logger.warning(
                        'Batch %d: invalid local ID %d (batch size: %d)',
                        batch_idx,
                        local_id,
                        len(batch),
                    )
            return results
        except Exception as e:
            logger.error('Batch %d failed: %s', batch_idx, e)
            return []

    # Process batches in parallel with semaphore
    all_results: list[tuple[int, NodeDuplicateWithScores]] = []
    batch_results = await semaphore_gather(
        *[process_batch(batch, idx) for idx, batch in enumerate(batches)],
        max_coroutines=DEDUP_PARALLELISM,
    )
    for batch_result in batch_results:
        all_results.extend(batch_result)

    # Process all resolutions
    valid_relative_range = range(len(state.unresolved_indices))
    processed_relative_ids: set[int] = set()

    received_ids = {global_idx for global_idx, _ in all_results}
    expected_ids = set(valid_relative_range)
    missing_ids = expected_ids - received_ids
    extra_ids = received_ids - expected_ids

    logger.debug(
        'Received %d resolutions for %d entities',
        len(all_results),
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

    for relative_id, resolution in all_results:
        best_match_idx: int = resolution.best_match_idx
        confidence: float = resolution.confidence

        if relative_id not in valid_relative_range:
            logger.warning(
                'Skipping invalid LLM dedupe id %d (valid range: 0-%d, received %d resolutions)',
                relative_id,
                len(state.unresolved_indices) - 1,
                len(all_results),
            )
            continue

        if relative_id in processed_relative_ids:
            logger.warning('Duplicate LLM dedupe id %s received; ignoring.', relative_id)
            continue
        processed_relative_ids.add(relative_id)

        original_index = state.unresolved_indices[relative_id]
        extracted_node = extracted_nodes[original_index]

        # Find the best match score from candidate_scores
        best_score = 0.0
        best_reasoning = ''
        for candidate_score in resolution.candidate_scores:
            if candidate_score.candidate_idx == best_match_idx and candidate_score.is_same_entity:
                best_score = candidate_score.similarity_score
                best_reasoning = candidate_score.reasoning
                break

        resolved_node: EntityNode
        # Apply threshold: only merge if score >= DEDUP_SIMILARITY_THRESHOLD
        if best_match_idx == -1 or best_score < DEDUP_SIMILARITY_THRESHOLD:
            resolved_node = extracted_node
            if best_match_idx != -1 and best_score < DEDUP_SIMILARITY_THRESHOLD:
                logger.info(
                    'Dedupe: "%s" NOT merged with "%s" - score %.2f < threshold %.2f (confidence: %.2f)',
                    extracted_node.name,
                    indexes.existing_nodes[best_match_idx].name if 0 <= best_match_idx < len(indexes.existing_nodes) else 'unknown',
                    best_score,
                    DEDUP_SIMILARITY_THRESHOLD,
                    confidence,
                )
        elif 0 <= best_match_idx < len(indexes.existing_nodes):
            resolved_node = indexes.existing_nodes[best_match_idx]
        else:
            logger.warning(
                'Invalid best_match_idx %s for extracted node %s; treating as no duplicate.',
                best_match_idx,
                extracted_node.uuid,
            )
            resolved_node = extracted_node

        state.resolved_nodes[original_index] = resolved_node
        state.uuid_map[extracted_node.uuid] = resolved_node.uuid
        if resolved_node.uuid != extracted_node.uuid:
            state.duplicate_pairs.append((extracted_node, resolved_node))
            # Log deduplication decision with score and reasoning for debugging
            logger.info(
                'Dedupe: "%s" -> "%s" (score: %.2f, confidence: %.2f, reasoning: %s)',
                extracted_node.name,
                resolved_node.name,
                best_score,
                confidence,
                best_reasoning or 'no reasoning provided',
            )


async def resolve_extracted_nodes(
    clients: GraphitiClients,
    extracted_nodes: list[EntityNode],
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_types: dict[str, type[BaseModel]] | None = None,
    existing_nodes_override: list[EntityNode] | None = None,
) -> tuple[list[EntityNode], dict[str, str], list[tuple[EntityNode, EntityNode]]]:
    """Search for existing nodes, resolve deterministic matches, then escalate holdouts to the LLM dedupe prompt."""
    llm_client = clients.llm_client
    driver = clients.driver
    existing_nodes = await _collect_candidate_nodes(
        clients,
        extracted_nodes,
        existing_nodes_override,
    )

    indexes: DedupCandidateIndexes = _build_candidate_indexes(existing_nodes)

    state = DedupResolutionState(
        resolved_nodes=[None] * len(extracted_nodes),
        uuid_map={},
        unresolved_indices=[],
    )

    _resolve_with_similarity(extracted_nodes, indexes, state)

    await _resolve_with_llm(
        llm_client,
        extracted_nodes,
        indexes,
        state,
        episode,
        previous_episodes,
        entity_types,
    )

    for idx, node in enumerate(extracted_nodes):
        if state.resolved_nodes[idx] is None:
            state.resolved_nodes[idx] = node
            state.uuid_map[node.uuid] = node.uuid

    logger.debug(
        'Resolved nodes: %s',
        [(node.name, node.uuid) for node in state.resolved_nodes if node is not None],
    )

    new_node_duplicates: list[
        tuple[EntityNode, EntityNode]
    ] = await filter_existing_duplicate_of_edges(driver, state.duplicate_pairs)

    return (
        [node for node in state.resolved_nodes if node is not None],
        state.uuid_map,
        new_node_duplicates,
    )


async def extract_attributes_from_nodes(
    clients: GraphitiClients,
    nodes: list[EntityNode],
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_types: dict[str, type[BaseModel]] | None = None,
    should_summarize_node: NodeSummaryFilter | None = None,
) -> list[EntityNode]:
    """Extract attributes and summaries from nodes.

    EasyOps optimization: Uses batch summary extraction to reduce LLM calls.
    - Attributes: extracted in parallel (each node needs its own entity_type)
    - Summaries: extracted in batches of SUMMARY_BATCH_SIZE
    """
    llm_client = clients.llm_client
    embedder = clients.embedder

    # Step 1: Extract attributes in parallel (each node needs different entity_type)
    await semaphore_gather(
        *[
            _extract_attributes_and_update(
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

    # Step 2: Extract summaries in batches
    # Split nodes into batches
    batches: list[list[EntityNode]] = []
    for i in range(0, len(nodes), SUMMARY_BATCH_SIZE):
        batches.append(nodes[i : i + SUMMARY_BATCH_SIZE])

    logger.info(
        f'[bulk_summary] Processing {len(nodes)} nodes in {len(batches)} batches '
        f'(batch_size={SUMMARY_BATCH_SIZE})'
    )

    # Process batches in parallel
    await semaphore_gather(
        *[
            _extract_entity_summaries_bulk(
                llm_client, batch, episode, previous_episodes, should_summarize_node
            )
            for batch in batches
        ]
    )

    await create_entity_node_embeddings(embedder, nodes)

    return nodes


async def _extract_attributes_and_update(
    llm_client: LLMClient,
    node: EntityNode,
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
    entity_type: type[BaseModel] | None,
) -> None:
    """Extract attributes and update node in place.

    EasyOps: Added exception handling to prevent single entity failure from
    affecting the entire batch. If attribute extraction fails, the node's
    attributes remain unchanged and a warning is logged.
    """
    try:
        result = await _extract_entity_attributes(
            llm_client, node, episode, previous_episodes, entity_type
        )
        node.attributes.update(result)
    except Exception as e:
        # Log warning but don't crash - let the batch continue
        logger.warning(
            f'Failed to extract attributes for entity "{node.name}" '
            f'(type: {node.labels}): {e}. Attributes will remain empty.'
        )


async def extract_attributes_from_node(
    llm_client: LLMClient,
    node: EntityNode,
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_type: type[BaseModel] | None = None,
    should_summarize_node: NodeSummaryFilter | None = None,
) -> EntityNode:
    """Extract attributes and summary for a single node.

    Note: This is kept for backwards compatibility. For bulk operations,
    use extract_attributes_from_nodes which uses batch summary extraction.
    """
    # Extract attributes if entity type is defined and has attributes
    llm_response = await _extract_entity_attributes(
        llm_client, node, episode, previous_episodes, entity_type
    )

    # Extract summary if needed
    await _extract_entity_summary(
        llm_client, node, episode, previous_episodes, should_summarize_node
    )

    node.attributes.update(llm_response)

    return node


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

    # validate response with graceful error handling for invalid enum values
    try:
        entity_type(**llm_response)
    except ValidationError as e:
        # EasyOps customization: handle invalid enum values gracefully
        # Remove fields that failed validation instead of crashing
        logger.warning(
            f'Entity attribute validation warning for {entity_type.__name__}: {e}. '
            f'Will remove invalid fields.'
        )
        logger.debug(f'LLM response was: {llm_response}')

        # Extract field names that have validation errors
        invalid_fields = set()
        for error in e.errors():
            if error.get('loc'):
                field_name = error['loc'][0]
                invalid_fields.add(field_name)
                logger.warning(
                    f'Removing invalid field "{field_name}" with value "{llm_response.get(field_name)}": '
                    f'{error.get("msg")}'
                )

        # Remove invalid fields and try again
        cleaned_response = {k: v for k, v in llm_response.items() if k not in invalid_fields}

        # Validate cleaned response
        try:
            entity_type(**cleaned_response)
            logger.info(f'Cleaned response validated successfully with {len(cleaned_response)} fields')
            return cleaned_response
        except ValidationError as e2:
            logger.error(f'Entity attribute validation still failed after cleanup: {e2}')
            # Return empty dict rather than crash
            return {}
    except Exception as e:
        logger.error(f'Entity attribute validation failed for {entity_type.__name__}: {e}')
        logger.error(f'LLM response was: {llm_response}')
        raise

    return llm_response


async def _extract_entity_summary(
    llm_client: LLMClient,
    node: EntityNode,
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
    should_summarize_node: NodeSummaryFilter | None,
) -> None:
    if should_summarize_node is not None and not await should_summarize_node(node):
        return

    summary_context = _build_episode_context(
        node_data={
            'name': node.name,
            'summary': truncate_at_sentence(node.summary, MAX_SUMMARY_CHARS),
            'entity_types': node.labels,
            'attributes': node.attributes,
        },
        episode=episode,
        previous_episodes=previous_episodes,
    )

    summary_response = await llm_client.generate_response(
        prompt_library.extract_nodes.extract_summary(summary_context),
        response_model=EntitySummary,
        model_size=ModelSize.small,
        group_id=node.group_id,
        prompt_name='extract_nodes.extract_summary',
    )

    node.summary = truncate_at_sentence(summary_response.get('summary', ''), MAX_SUMMARY_CHARS)


# EasyOps: Batch summary extraction - process multiple entities in one LLM call
async def _extract_entity_summaries_bulk(
    llm_client: LLMClient,
    nodes: list[EntityNode],
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
    should_summarize_node: NodeSummaryFilter | None,
) -> None:
    """Extract summaries for multiple entities in a single LLM call.

    This reduces LLM API calls from N to N/SUMMARY_BATCH_SIZE.
    Each batch is processed in parallel with other batches.
    """
    if not nodes:
        return

    # Filter nodes that need summarization
    nodes_to_process: list[EntityNode] = []
    if should_summarize_node is not None:
        for node in nodes:
            if await should_summarize_node(node):
                nodes_to_process.append(node)
    else:
        nodes_to_process = list(nodes)

    if not nodes_to_process:
        return

    # Build entities list for the prompt
    entities_data = []
    for i, node in enumerate(nodes_to_process):
        entities_data.append({
            'id': i,
            'name': node.name,
            'summary': truncate_at_sentence(node.summary, MAX_SUMMARY_CHARS) if node.summary else '',
            'entity_types': node.labels,
            'attributes': node.attributes,
        })

    # Build context for bulk extraction
    context = {
        'entities': entities_data,
        'episode_content': episode.content if episode is not None else '',
        'previous_episodes': (
            [ep.content for ep in previous_episodes] if previous_episodes is not None else []
        ),
    }

    # Use the first node's group_id for logging
    group_id = nodes_to_process[0].group_id if nodes_to_process else None

    try:
        response = await llm_client.generate_response(
            prompt_library.extract_nodes.extract_summaries_bulk(context),
            response_model=EntitySummaries,
            model_size=ModelSize.small,
            group_id=group_id,
            prompt_name='extract_nodes.extract_summaries_bulk',
        )

        # Parse response and update node summaries
        summaries_list = response.get('summaries', [])
        summaries_by_id: dict[int, str] = {}
        for item in summaries_list:
            entity_id = item.get('entity_id')
            summary = item.get('summary', '')
            if entity_id is not None:
                summaries_by_id[entity_id] = summary

        # Update nodes with extracted summaries
        for i, node in enumerate(nodes_to_process):
            if i in summaries_by_id:
                node.summary = truncate_at_sentence(summaries_by_id[i], MAX_SUMMARY_CHARS)
            else:
                logger.warning(f'No summary returned for entity {node.name} (id={i})')

        logger.info(
            f'[bulk_summary] Extracted summaries for {len(summaries_by_id)}/{len(nodes_to_process)} entities'
        )

    except Exception as e:
        logger.error(f'[bulk_summary] Failed to extract summaries in batch: {e}')
        # Fallback to individual extraction
        logger.info(f'[bulk_summary] Falling back to individual extraction for {len(nodes_to_process)} entities')
        for node in nodes_to_process:
            try:
                await _extract_entity_summary(
                    llm_client, node, episode, previous_episodes, should_summarize_node
                )
            except Exception as fallback_error:
                logger.error(f'[bulk_summary] Individual fallback failed for {node.name}: {fallback_error}')


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


# =============================================================================
# Phase 2: Subtype Refinement (Two-Phase Type Recognition)
# =============================================================================


def _get_primary_type(labels: list[str]) -> str:
    """Get the most specific (non-Entity) type from labels."""
    for label in labels:
        if label != 'Entity':
            return label
    return 'Entity'


def _update_labels(labels: list[str], kernel_type: str, refined_type: str) -> list[str]:
    """
    Update labels to include full inheritance chain.

    Input: labels=["Entity", "Event"], kernel="Event", refined="Ticket"
    Output: ["Entity", "Event", "Ticket"]

    Ensures:
    1. Entity is always first
    2. Kernel type is present
    3. Refined type is appended
    4. No duplicates
    """
    # Start with Entity
    new_labels = ['Entity']

    # Add Kernel type if not Entity
    if kernel_type != 'Entity' and kernel_type not in new_labels:
        new_labels.append(kernel_type)

    # Add refined type
    if refined_type not in new_labels:
        new_labels.append(refined_type)

    return new_labels


async def refine_entity_types_from_summary(
    llm_client: LLMClient,
    nodes: list[EntityNode],
    kernel_to_subtypes: dict[str, list[str]],
    entity_types: dict[str, type[BaseModel]],
    group_id: str,
    max_coroutines: int | None = None,
) -> list[EntityNode]:
    """
    Phase 2: Refine entity types from Kernel to specific subtypes.

    Triggered after extract_attributes_from_nodes() when summaries are available.
    Only refines entities whose current type is a Kernel type with defined subtypes.

    Process:
    1. Group nodes by Kernel type
    2. For each Kernel type with subtypes, call LLM to determine specific subtype
    3. Update node labels with inheritance chain (e.g., ["Entity", "Event", "Ticket"])
    4. Store refinement confidence and reasoning

    Args:
        llm_client: LLM client for inference
        nodes: Entities with summaries already generated
        kernel_to_subtypes: {kernel_type: [subtype_names]} mapping
        entity_types: Full entity type definitions (for subtype descriptions)
        group_id: Tenant ID for logging
        max_coroutines: Concurrency limit for parallel LLM calls

    Returns:
        Updated nodes with refined types and labels
    """
    perf_logger = logging.getLogger('graphiti.performance')
    start = time()

    # Identify nodes that need refinement
    nodes_to_refine: list[tuple[EntityNode, str, list[str]]] = []
    for node in nodes:
        current_type = _get_primary_type(node.labels)
        if current_type in kernel_to_subtypes:
            subtypes = kernel_to_subtypes[current_type]
            if subtypes:  # Only refine if subtypes exist
                nodes_to_refine.append((node, current_type, subtypes))

    if not nodes_to_refine:
        logger.debug('Phase 2: No nodes require subtype refinement')
        return nodes

    logger.info(f'Phase 2: Refining {len(nodes_to_refine)} nodes with potential subtypes')

    # Group by Kernel type for batch processing
    by_kernel: dict[str, list[EntityNode]] = defaultdict(list)
    kernel_subtypes_map: dict[str, list[str]] = {}

    for node, kernel_type, subtypes in nodes_to_refine:
        by_kernel[kernel_type].append(node)
        kernel_subtypes_map[kernel_type] = subtypes

    async def _refine_batch(
        kernel_type: str,
        batch_nodes: list[EntityNode],
    ) -> None:
        """Refine a batch of nodes belonging to the same Kernel type."""
        subtypes = kernel_subtypes_map[kernel_type]

        # Build subtype definitions for prompt
        subtypes_defs = []
        for st_name in subtypes:
            st_model = entity_types.get(st_name)
            if st_model:
                subtypes_defs.append({
                    'name': st_name,
                    'description': st_model.__doc__ or f'Subtype: {st_name}',
                })
            else:
                subtypes_defs.append({
                    'name': st_name,
                    'description': f'Subtype of {kernel_type}',
                })

        # Build entities context (name + summary + attributes)
        entities_context = [
            {
                'name': node.name,
                'summary': node.summary or '',
                'attributes': node.attributes or {},
            }
            for node in batch_nodes
        ]

        context = {
            'kernel_type': kernel_type,
            'subtypes': subtypes_defs,
            'entities': entities_context,
        }

        try:
            response = await llm_client.generate_response(
                prompt_library.extract_nodes.refine_types_from_summary(context),
                response_model=RefinedEntityTypes,
                group_id=group_id,
                prompt_name='extract_nodes.refine_types_from_summary',
            )

            refinements = RefinedEntityTypes(**response)

            # Build refinement map by entity name
            refinement_map = {r.entity_name: r for r in refinements.refinements}

            # Apply refinements
            for node in batch_nodes:
                if node.name in refinement_map:
                    r = refinement_map[node.name]

                    if r.refined_type and r.refined_type in subtypes:
                        # Update labels to include inheritance chain
                        node.labels = _update_labels(node.labels, kernel_type, r.refined_type)

                        # Store Phase 2 confidence and reasoning
                        node.type_confidence = r.confidence
                        node.reasoning = (
                            f'[Phase 2 refinement] {r.reasoning}\n'
                            f'Refined from {kernel_type} to {r.refined_type}'
                        )

                        # Update type_scores with Phase 2 data
                        if node.type_scores is None:
                            node.type_scores = {}
                        node.type_scores['phase2_refinement'] = {
                            'kernel_type': kernel_type,
                            'refined_type': r.refined_type,
                            'confidence': r.confidence,
                            'reasoning': r.reasoning,
                        }

                        logger.info(
                            f'Phase 2: Refined "{node.name}" from {kernel_type} to {r.refined_type} '
                            f'(confidence: {r.confidence:.2f})'
                        )
                    else:
                        # Keep as Kernel type (Fallback)
                        logger.debug(
                            f'Phase 2: Kept "{node.name}" as {kernel_type} '
                            f'(no matching subtype, refined_type={r.refined_type})'
                        )

        except Exception as e:
            logger.warning(f'Phase 2 refinement failed for {kernel_type}: {e}')

    # Run refinements in parallel by Kernel type (using semaphore_gather)
    await semaphore_gather(
        *[_refine_batch(kt, batch_nodes) for kt, batch_nodes in by_kernel.items()],
        max_coroutines=max_coroutines,
    )

    end = time()
    refined_count = sum(
        1 for node, kt, _ in nodes_to_refine
        if _get_primary_type(node.labels) != kt  # Type was changed
    )
    perf_logger.info(
        f'[PERF] refine_entity_types (Phase 2): {(end - start)*1000:.0f}ms, '
        f'candidates={len(nodes_to_refine)}, refined={refined_count}'
    )

    return nodes
