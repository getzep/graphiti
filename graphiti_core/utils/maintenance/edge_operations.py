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
import re
from datetime import datetime
from difflib import SequenceMatcher
from time import time
from typing import Literal

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
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate
from graphiti_core.prompts.extract_edges import Edge as ExtractedEdge
from graphiti_core.prompts.extract_edges import ExtractedEdges
from graphiti_core.search.search import search
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# constrained_soft enforcement helpers
# ---------------------------------------------------------------------------

# Generic connector relation types that carry no domain-specific signal.
# In constrained_soft mode, edges with these relation_types that have no
# ontology match are dropped after LLM extraction.
# All names are stored in SCREAMING_SNAKE_CASE; comparison uses normalized form
# so mixed-case / spaced variants from the LLM (e.g. 'relates to') are caught.
_GENERIC_EDGE_NAMES: frozenset[str] = frozenset({
    'RELATES_TO',
    'IS_RELATED_TO',
    'IS_RELATED',
    'MENTIONS',
    'DISCUSSED',
    'CONNECTED_TO',
    'ASSOCIATED_WITH',
    'HAS',
    'CONTAINS',
    'INCLUDES',
    'LINKS_TO',
    'REFERENCES',
    'IS_CONNECTED_TO',
    'IS_ASSOCIATED_WITH',
})

# Similarity threshold for near-miss canonicalization.
# A relation_type that scores ≥ this against any ontology name is snapped to that name.
_CANONICALIZE_THRESHOLD: float = 0.78


def _normalize_relation_type(relation_type: str) -> str:
    """Normalize a relation type string to SCREAMING_SNAKE_CASE for comparison.

    Performs these transforms in order:
    1. Strip surrounding whitespace.
    2. Replace ANY non-alphanumeric character (spaces, hyphens, dots, carets,
       colons, etc.) with an underscore — closes punctuation-bypass vectors
       like ``'RELATES^TO'`` or ``'MENTIONS.'``.
    3. Collapse runs of consecutive underscores to a single underscore.
    4. Trim leading/trailing underscores left by step 2.
    5. Uppercase everything.

    This ensures that LLM outputs like ``'relates_to'``, ``'Relates To'``,
    ``'relates-to'``, ``'RELATES^TO'``, or ``'MENTIONS.'`` all compare equal
    to the canonical ``'RELATES_TO'`` / ``'MENTIONS'`` entry in the ontology /
    noise filter, without requiring an exact-case match.
    """
    s = relation_type.strip()
    s = re.sub(r'[^a-zA-Z0-9]+', '_', s)   # any non-alnum → underscore
    s = re.sub(r'_+', '_', s)               # collapse repeated underscores
    s = s.strip('_')                         # trim leading/trailing underscores
    return s.upper()


def _canonicalize_edge_name(
    relation_type: str,
    ontology_names: frozenset[str],
    threshold: float = _CANONICALIZE_THRESHOLD,
) -> str:
    """Snap a near-miss relation_type to the closest ontology name.

    Normalizes the input to SCREAMING_SNAKE_CASE before comparison so that
    LLM variants with different casing or separators are handled consistently.
    If the best similarity ratio is below *threshold*, returns the *normalized*
    form of the original (not an ontology name).  Only exact-match or
    high-similarity (≥ threshold) names are canonicalised.

    A negation polarity guard prevents canonicalization from flipping the
    semantic polarity: a ``NOT_*`` relation will never be snapped to a non-
    ``NOT_*`` ontology name (and vice versa).

    Parameters
    ----------
    relation_type:
        The relation type string returned by the LLM.
    ontology_names:
        Set of canonical relation type names from the lane ontology.
    threshold:
        Minimum SequenceMatcher ratio to accept a canonicalisation.

    Returns
    -------
    str
        Canonical ontology name if a close match exists; otherwise the
        normalized (SCREAMING_SNAKE_CASE) form of the original input.
    """
    # Normalize first — ensures consistent casing for all subsequent comparisons.
    normalized = _normalize_relation_type(relation_type)

    if not ontology_names or normalized in ontology_names:
        # Exact match after normalization (or no ontology to check).
        if normalized != relation_type:
            logger.info(
                'constrained_soft: canonicalized edge %r → %r (normalization)',
                relation_type,
                normalized,
            )
        return normalized

    best_name = normalized
    best_ratio = 0.0
    for canonical in ontology_names:
        ratio = SequenceMatcher(None, normalized, canonical).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = canonical

    if best_ratio >= threshold:
        # Negation polarity guard: never snap across the NOT_ boundary.
        # e.g., NOT_RELATED_TO must not be canonicalized to RELATED_TO.
        original_is_negated = normalized.startswith('NOT_')
        candidate_is_negated = best_name.startswith('NOT_')
        if original_is_negated != candidate_is_negated:
            logger.debug(
                'constrained_soft: polarity guard blocked %r → %r (NOT_ boundary mismatch)',
                normalized,
                best_name,
            )
            return normalized  # keep normalized form; do not flip polarity

        if best_name != relation_type:
            logger.info(
                'constrained_soft: canonicalized edge %r → %r (ratio=%.2f)',
                relation_type,
                best_name,
                best_ratio,
            )
        return best_name

    # No close ontology match — return the normalized form so casing is consistent.
    return normalized


def _should_filter_constrained_edge(
    relation_type: str,
    ontology_names: frozenset[str],
) -> bool:
    """Return True if an edge should be dropped in constrained_soft mode.

    Drops edges only when ALL of the following hold:
    - The relation_type is NOT in the ontology (no exact/near-miss match was applied).
    - The normalized relation_type is a known generic/connector type with no signal.

    Comparison is performed on the normalized (SCREAMING_SNAKE_CASE) form so
    that mixed-case LLM outputs like ``'relates to'`` or ``'Mentions'`` are
    caught by the filter even if the caller did not pre-normalize.

    Keeps domain-specific off-ontology edges (they may still carry value).

    Parameters
    ----------
    relation_type:
        Post-canonicalization relation type string (may already be normalized).
    ontology_names:
        Set of canonical relation type names from the lane ontology.
    """
    # Normalize defensively — callers may pass already-normalized strings, no-op then.
    normalized = _normalize_relation_type(relation_type)

    if normalized in ontology_names:
        return False  # ontology match → keep
    if normalized in _GENERIC_EDGE_NAMES:
        logger.debug(
            'constrained_soft: dropping generic off-ontology edge %r',
            relation_type,
        )
        return True  # generic noise → drop
    return False  # specific off-ontology → allow (limited)


# ---------------------------------------------------------------------------
# Public API alias (no leading underscore) for use by external maintenance
# scripts and downstream tooling.
# ---------------------------------------------------------------------------

#: Public alias for :func:`_normalize_relation_type`.
#: Normalizes a relation type string to canonical SCREAMING_SNAKE_CASE.
#: Safe to call from offline maintenance scripts and ingestion helpers.
normalize_relation_type = _normalize_relation_type


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
    episode: EpisodicNode,
    nodes: list[EntityNode],
    previous_episodes: list[EpisodicNode],
    edge_type_map: dict[tuple[str, str], list[str]],
    group_id: str = '',
    edge_types: dict[str, type[BaseModel]] | None = None,
    custom_extraction_instructions: str | None = None,
    extraction_mode: str = 'permissive',
) -> list[EntityEdge]:
    """Extract relationship edges from an episode.

    Parameters
    ----------
    extraction_mode : str
        Extraction behaviour: ``'permissive'`` (default — extract broadly) or
        ``'constrained_soft'`` (ontology-conformant — after LLM extraction,
        near-miss relation types are canonicalized to ontology names and
        generic off-ontology noise is dropped).
    """
    start = time()

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

    # Build name-to-node mapping for validation
    name_to_node: dict[str, EntityNode] = {node.name: node for node in nodes}

    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'nodes': [{'name': node.name, 'entity_types': node.labels} for node in nodes],
        'previous_episodes': [ep.content for ep in previous_episodes],
        'reference_time': episode.valid_at,
        'edge_types': edge_types_context,
        'custom_extraction_instructions': custom_extraction_instructions or '',
        'extraction_mode': extraction_mode,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_edges.edge(context),
        response_model=ExtractedEdges,
        max_tokens=extract_edges_max_tokens,
        group_id=group_id,
        prompt_name='extract_edges.edge',
    )
    all_edges_data = ExtractedEdges(**llm_response).edges

    # Validate entity names
    edges_data: list[ExtractedEdge] = []
    for edge_data in all_edges_data:
        source_name = edge_data.source_entity_name
        target_name = edge_data.target_entity_name

        # Validate LLM-returned names exist in the nodes list
        if source_name not in name_to_node:
            logger.warning(
                'Source entity not found in nodes for edge relation: %s',
                edge_data.relation_type,
            )
            continue

        if target_name not in name_to_node:
            logger.warning(
                'Target entity not found in nodes for edge relation: %s',
                edge_data.relation_type,
            )
            continue

        edges_data.append(edge_data)

    # -----------------------------------------------------------------------
    # Universal SCREAMING_SNAKE_CASE normalization (all extraction modes)
    # -----------------------------------------------------------------------
    # Regardless of mode, always normalize edge relation_type to canonical
    # SCREAMING_SNAKE_CASE before storing.  This is a cosmetic-only transform:
    # it collapses whitespace/hyphens/punctuation into underscores and
    # uppercases — it never drops an edge or changes its semantic meaning.
    # Downstream dedup and search benefit from a consistent representation.
    for edge_data in edges_data:
        norm = _normalize_relation_type(edge_data.relation_type)
        if norm != edge_data.relation_type:
            logger.debug(
                'edge_norm: normalized relation_type %r → %r',
                edge_data.relation_type,
                norm,
            )
            edge_data.relation_type = norm

    # -----------------------------------------------------------------------
    # constrained_soft post-extraction enforcement
    # -----------------------------------------------------------------------
    # After universal normalization, apply constrained_soft-specific passes:
    #   1. Canonicalize near-miss relation types to ontology names.
    #   2. Filter generic off-ontology noise.
    # This is intentionally done in code (not prompt) to avoid conflicting
    # directives and ensure deterministic enforcement regardless of LLM drift.
    if extraction_mode == 'constrained_soft' and edge_types is not None:
        ontology_names: frozenset[str] = frozenset(edge_types.keys())
        enforced: list[ExtractedEdge] = []
        for edge_data in edges_data:
            canonical = _canonicalize_edge_name(edge_data.relation_type, ontology_names)
            edge_data.relation_type = canonical
            if _should_filter_constrained_edge(canonical, ontology_names):
                logger.info(
                    'constrained_soft: dropped generic edge %r between %r → %r',
                    canonical,
                    edge_data.source_entity_name,
                    edge_data.target_entity_name,
                )
                continue
            enforced.append(edge_data)
        dropped = len(edges_data) - len(enforced)
        if dropped:
            logger.info(
                'constrained_soft enforcement: kept %d/%d edges (dropped %d generic/noise)',
                len(enforced),
                len(edges_data),
                dropped,
            )
        edges_data = enforced

    end = time()
    logger.debug(f'Extracted {len(edges_data)} new edges in {(end - start) * 1000:.0f} ms')

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

        # Filter out empty edges
        if not edge_data.fact.strip():
            continue

        # Names already validated above
        source_node = name_to_node.get(edge_data.source_entity_name)
        target_node = name_to_node.get(edge_data.target_entity_name)

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
            f'Created new edge {edge.uuid} from {edge.source_node_uuid} to {edge.target_node_uuid}'
        )

    logger.debug(f'Extracted edges: {[e.uuid for e in edges]}')

    return edges


async def resolve_extracted_edges(
    clients: GraphitiClients,
    extracted_edges: list[EntityEdge],
    episode: EpisodicNode,
    entities: list[EntityNode],
    edge_types: dict[str, type[BaseModel]],
    edge_type_map: dict[tuple[str, str], list[str]],
    dedupe_mode: Literal['semantic', 'deterministic'] = 'semantic',
) -> tuple[list[EntityEdge], list[EntityEdge], list[EntityEdge]]:
    """Resolve extracted edges against existing graph context.

    Returns
    -------
    tuple[list[EntityEdge], list[EntityEdge], list[EntityEdge]]
        A tuple of (resolved_edges, invalidated_edges, new_edges) where:
        - resolved_edges: All edges after resolution (may include existing edges if duplicates found)
        - invalidated_edges: Edges that were invalidated/contradicted by new information
        - new_edges: Only edges that are new to the graph (not duplicates of existing edges)
    """
    # Fast path: deduplicate exact matches within the extracted edges before parallel processing
    seen: dict[tuple[str, str, str], EntityEdge] = {}
    deduplicated_edges: list[EntityEdge] = []

    for edge in extracted_edges:
        key = (
            edge.source_node_uuid,
            edge.target_node_uuid,
            _normalize_string_exact(edge.fact),
        )
        if key not in seen:
            seen[key] = edge
            deduplicated_edges.append(edge)

    extracted_edges = deduplicated_edges

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

    # Remove duplicates: if an edge appears in both duplicate candidates and invalidation candidates,
    # keep it only in duplicate candidates
    edge_invalidation_candidates: list[list[EntityEdge]] = []
    for related_edges, invalidation_result in zip(
        related_edges_lists, edge_invalidation_candidate_results, strict=True
    ):
        related_uuids = {edge.uuid for edge in related_edges}
        deduplicated = [
            edge for edge in invalidation_result.edges if edge.uuid not in related_uuids
        ]
        edge_invalidation_candidates.append(deduplicated)

    logger.debug(
        f'Related edges: {[e.uuid for edges_lst in related_edges_lists for e in edges_lst]}'
    )

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
        missing_nodes = await EntityNode.get_by_uuids(driver, list(referenced_node_uuids))
        for node in missing_nodes:
            uuid_entity_map[node.uuid] = node

    # Determine which edge types are relevant for each edge based on node signatures.
    # `edge_types_lst` stores the subset of custom edge definitions whose
    # node signature matches each extracted edge.
    edge_types_lst: list[dict[str, type[BaseModel]]] = []
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
    # Keep semantic mode call signature backward-compatible for test monkeypatches.
    if dedupe_mode == 'semantic':
        resolve_coros = [
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
    else:
        resolve_coros = [
            resolve_extracted_edge(
                llm_client,
                extracted_edge,
                related_edges,
                existing_edges,
                episode,
                extracted_edge_types,
                dedupe_mode=dedupe_mode,
            )
            for extracted_edge, related_edges, existing_edges, extracted_edge_types in zip(
                extracted_edges,
                related_edges_lists,
                edge_invalidation_candidates,
                edge_types_lst,
                strict=True,
            )
        ]

    results: list[tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]] = list(
        await semaphore_gather(*resolve_coros)
    )

    resolved_edges: list[EntityEdge] = []
    invalidated_edges: list[EntityEdge] = []
    new_edges: list[EntityEdge] = []
    for extracted_edge, result in zip(extracted_edges, results, strict=True):
        resolved_edge = result[0]
        invalidated_edge_chunk = result[1]
        # result[2] is duplicate_edges list

        resolved_edges.append(resolved_edge)
        invalidated_edges.extend(invalidated_edge_chunk)

        # Track edges that are new (not duplicates of existing edges)
        # An edge is new if the resolved edge UUID matches the extracted edge UUID
        if resolved_edge.uuid == extracted_edge.uuid:
            new_edges.append(resolved_edge)

    logger.debug(f'Resolved edges: {[e.uuid for e in resolved_edges]}')
    logger.debug(f'New edges (non-duplicates): {[e.uuid for e in new_edges]}')

    await semaphore_gather(
        create_entity_edge_embeddings(embedder, resolved_edges),
        create_entity_edge_embeddings(embedder, invalidated_edges),
    )

    return resolved_edges, invalidated_edges, new_edges


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


async def resolve_extracted_edge(
    llm_client: LLMClient,
    extracted_edge: EntityEdge,
    related_edges: list[EntityEdge],
    existing_edges: list[EntityEdge],
    episode: EpisodicNode,
    edge_type_candidates: dict[str, type[BaseModel]] | None = None,
    dedupe_mode: Literal['semantic', 'deterministic'] = 'semantic',
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

    Returns
    -------
    tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]
        The resolved edge, any duplicates, and edges to invalidate.
    """
    if len(related_edges) == 0 and len(existing_edges) == 0:
        # Still extract custom attributes even when no dedup/invalidation is needed
        edge_model = (
            edge_type_candidates.get(extracted_edge.name) if edge_type_candidates else None
        )
        if edge_model is not None and len(edge_model.model_fields) != 0:
            edge_attributes_context = {
                'fact': extracted_edge.fact,
                'reference_time': episode.valid_at if episode is not None else None,
                'existing_attributes': extracted_edge.attributes,
            }
            edge_attributes_response = await llm_client.generate_response(
                prompt_library.extract_edges.extract_attributes(edge_attributes_context),
                response_model=edge_model,  # type: ignore
                model_size=ModelSize.small,
                prompt_name='extract_edges.extract_attributes',
            )
            extracted_edge.attributes = edge_attributes_response

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

    # Migration-only deterministic dedupe mode:
    # Skip LLM duplicate/contradiction resolution and keep exact-match-only behavior.
    # This mode is intended for controlled backfills where semantic dedupe may be unstable.
    if dedupe_mode == 'deterministic':
        resolved_edge = extracted_edge

        # Preserve optional structured attribute extraction for allowed edge types.
        edge_model = edge_type_candidates.get(resolved_edge.name) if edge_type_candidates else None
        if edge_model is not None and len(edge_model.model_fields) != 0 and episode is not None:
            edge_attributes_context = {
                'episode_content': episode.content,
                'reference_time': episode.valid_at,
                'fact': resolved_edge.fact,
            }
            edge_attributes_response = await llm_client.generate_response(
                prompt_library.extract_edges.extract_attributes(edge_attributes_context),
                response_model=edge_model,  # type: ignore
                model_size=ModelSize.small,
                prompt_name='extract_edges.extract_attributes',
            )
            resolved_edge.attributes = edge_attributes_response
        else:
            resolved_edge.attributes = {}

        logger.info(
            'resolve_extracted_edge: dedupe_mode=deterministic, skipping semantic dedupe for edge %s',
            resolved_edge.name,
        )
        return resolved_edge, [], []

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

    llm_response = await llm_client.generate_response(
        prompt_library.dedupe_edges.resolve_edge(context),
        response_model=EdgeDuplicate,
        model_size=ModelSize.small,
        prompt_name='dedupe_edges.resolve_edge',
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

        edge_attributes_response = await llm_client.generate_response(
            prompt_library.extract_edges.extract_attributes(edge_attributes_context),
            response_model=edge_model,  # type: ignore
            model_size=ModelSize.small,
            prompt_name='extract_edges.extract_attributes',
        )

        resolved_edge.attributes = edge_attributes_response
    else:
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
        invalidation_candidates.sort(key=lambda c: (c.valid_at is None, ensure_utc(c.valid_at)))
        for candidate in invalidation_candidates:
            candidate_valid_at_utc = ensure_utc(candidate.valid_at)
            resolved_edge_valid_at_utc = ensure_utc(resolved_edge.valid_at)
            if (
                candidate_valid_at_utc is not None
                and resolved_edge_valid_at_utc is not None
                and candidate_valid_at_utc > resolved_edge_valid_at_utc
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
