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

import json
import logging
import typing
from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field
from typing_extensions import Any

from graphiti_core.driver.driver import (
    GraphDriver,
    GraphDriverSession,
    GraphProvider,
)
from graphiti_core.edges import Edge, EntityEdge, EpisodicEdge, create_entity_edge_embeddings
from graphiti_core.embedder import EmbedderClient
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import normalize_l2, semaphore_gather
from graphiti_core.models.edges.edge_db_queries import (
    get_entity_edge_save_bulk_query,
    get_entity_edge_save_bulk_query_by_type,
    get_episodic_edge_save_bulk_query,
)
from graphiti_core.models.nodes.node_db_queries import (
    get_entity_node_save_bulk_query,
    get_episode_node_save_bulk_query,
)
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings
from graphiti_core.utils.maintenance.dedup_helpers import (
    DedupResolutionState,
    _build_candidate_indexes,
    _normalize_string_exact,
    _resolve_with_similarity,
)
from graphiti_core.utils.maintenance.edge_operations import (
    extract_edges,
    resolve_extracted_edge,
)
from graphiti_core.utils.maintenance.graph_data_operations import (
    EPISODE_WINDOW_LEN,
    retrieve_episodes,
)
from graphiti_core.llm_client import LLMClient
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_nodes import EntityClustering, NodeResolutions
from graphiti_core.search.search import search
from graphiti_core.search.search_config import SearchConfig, NodeSearchConfig, NodeSearchMethod, NodeReranker
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.node_operations import (
    extract_nodes,
    resolve_extracted_nodes,
    _collect_candidate_nodes,
)

# Search config for dedup: limit 10 candidates per node
DEDUP_SEARCH_LIMIT = 10
DEDUP_SEARCH_CONFIG = SearchConfig(
    limit=DEDUP_SEARCH_LIMIT,
    node_config=NodeSearchConfig(
        search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
        reranker=NodeReranker.rrf,
    )
)

logger = logging.getLogger(__name__)

CHUNK_SIZE = 10


def _get_entity_type_label(node: EntityNode) -> str:
    """Get the most specific entity type label (non-'Entity') from a node."""
    for label in node.labels:
        if label != 'Entity':
            return label
    return 'Entity'


async def _search_db_candidates_by_type(
    clients: GraphitiClients,
    extracted_nodes: list[EntityNode],
    entity_type: str,
) -> list[EntityNode]:
    """Search DB for existing entities of the given type.

    EasyOps: This ensures cross-request deduplication - entities extracted in different
    requests can be deduplicated against entities already in the database.

    Args:
        clients: GraphitiClients instance
        extracted_nodes: Nodes from current extraction (to get group_id and for search query)
        entity_type: The entity type label to search for

    Returns:
        List of existing EntityNodes from the database (excluding those in extracted_nodes)
    """
    if not extracted_nodes:
        return []

    # Use the first node to get group_id
    group_id = extracted_nodes[0].group_id
    extracted_uuids = {node.uuid for node in extracted_nodes}

    # Search for each extracted node to find similar DB entities
    all_db_candidates: dict[str, EntityNode] = {}

    # Search for each extracted node's name
    search_tasks = []
    for node in extracted_nodes:
        search_tasks.append(
            search(
                clients,
                query=node.name,
                group_ids=[group_id],
                config=DEDUP_SEARCH_CONFIG,
                search_filter=SearchFilters(node_labels=[entity_type]),
            )
        )

    search_results = await semaphore_gather(*search_tasks)

    for result in search_results:
        for db_node in result.nodes:
            # Skip nodes that are in the current extracted batch
            if db_node.uuid not in extracted_uuids:
                all_db_candidates[db_node.uuid] = db_node

    if all_db_candidates:
        logger.info(
            '[semantic_dedup] Found %d DB candidates for %s type',
            len(all_db_candidates), entity_type,
        )

    return list(all_db_candidates.values())


async def _cluster_entities_with_llm(
    llm_client: LLMClient,
    nodes: list[EntityNode],
    entity_type: str,
    type_definition: str,
) -> list[tuple[str, str]]:
    """Use LLM to cluster entities of the same type, returns (source_uuid, canonical_uuid) pairs.

    EasyOps: This clustering approach avoids the self-matching problem in batch deduplication
    by asking the LLM to group entities instead of comparing them pairwise.

    Args:
        llm_client: LLM client for generating responses
        nodes: List of nodes to cluster (all same entity type)
        entity_type: The entity type label (e.g., 'ProductModule')
        type_definition: Description of the entity type (may contain alias patterns)

    Returns:
        List of (source_uuid, canonical_uuid) pairs for detected duplicates.
    """
    if len(nodes) <= 1:
        return []

    # Build entity info (only id, name, summary - type info is given once at the top)
    entities_info = []
    for i, node in enumerate(nodes):
        entities_info.append({
            'id': i,
            'name': node.name,
            'summary': node.summary or '',
        })

    context = {
        'entity_type': entity_type,
        'type_definition': type_definition,
        'entities': entities_info,
    }

    logger.info(
        '[cluster_dedup] Clustering %d %s entities with LLM',
        len(nodes), entity_type,
    )

    try:
        llm_response = await llm_client.generate_response(
            prompt_library.dedupe_nodes.cluster_entities(context),
            response_model=EntityClustering,
            prompt_name='dedupe_nodes.cluster_entities',
        )

        clustering = EntityClustering(**llm_response)

        # Convert groups to (source_uuid, canonical_uuid) pairs
        duplicate_pairs: list[tuple[str, str]] = []
        for group in clustering.groups:
            if len(group.entity_ids) <= 1:
                continue

            # Validate canonical_id
            if group.canonical_id < 0 or group.canonical_id >= len(nodes):
                logger.warning(
                    '[cluster_dedup] Invalid canonical_id %d for group %s',
                    group.canonical_id, group.entity_ids,
                )
                continue

            canonical_node = nodes[group.canonical_id]
            for entity_id in group.entity_ids:
                if entity_id == group.canonical_id:
                    continue
                if entity_id < 0 or entity_id >= len(nodes):
                    logger.warning(
                        '[cluster_dedup] Invalid entity_id %d in group',
                        entity_id,
                    )
                    continue

                source_node = nodes[entity_id]
                duplicate_pairs.append((source_node.uuid, canonical_node.uuid))
                logger.info(
                    '[cluster_dedup] Grouped: "%s" -> "%s", reason: %s',
                    source_node.name, canonical_node.name, group.reasoning or 'no reasoning',
                )

        return duplicate_pairs

    except Exception as e:
        logger.error('[cluster_dedup] LLM clustering failed: %s', e)
        return []


def _merge_node_into_canonical(source: EntityNode, canonical: EntityNode) -> None:
    """Merge source node's summary and attributes into canonical node.

    EasyOps customization: When LLM identifies duplicates, merge their information
    to preserve all extracted knowledge.
    """
    # Merge summary: concatenate if both exist, prefer non-empty
    if source.summary and canonical.summary:
        if source.summary not in canonical.summary:
            canonical.summary = f"{canonical.summary} {source.summary}"
    elif source.summary and not canonical.summary:
        canonical.summary = source.summary

    # Merge attributes: source attributes fill in missing canonical attributes
    for key, value in source.attributes.items():
        if key not in canonical.attributes or not canonical.attributes[key]:
            canonical.attributes[key] = value

    # EasyOps: Keep first reasoning for deduplicated nodes (avoid unbounded growth)
    if not canonical.reasoning and source.reasoning:
        canonical.reasoning = source.reasoning


async def semantic_dedupe_nodes_bulk(
    clients: GraphitiClients,
    nodes: list[EntityNode],
    entity_types: dict[str, type[BaseModel]] | None = None,
    batch_size: int = 10,
) -> tuple[list[EntityNode], dict[str, str]]:
    """Semantic deduplication using clustering approach.

    EasyOps optimized design:
    - Groups nodes by entity type
    - Searches DB for existing entities of each type (cross-request dedup)
    - Uses LLM clustering to find duplicates (avoids self-matching problem)
    - Batches large type groups and reduces across batches

    Args:
        clients: GraphitiClients instance
        nodes: List of extracted nodes to deduplicate
        entity_types: Entity type definitions for context
        batch_size: Max nodes per LLM clustering call

    Returns:
        Tuple of (deduplicated nodes, uuid_map from duplicates to canonical).
    """
    logger.info(
        '[semantic_dedup] Starting clustering dedup: %d nodes, %d entity_types, batch_size=%d',
        len(nodes),
        len(entity_types) if entity_types else 0,
        batch_size,
    )

    if not nodes:
        logger.info('[semantic_dedup] No nodes to process, returning early')
        return nodes, {}

    entity_types_dict = entity_types or {}

    # Group nodes by entity type
    nodes_by_type: dict[str, list[EntityNode]] = {}
    for node in nodes:
        entity_type = _get_entity_type_label(node)
        if entity_type not in nodes_by_type:
            nodes_by_type[entity_type] = []
        nodes_by_type[entity_type].append(node)

    # Track nodes by uuid for merging
    nodes_by_uuid: dict[str, EntityNode] = {node.uuid: node for node in nodes}

    all_duplicate_pairs: list[tuple[str, str]] = []  # (source_uuid, canonical_uuid)

    # Process each entity type (can be parallelized across types)
    type_tasks = []
    for entity_type, type_nodes in nodes_by_type.items():
        # Note: Even with 1 node, we still search DB for potential duplicates
        # Get type definition (may contain alias patterns)
        type_model = entity_types_dict.get(entity_type)
        type_definition = type_model.__doc__ if type_model else ''

        type_tasks.append(
            _cluster_entity_type_with_batching(
                clients,  # Pass full clients for DB search
                type_nodes,
                entity_type,
                type_definition,
                batch_size,
            )
        )

    # Process all entity types in parallel
    if type_tasks:
        results = await semaphore_gather(*type_tasks)
        for duplicate_pairs in results:
            all_duplicate_pairs.extend(duplicate_pairs)

    # Build final uuid_map using union-find to handle chains
    uuid_map = _build_uuid_map_from_pairs(all_duplicate_pairs)

    # Apply merges for duplicates (among extracted nodes)
    for source_uuid, canonical_uuid in all_duplicate_pairs:
        if source_uuid in nodes_by_uuid and canonical_uuid in nodes_by_uuid:
            source_node = nodes_by_uuid[source_uuid]
            canonical_node = nodes_by_uuid[canonical_uuid]
            _merge_node_into_canonical(source_node, canonical_node)

    # Filter out duplicates from final list
    final_deduped_nodes = [node for node in nodes if node.uuid not in uuid_map]

    if uuid_map:
        logger.info(
            '[semantic_dedup] Completed: %d duplicates found, %d unique nodes remain',
            len(uuid_map), len(final_deduped_nodes),
        )

    return final_deduped_nodes, uuid_map


def _redirect_to_db_canonical(
    pairs: list[tuple[str, str]],
    extracted_uuids: set[str],
    db_uuids: set[str],
) -> list[tuple[str, str]]:
    """Redirect pairs to use DB node as canonical when available.

    EasyOps fix: When LLM chooses an extracted node as canonical but a DB node
    exists in the same group (indicated by DB->extracted pair), we should
    redirect all pairs pointing to that extracted node to point to the DB node
    instead. This ensures extracted nodes are properly deduplicated against
    existing DB entities.

    Example:
    - Input pairs: [(ext1, ext2), (db1, ext2)]  # LLM chose ext2 as canonical
    - Output pairs: [(ext1, db1), (db1, db1)]   # Redirect to db1
    - After filtering (only keep extracted->*): [(ext1, db1)]
    """
    if not pairs:
        return pairs

    # Find extracted nodes that should be redirected to DB nodes
    # If (db_uuid, extracted_uuid) exists, it means they're the same entity
    # but LLM wrongly chose extracted as canonical
    db_for_extracted: dict[str, str] = {}
    for src, tgt in pairs:
        if src in db_uuids and tgt in extracted_uuids:
            # DB node points to extracted node - remember this mapping
            # If multiple DB nodes point to same extracted, use the first one
            if tgt not in db_for_extracted:
                db_for_extracted[tgt] = src
                logger.info(
                    '[redirect_canonical] Redirecting extracted canonical "%s" to DB node "%s"',
                    tgt, src,
                )

    if not db_for_extracted:
        return pairs

    # Redirect all pairs to use DB node as canonical
    redirected_pairs: list[tuple[str, str]] = []
    for src, tgt in pairs:
        new_tgt = db_for_extracted.get(tgt, tgt)
        if src != new_tgt:  # Don't create self-loop
            redirected_pairs.append((src, new_tgt))

    # IMPORTANT: Add pairs for extracted canonicals that got redirected
    # These nodes were chosen as canonical by LLM but need to be mapped to DB nodes
    # Without this, the extracted canonical would not be in uuid_map
    for ext_canonical, db_canonical in db_for_extracted.items():
        pair = (ext_canonical, db_canonical)
        if pair not in redirected_pairs:
            redirected_pairs.append(pair)
            logger.info(
                '[redirect_canonical] Added pair for redirected canonical: "%s" -> "%s"',
                ext_canonical, db_canonical,
            )

    return redirected_pairs


def _select_group_representative(
    group_uuids: list[str],
    nodes_by_uuid: dict[str, EntityNode],
    db_uuids: set[str],
) -> str:
    """Select a representative from a group of duplicate nodes.

    Priority: DB node > extracted node (by uuid for determinism)
    """
    # Prefer DB nodes
    db_nodes_in_group = [uuid for uuid in group_uuids if uuid in db_uuids]
    if db_nodes_in_group:
        return min(db_nodes_in_group)  # Deterministic selection

    # Otherwise pick the first extracted node (by uuid)
    return min(group_uuids)


def _get_group_representatives(
    pairs: list[tuple[str, str]],
    all_uuids: set[str],
    nodes_by_uuid: dict[str, EntityNode],
    db_uuids: set[str],
) -> list[EntityNode]:
    """Get one representative per group from clustering results.

    Uses union-find to identify groups, then selects one representative per group.
    DB nodes are preferred as representatives.
    """
    # Build groups using union-find
    parent: dict[str, str] = {uuid: uuid for uuid in all_uuids}

    def find(x: str) -> str:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    for src, tgt in pairs:
        if src in parent and tgt in parent:
            union(src, tgt)

    # Group nodes by their root
    groups: dict[str, list[str]] = {}
    for uuid in all_uuids:
        root = find(uuid)
        if root not in groups:
            groups[root] = []
        groups[root].append(uuid)

    # Select representative for each group
    representatives: list[EntityNode] = []
    for group_uuids in groups.values():
        rep_uuid = _select_group_representative(group_uuids, nodes_by_uuid, db_uuids)
        if rep_uuid in nodes_by_uuid:
            representatives.append(nodes_by_uuid[rep_uuid])

    return representatives


async def _recursive_reduce(
    llm_client: LLMClient,
    embedder: EmbedderClient,
    nodes: list[EntityNode],
    entity_type: str,
    type_definition: str,
    batch_size: int,
    db_uuids: set[str],
    nodes_by_uuid: dict[str, EntityNode],
    depth: int = 0,
) -> list[tuple[str, str]]:
    """Recursively reduce nodes until batch_size or fewer remain.

    Each reduce round:
    1. Use embedding clustering to group similar nodes
    2. Cluster each batch with LLM
    3. Select one representative per group (DB priority)
    4. If representatives > batch_size, recurse
    """
    if len(nodes) <= 1:
        return []

    if len(nodes) <= batch_size:
        # Final clustering - small enough for single LLM call
        logger.info(
            '[semantic_dedup] Reduce depth=%d: final clustering %d %s entities',
            depth, len(nodes), entity_type,
        )
        return await _cluster_entities_with_llm(
            llm_client, nodes, entity_type, type_definition
        )

    # Use embedding clustering instead of sequential splitting
    batches = await _cluster_by_embedding(embedder, nodes, batch_size)

    logger.info(
        '[semantic_dedup] Reduce depth=%d: clustering %d %s entities in %d embedding clusters',
        depth, len(nodes), entity_type, len(batches),
    )

    # Cluster each batch in parallel
    batch_tasks = [
        _cluster_entities_with_llm(llm_client, batch, entity_type, type_definition)
        for batch in batches
    ]
    batch_results = await semaphore_gather(*batch_tasks)

    # Collect pairs from this round
    round_pairs: list[tuple[str, str]] = []
    for pairs in batch_results:
        round_pairs.extend(pairs)

    # Get representatives for next round
    current_uuids = {node.uuid for node in nodes}
    representatives = _get_group_representatives(
        round_pairs, current_uuids, nodes_by_uuid, db_uuids
    )

    logger.info(
        '[semantic_dedup] Reduce depth=%d: %d groups -> %d representatives',
        depth, len(nodes), len(representatives),
    )

    # Recurse if still too many representatives
    if len(representatives) > batch_size:
        next_round_pairs = await _recursive_reduce(
            llm_client, embedder, representatives, entity_type, type_definition,
            batch_size, db_uuids, nodes_by_uuid, depth + 1,
        )
        round_pairs.extend(next_round_pairs)

    elif len(representatives) > 1:
        # Final reduce of representatives
        final_pairs = await _cluster_entities_with_llm(
            llm_client, representatives, entity_type, type_definition
        )
        round_pairs.extend(final_pairs)

    return round_pairs


async def _cluster_by_embedding(
    embedder: EmbedderClient,
    nodes: list[EntityNode],
    max_cluster_size: int,
) -> list[list[EntityNode]]:
    """Cluster nodes by embedding similarity using K-means.

    EasyOps optimization: Instead of sequential splitting, group similar entities together.
    This ensures:
    - "Hong Kong", "Lung Fu Shan", "Mid-levels" are in DIFFERENT clusters (dissimilar)
    - "UK" and "United Kingdom" are in the SAME cluster (similar)

    Each cluster is then processed by LLM for fine-grained deduplication.

    Args:
        embedder: Embedding client
        nodes: Nodes to cluster
        max_cluster_size: Maximum nodes per cluster

    Returns:
        List of node clusters
    """
    if len(nodes) <= max_cluster_size:
        return [nodes]

    # Ensure all nodes have embeddings
    nodes_without_embedding = [n for n in nodes if n.name_embedding is None]
    if nodes_without_embedding:
        texts = [n.name for n in nodes_without_embedding]
        embeddings = await embedder.create(texts)
        for node, emb in zip(nodes_without_embedding, embeddings):
            node.name_embedding = normalize_l2(emb)

    # Build embedding matrix
    embedding_dim = len(nodes[0].name_embedding)
    X = np.array([n.name_embedding for n in nodes])

    # Determine number of clusters
    n_clusters = max(2, (len(nodes) + max_cluster_size - 1) // max_cluster_size)

    # K-means clustering
    labels = _kmeans_cluster(X, n_clusters, max_iter=20)

    # Group nodes by cluster label
    clusters: dict[int, list[EntityNode]] = {}
    for node, label in zip(nodes, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(node)

    # Handle oversized clusters by splitting
    result: list[list[EntityNode]] = []
    for cluster_nodes in clusters.values():
        if len(cluster_nodes) > max_cluster_size:
            # Recursively split large clusters
            sub_clusters = await _cluster_by_embedding(embedder, cluster_nodes, max_cluster_size)
            result.extend(sub_clusters)
        else:
            result.append(cluster_nodes)

    logger.info(
        '[embedding_cluster] Grouped %d nodes into %d clusters (target size: %d)',
        len(nodes), len(result), max_cluster_size,
    )

    return result


def _kmeans_cluster(X: np.ndarray, n_clusters: int, max_iter: int = 20) -> list[int]:
    """Simple K-means clustering implementation.

    Avoids sklearn dependency. Uses K-means++ initialization.
    """
    n_samples = X.shape[0]

    if n_clusters >= n_samples:
        return list(range(n_samples))

    # K-means++ initialization
    centers = [X[np.random.randint(n_samples)]]
    for _ in range(1, n_clusters):
        # Compute distances to nearest center
        dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        # Sample proportional to distance squared
        probs = dists / dists.sum()
        idx = np.random.choice(n_samples, p=probs)
        centers.append(X[idx])
    centers = np.array(centers)

    # K-means iterations
    labels = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        # Assign to nearest center
        dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centers])
        new_labels = np.argmin(dists, axis=0)

        # Check convergence
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        # Update centers
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                centers[k] = X[mask].mean(axis=0)

    return labels.tolist()


async def _cluster_entity_type_with_batching(
    clients: GraphitiClients,
    type_nodes: list[EntityNode],
    entity_type: str,
    type_definition: str,
    batch_size: int,
) -> list[tuple[str, str]]:
    """Cluster nodes of a single entity type, with batching and recursive reduce.

    EasyOps: Now includes DB candidate search for cross-request deduplication.

    Flow:
    1. Search DB for existing entities of this type
    2. Combine extracted nodes with DB candidates
    3. Cluster the combined list
    4. Return pairs where source is an extracted node (DB nodes are only targets)

    Map phase: Split into batches, cluster each batch
    Reduce phase: Recursively cluster representatives until batch_size or fewer
    """
    extracted_uuids = {node.uuid for node in type_nodes}

    # Step 1: Search DB for existing entities of this type
    db_candidates = await _search_db_candidates_by_type(
        clients, type_nodes, entity_type
    )
    db_uuids = {node.uuid for node in db_candidates}

    # Combine extracted nodes with DB candidates
    all_nodes = type_nodes + db_candidates
    nodes_by_uuid = {node.uuid: node for node in all_nodes}

    logger.info(
        '[semantic_dedup] Clustering %s: %d extracted + %d DB candidates = %d total',
        entity_type, len(type_nodes), len(db_candidates), len(all_nodes),
    )

    if len(all_nodes) <= 1:
        return []

    # If small enough, single LLM call
    if len(all_nodes) <= batch_size:
        logger.info(
            '[semantic_dedup] Clustering %d %s entities (single batch)',
            len(all_nodes), entity_type,
        )
        all_pairs = await _cluster_entities_with_llm(
            clients.llm_client, all_nodes, entity_type, type_definition
        )
        # Redirect extracted canonical to DB node when both exist in same group
        all_pairs = _redirect_to_db_canonical(all_pairs, extracted_uuids, db_uuids)
        # Filter: only return pairs where source is an extracted node
        return [(src, tgt) for src, tgt in all_pairs if src in extracted_uuids]

    # Map phase: use embedding clustering instead of sequential splitting
    # This groups similar entities together, so:
    # - "Hong Kong", "Mid-levels", "Lung Fu Shan" go to DIFFERENT clusters (dissimilar)
    # - "UK" and "United Kingdom" go to the SAME cluster (similar)
    batches = await _cluster_by_embedding(clients.embedder, all_nodes, batch_size)

    logger.info(
        '[semantic_dedup] Map phase: %d %s entities in %d embedding clusters',
        len(all_nodes), entity_type, len(batches),
    )

    # Cluster each batch in parallel
    batch_tasks = [
        _cluster_entities_with_llm(clients.llm_client, batch, entity_type, type_definition)
        for batch in batches
    ]
    batch_results = await semaphore_gather(*batch_tasks)

    # Collect all duplicate pairs from map phase
    all_pairs: list[tuple[str, str]] = []
    for pairs in batch_results:
        all_pairs.extend(pairs)

    # Get representatives for reduce phase (one per group, DB priority)
    all_uuids = {node.uuid for node in all_nodes}
    representatives = _get_group_representatives(
        all_pairs, all_uuids, nodes_by_uuid, db_uuids
    )

    logger.info(
        '[semantic_dedup] Map phase complete: %d entities -> %d representatives',
        len(all_nodes), len(representatives),
    )

    # Reduce phase: single LLM call for all representatives
    # Since embedding clustering already grouped similar entities in Map phase,
    # representatives from different clusters are likely different entities.
    # We just need one final LLM call to catch any cross-cluster duplicates.
    if len(representatives) > 1:
        logger.info(
            '[semantic_dedup] Reduce phase: clustering %d representatives',
            len(representatives),
        )
        reduce_pairs = await _cluster_entities_with_llm(
            clients.llm_client, representatives, entity_type, type_definition
        )
        all_pairs.extend(reduce_pairs)

    # Redirect extracted canonical to DB node when both exist in same group
    all_pairs = _redirect_to_db_canonical(all_pairs, extracted_uuids, db_uuids)

    # Filter: only return pairs where source is an extracted node (not a DB node)
    filtered_pairs = [(src, tgt) for src, tgt in all_pairs if src in extracted_uuids]

    if len(filtered_pairs) != len(all_pairs):
        logger.info(
            '[semantic_dedup] Filtered %d pairs to %d (removed DB->* pairs)',
            len(all_pairs), len(filtered_pairs),
        )

    return filtered_pairs


def _build_uuid_map_from_pairs(duplicate_pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Build uuid_map using union-find to handle chains.

    Handles:
    - A->B, B->C becomes A->C, B->C
    - A->B, A->C keeps first (A->B)
    """
    if not duplicate_pairs:
        return {}

    logger.info('[build_uuid_map] Processing %d duplicate pairs', len(duplicate_pairs))

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    # Process pairs: source points to canonical
    for source, canonical in duplicate_pairs:
        ps, pc = find(source), find(canonical)
        if ps != pc:
            parent[ps] = pc  # source's root points to canonical's root

    # Build final map
    uuid_map: dict[str, str] = {}
    for source, _ in duplicate_pairs:
        root = find(source)
        if root != source:
            uuid_map[source] = root

    return uuid_map


def _sanitize_string_for_falkordb(value: str) -> str:
    """Sanitize string content for FalkorDB query parameters.

    FalkorDB's stringify_param_value only escapes backslashes and double quotes,
    but control characters (newlines, carriage returns, tabs, etc.) can break
    the Cypher query parsing. This function escapes these characters.
    """
    if not isinstance(value, str):
        return value
    # Escape control characters that can break FalkorDB query parsing
    # Note: backslash must be escaped first to avoid double-escaping
    value = value.replace('\\', '\\\\')
    value = value.replace('\n', '\\n')
    value = value.replace('\r', '\\r')
    value = value.replace('\t', '\\t')
    value = value.replace('\0', '')  # Remove null bytes entirely
    return value


def _sanitize_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    """Sanitize attributes to only include primitive types that FalkorDB supports.

    FalkorDB only supports primitive types (str, int, float, bool) or arrays of primitive types
    as property values. This function filters out any non-primitive values.
    """
    if not attributes:
        return {}

    sanitized: dict[str, Any] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, list):
            # Only keep arrays of primitive types
            if all(isinstance(item, (str, int, float, bool)) for item in value):
                sanitized[key] = value
            else:
                logger.warning(
                    f'Skipping attribute {key}: array contains non-primitive types'
                )
        else:
            logger.warning(
                f'Skipping attribute {key}: value type {type(value).__name__} is not supported by FalkorDB'
            )
    return sanitized


def _build_directed_uuid_map(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Collapse alias -> canonical chains while preserving direction.

    The incoming pairs represent directed mappings discovered during node dedupe. We use a simple
    union-find with iterative path compression to ensure every source UUID resolves to its ultimate
    canonical target, even if aliases appear lexicographically smaller than the canonical UUID.
    """

    parent: dict[str, str] = {}

    def find(uuid: str) -> str:
        """Directed union-find lookup using iterative path compression."""
        parent.setdefault(uuid, uuid)
        root = uuid
        while parent[root] != root:
            root = parent[root]

        while parent[uuid] != root:
            next_uuid = parent[uuid]
            parent[uuid] = root
            uuid = next_uuid

        return root

    for source_uuid, target_uuid in pairs:
        parent.setdefault(source_uuid, source_uuid)
        parent.setdefault(target_uuid, target_uuid)
        parent[find(source_uuid)] = find(target_uuid)

    return {uuid: find(uuid) for uuid in parent}


class RawEpisode(BaseModel):
    name: str
    uuid: str | None = Field(default=None)
    content: str
    source_description: str
    source: EpisodeType
    reference_time: datetime
    user_id: str | None = Field(default=None)


async def retrieve_previous_episodes_bulk(
    driver: GraphDriver, episodes: list[EpisodicNode]
) -> list[tuple[EpisodicNode, list[EpisodicNode]]]:
    previous_episodes_list = await semaphore_gather(
        *[
            retrieve_episodes(
                driver, episode.valid_at, last_n=EPISODE_WINDOW_LEN, group_ids=[episode.group_id], user_ids=[episode.user_id] if episode.user_id else None
            )
            for episode in episodes
        ]
    )
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]] = [
        (episode, previous_episodes_list[i]) for i, episode in enumerate(episodes)
    ]

    return episode_tuples


async def add_nodes_and_edges_bulk(
    driver: GraphDriver,
    episodic_nodes: list[EpisodicNode],
    episodic_edges: list[EpisodicEdge],
    entity_nodes: list[EntityNode],
    entity_edges: list[EntityEdge],
    embedder: EmbedderClient,
):
    session = driver.session()
    try:
        await session.execute_write(
            add_nodes_and_edges_bulk_tx,
            episodic_nodes,
            episodic_edges,
            entity_nodes,
            entity_edges,
            embedder,
            driver=driver,
        )
    finally:
        await session.close()


async def add_nodes_and_edges_bulk_tx(
    tx: GraphDriverSession,
    episodic_nodes: list[EpisodicNode],
    episodic_edges: list[EpisodicEdge],
    entity_nodes: list[EntityNode],
    entity_edges: list[EntityEdge],
    embedder: EmbedderClient,
    driver: GraphDriver,
):
    episodes = [dict(episode) for episode in episodic_nodes]
    for episode in episodes:
        episode['source'] = str(episode['source'].value)
        episode.pop('labels', None)
        # Sanitize string fields to prevent FalkorDB query parsing issues
        if driver.provider == GraphProvider.FALKORDB:
            if 'content' in episode and episode['content']:
                episode['content'] = _sanitize_string_for_falkordb(episode['content'])
            if 'name' in episode and episode['name']:
                episode['name'] = _sanitize_string_for_falkordb(episode['name'])
            if 'source_description' in episode and episode['source_description']:
                episode['source_description'] = _sanitize_string_for_falkordb(episode['source_description'])

    # Prepare episodes for save (handles external content storage if driver supports it)
    episodes = [await driver.prepare_episode_for_save(ep) for ep in episodes]

    nodes = []

    for node in entity_nodes:
        if node.name_embedding is None:
            await node.generate_name_embedding(embedder)

        # Sanitize string fields for FalkorDB
        name = node.name
        summary = node.summary
        reasoning = node.reasoning
        if driver.provider == GraphProvider.FALKORDB:
            name = _sanitize_string_for_falkordb(name) if name else name
            summary = _sanitize_string_for_falkordb(summary) if summary else summary
            reasoning = _sanitize_string_for_falkordb(reasoning) if reasoning else reasoning

        entity_data: dict[str, Any] = {
            'uuid': node.uuid,
            'name': name,
            'group_id': node.group_id,
            'summary': summary,
            'created_at': node.created_at,
            'name_embedding': node.name_embedding,
            'summary_embedding': node.summary_embedding,  # EasyOps: 双向量策略
            'labels': list(set(node.labels + ['Entity'])),
            'reasoning': reasoning,
            # EasyOps: Save type classification scores (same as nodes.py save())
            'type_scores': json.dumps(node.type_scores) if node.type_scores else None,
            'type_confidence': node.type_confidence,
        }

        if driver.provider == GraphProvider.KUZU:
            attributes = convert_datetimes_to_strings(node.attributes) if node.attributes else {}
            entity_data['attributes'] = json.dumps(attributes)
        else:
            # Sanitize attributes to only include primitive types (FalkorDB requirement)
            entity_data.update(_sanitize_attributes(node.attributes))

        nodes.append(entity_data)

    edges = []
    for edge in entity_edges:
        if edge.fact_embedding is None:
            await edge.generate_embedding(embedder)

        # Sanitize string fields for FalkorDB
        edge_name = edge.name
        edge_fact = edge.fact
        if driver.provider == GraphProvider.FALKORDB:
            edge_name = _sanitize_string_for_falkordb(edge_name) if edge_name else edge_name
            edge_fact = _sanitize_string_for_falkordb(edge_fact) if edge_fact else edge_fact

        edge_data: dict[str, Any] = {
            'uuid': edge.uuid,
            'source_node_uuid': edge.source_node_uuid,
            'target_node_uuid': edge.target_node_uuid,
            'name': edge_name,
            'fact': edge_fact,
            'group_id': edge.group_id,
            'episodes': edge.episodes,
            'created_at': edge.created_at,
            'expired_at': edge.expired_at,
            'valid_at': edge.valid_at,
            'invalid_at': edge.invalid_at,
            'fact_embedding': edge.fact_embedding,
        }

        if driver.provider == GraphProvider.KUZU:
            attributes = convert_datetimes_to_strings(edge.attributes) if edge.attributes else {}
            edge_data['attributes'] = json.dumps(attributes)
        else:
            # Sanitize attributes to only include primitive types (FalkorDB requirement)
            edge_data.update(_sanitize_attributes(edge.attributes))

        edges.append(edge_data)

    if driver.graph_operations_interface:
        await driver.graph_operations_interface.episodic_node_save_bulk(None, driver, tx, episodes)
        await driver.graph_operations_interface.node_save_bulk(None, driver, tx, nodes)
        await driver.graph_operations_interface.episodic_edge_save_bulk(
            None, driver, tx, [edge.model_dump() for edge in episodic_edges]
        )
        await driver.graph_operations_interface.edge_save_bulk(None, driver, tx, edges)

    elif driver.provider == GraphProvider.KUZU:
        # FIXME: Kuzu's UNWIND does not currently support STRUCT[] type properly, so we insert the data one by one instead for now.
        episode_query = get_episode_node_save_bulk_query(driver.provider)
        for episode in episodes:
            await tx.run(episode_query, **episode)
        entity_node_query = get_entity_node_save_bulk_query(driver.provider, nodes)
        for node in nodes:
            await tx.run(entity_node_query, **node)
        entity_edge_query = get_entity_edge_save_bulk_query(driver.provider)
        for edge in edges:
            await tx.run(entity_edge_query, **edge)
        episodic_edge_query = get_episodic_edge_save_bulk_query(driver.provider)
        for edge in episodic_edges:
            await tx.run(episodic_edge_query, **edge.model_dump())
    elif driver.provider == GraphProvider.FALKORDB:
        # FalkorDB: get_entity_node_save_bulk_query returns a list of (query, params) tuples
        # because FalkorDB needs to set labels one by one
        episode_query = get_episode_node_save_bulk_query(driver.provider)
        logger.info(f'[bulk_save] Saving {len(episodes)} episodes')
        await tx.run(episode_query, episodes=episodes)

        # Entity nodes - iterate over query tuples
        entity_node_queries = get_entity_node_save_bulk_query(driver.provider, nodes)
        logger.info(f'[bulk_save] Saving {len(nodes)} entity nodes ({len(entity_node_queries)} queries)')
        for query, params in entity_node_queries:
            await tx.run(query, **params)

        # Episodic edges
        await tx.run(
            get_episodic_edge_save_bulk_query(driver.provider),
            episodic_edges=[edge.model_dump() for edge in episodic_edges],
        )

        # Entity edges by type
        edges_by_type: dict[str, list[dict[str, Any]]] = {}
        for edge in edges:
            edge_type = edge.get('name', 'RELATES_TO')
            if edge_type not in edges_by_type:
                edges_by_type[edge_type] = []
            edges_by_type[edge_type].append(edge)

        for edge_type, typed_edges in edges_by_type.items():
            query = get_entity_edge_save_bulk_query_by_type(driver.provider, edge_type)
            logger.info(f'[bulk_save] Saving {len(typed_edges)} edges of type {edge_type}')
            await tx.run(query, entity_edges=typed_edges)
    else:
        # Log bulk save operation details for debugging
        episode_query = get_episode_node_save_bulk_query(driver.provider)
        logger.info(f'[bulk_save] Saving {len(episodes)} episodes, query_len={len(episode_query)}, first_episode_uuid={episodes[0]["uuid"] if episodes else "none"}')
        if not episode_query or not episode_query.strip():
            logger.error(f'[bulk_save] Empty episode query! provider={driver.provider}, episodes_count={len(episodes)}')
        await tx.run(episode_query, episodes=episodes)
        logger.info(f'[bulk_save] Saving {len(nodes)} entity nodes')
        await tx.run(
            get_entity_node_save_bulk_query(driver.provider, nodes),
            nodes=nodes,
        )
        await tx.run(
            get_episodic_edge_save_bulk_query(driver.provider),
            episodic_edges=[edge.model_dump() for edge in episodic_edges],
        )
        # Group edges by type and save each group with the correct relationship type
        # This is necessary because Cypher doesn't support dynamic relationship types
        edges_by_type: dict[str, list[dict[str, Any]]] = {}
        for edge in edges:
            edge_type = edge.get('name', 'RELATES_TO')
            if edge_type not in edges_by_type:
                edges_by_type[edge_type] = []
            edges_by_type[edge_type].append(edge)

        for edge_type, typed_edges in edges_by_type.items():
            query = get_entity_edge_save_bulk_query_by_type(driver.provider, edge_type)
            logger.info(f'[bulk_save] Saving {len(typed_edges)} edges of type {edge_type}')
            await tx.run(query, entity_edges=typed_edges)


async def extract_nodes_and_edges_bulk(
    clients: GraphitiClients,
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]],
    edge_type_map: dict[tuple[str, str], list[str]],
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    edge_types: dict[str, type[BaseModel]] | None = None,
) -> tuple[list[list[EntityNode]], list[list[EntityEdge]]]:
    extracted_nodes_bulk: list[list[EntityNode]] = await semaphore_gather(
        *[
            extract_nodes(clients, episode, previous_episodes, entity_types, excluded_entity_types)
            for episode, previous_episodes in episode_tuples
        ]
    )

    extracted_edges_bulk: list[list[EntityEdge]] = await semaphore_gather(
        *[
            extract_edges(
                clients,
                episode,
                extracted_nodes_bulk[i],
                previous_episodes,
                edge_type_map=edge_type_map,
                group_id=episode.group_id,
                edge_types=edge_types,
            )
            for i, (episode, previous_episodes) in enumerate(episode_tuples)
        ]
    )

    return extracted_nodes_bulk, extracted_edges_bulk


async def dedupe_nodes_bulk(
    clients: GraphitiClients,
    extracted_nodes: list[list[EntityNode]],
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]],
    entity_types: dict[str, type[BaseModel]] | None = None,
) -> tuple[dict[str, list[EntityNode]], dict[str, str]]:
    """Resolve entity duplicates across an in-memory batch using deterministic matching only.

    EasyOps optimization: This function no longer uses LLM for deduplication.
    It only performs deterministic matching (exact string + MinHash similarity).

    LLM-based semantic deduplication is handled by `semantic_dedupe_nodes_bulk`
    which:
    - Runs AFTER attributes extraction (so summary is available)
    - Searches DB candidates per batch (not globally)
    - Uses one LLM call per batch

    Returns:
        Tuple of:
        - nodes_by_episode: Dict mapping episode UUID to deduplicated nodes
        - compressed_map: UUID mapping from duplicates to canonical nodes
    """
    all_extracted = [node for nodes in extracted_nodes for node in nodes]

    # Step 1: Deterministic matching within batch (exact string + MinHash)
    duplicate_pairs: list[tuple[str, str]] = []
    canonical_nodes: dict[str, EntityNode] = {}

    for nodes in extracted_nodes:
        for node in nodes:
            if not canonical_nodes:
                canonical_nodes[node.uuid] = node
                continue

            existing_candidates = list(canonical_nodes.values())
            normalized = _normalize_string_exact(node.name)

            # Exact string match
            exact_match = next(
                (
                    candidate
                    for candidate in existing_candidates
                    if _normalize_string_exact(candidate.name) == normalized
                ),
                None,
            )
            if exact_match is not None:
                if exact_match.uuid != node.uuid:
                    duplicate_pairs.append((node.uuid, exact_match.uuid))
                continue

            # MinHash similarity match
            indexes = _build_candidate_indexes(existing_candidates)
            state = DedupResolutionState(
                resolved_nodes=[None],
                uuid_map={},
                unresolved_indices=[],
            )
            _resolve_with_similarity([node], indexes, state)

            resolved = state.resolved_nodes[0]
            if resolved is None:
                canonical_nodes[node.uuid] = node
                continue

            canonical_uuid = resolved.uuid
            canonical_nodes.setdefault(canonical_uuid, resolved)
            if canonical_uuid != node.uuid:
                duplicate_pairs.append((node.uuid, canonical_uuid))

    # Build compressed UUID map
    compressed_map: dict[str, str] = _build_directed_uuid_map(duplicate_pairs)

    # Build nodes_by_episode
    nodes_by_episode: dict[str, list[EntityNode]] = {}
    for i, (episode, _) in enumerate(episode_tuples):
        deduped_nodes: list[EntityNode] = []
        seen: set[str] = set()

        for node in extracted_nodes[i]:
            canonical_uuid = compressed_map.get(node.uuid, node.uuid)
            if canonical_uuid in seen:
                continue
            seen.add(canonical_uuid)

            canonical_node = canonical_nodes.get(canonical_uuid)
            if canonical_node is None:
                canonical_node = node
            deduped_nodes.append(canonical_node)

        nodes_by_episode[episode.uuid] = deduped_nodes

    logger.info(
        '[dedupe_nodes_bulk] Deterministic dedup: %d pairs found',
        len(duplicate_pairs),
    )

    return nodes_by_episode, compressed_map


async def dedupe_edges_bulk(
    clients: GraphitiClients,
    extracted_edges: list[list[EntityEdge]],
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]],
    _entities: list[EntityNode],
    edge_types: dict[str, type[BaseModel]],
    _edge_type_map: dict[tuple[str, str], list[str]],
) -> dict[str, list[EntityEdge]]:
    embedder = clients.embedder
    min_score = 0.6

    # generate embeddings
    await semaphore_gather(
        *[create_entity_edge_embeddings(embedder, edges) for edges in extracted_edges]
    )

    # Find similar results
    dedupe_tuples: list[tuple[EpisodicNode, EntityEdge, list[EntityEdge]]] = []
    for i, edges_i in enumerate(extracted_edges):
        existing_edges: list[EntityEdge] = []
        for edges_j in extracted_edges:
            existing_edges += edges_j

        for edge in edges_i:
            candidates: list[EntityEdge] = []
            for existing_edge in existing_edges:
                # Skip self-comparison
                if edge.uuid == existing_edge.uuid:
                    continue
                # EasyOps customization: Match by (source, target, edge_type) instead of fact similarity
                # This ensures edges with same type between same nodes are always deduplicated
                if (
                    edge.source_node_uuid != existing_edge.source_node_uuid
                    or edge.target_node_uuid != existing_edge.target_node_uuid
                ):
                    continue
                # Same source and target - check if same edge type
                if edge.name == existing_edge.name:
                    candidates.append(existing_edge)

            dedupe_tuples.append((episode_tuples[i][0], edge, candidates))

    bulk_edge_resolutions: list[
        tuple[EntityEdge, EntityEdge, list[EntityEdge]]
    ] = await semaphore_gather(
        *[
            resolve_extracted_edge(
                clients.llm_client,
                edge,
                candidates,
                candidates,
                episode,
                edge_types,
                set(edge_types),
            )
            for episode, edge, candidates in dedupe_tuples
        ]
    )

    # For now we won't track edge invalidation
    duplicate_pairs: list[tuple[str, str]] = []
    for i, (_, _, duplicates) in enumerate(bulk_edge_resolutions):
        episode, edge, candidates = dedupe_tuples[i]
        for duplicate in duplicates:
            duplicate_pairs.append((edge.uuid, duplicate.uuid))

    # Now we compress the duplicate_map, so that 3 -> 2 and 2 -> becomes 3 -> 1 (sorted by uuid)
    compressed_map: dict[str, str] = compress_uuid_map(duplicate_pairs)

    edge_uuid_map: dict[str, EntityEdge] = {
        edge.uuid: edge for edges in extracted_edges for edge in edges
    }

    edges_by_episode: dict[str, list[EntityEdge]] = {}
    for i, edges in enumerate(extracted_edges):
        episode = episode_tuples[i][0]

        edges_by_episode[episode.uuid] = [
            edge_uuid_map[compressed_map.get(edge.uuid, edge.uuid)] for edge in edges
        ]

    return edges_by_episode


class UnionFind:
    def __init__(self, elements):
        # start each element in its own set
        self.parent = {e: e for e in elements}

    def find(self, x):
        # path‐compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # attach the lexicographically larger root under the smaller
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def compress_uuid_map(duplicate_pairs: list[tuple[str, str]]) -> dict[str, str]:
    """
    all_ids: iterable of all entity IDs (strings)
    duplicate_pairs: iterable of (id1, id2) pairs
    returns: dict mapping each id -> lexicographically smallest id in its duplicate set
    """
    all_uuids = set()
    for pair in duplicate_pairs:
        all_uuids.add(pair[0])
        all_uuids.add(pair[1])

    uf = UnionFind(all_uuids)
    for a, b in duplicate_pairs:
        uf.union(a, b)
    # ensure full path‐compression before mapping
    return {uuid: uf.find(uuid) for uuid in all_uuids}


E = typing.TypeVar('E', bound=Edge)


def resolve_edge_pointers(edges: list[E], uuid_map: dict[str, str]):
    for edge in edges:
        source_uuid = edge.source_node_uuid
        target_uuid = edge.target_node_uuid
        edge.source_node_uuid = uuid_map.get(source_uuid, source_uuid)
        edge.target_node_uuid = uuid_map.get(target_uuid, target_uuid)

    return edges
