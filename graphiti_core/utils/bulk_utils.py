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

import asyncio
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
from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface
from graphiti_core.edges import Edge, EntityEdge, EpisodicEdge, create_entity_edge_embeddings
from graphiti_core.embedder import EmbedderClient
from graphiti_core.graphiti_types import GraphitiClients, uses_prompt_routing
from graphiti_core.helpers import normalize_l2, semaphore_gather
from graphiti_core.models.edges.edge_db_queries import (
    get_entity_edge_save_bulk_query,
    get_entity_edge_update_bulk_query,
    get_episodic_edge_save_bulk_query,
)
from graphiti_core.models.nodes.node_db_queries import (
    get_entity_node_save_bulk_query,
    get_entity_node_update_bulk_query,
    get_episode_node_save_bulk_query,
)
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.maintenance.dedup_helpers import (
    DedupResolutionState,
    _build_candidate_indexes,
    _normalize_string_exact,
    _resolve_with_similarity,
)
from graphiti_core.utils.maintenance.edge_operations import extract_edges, resolve_extracted_edge
from graphiti_core.utils.maintenance.graph_data_operations import (
    EPISODE_WINDOW_LEN,
    retrieve_episodes,
)
from graphiti_core.utils.maintenance.node_operations import (
    _identity_key_for_node,
    _identity_type_for_node,
    extract_nodes,
    resolve_extracted_nodes,
)

if typing.TYPE_CHECKING:
    from graphiti_core.ingestion.change_set import ResolutionTrace

logger = logging.getLogger(__name__)

CHUNK_SIZE = 10


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


async def retrieve_previous_episodes_bulk(
    driver: GraphDriver,
    episodes: list[EpisodicNode],
    max_coroutines: int | None = None,
) -> list[tuple[EpisodicNode, list[EpisodicNode]]]:
    previous_episodes_list = await semaphore_gather(
        *[
            retrieve_episodes(
                driver, episode.valid_at, last_n=EPISODE_WINDOW_LEN, group_ids=[episode.group_id]
            )
            for episode in episodes
        ],
        max_coroutines=max_coroutines,
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
    write_timeout: float | None = None,
):
    """Write the records in one session.

    `write_timeout` is a time limit in seconds for the write. With the default `None`,
    the write has no time limit.
    """
    if write_timeout is not None:
        await execute_write_with_timeout(
            driver,
            write_timeout,
            add_nodes_and_edges_bulk_tx,
            episodic_nodes,
            episodic_edges,
            entity_nodes,
            entity_edges,
            embedder,
            driver=driver,
        )
        return

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


async def execute_write_with_timeout(
    graph_driver: GraphDriver, timeout: float, func: Any, /, *args: Any, **kwargs: Any
) -> Any:
    """Run a write function in a new session with a time limit in seconds.

    On Neo4j, the limit is the transaction timeout, which the server applies. On
    FalkorDB, each query gets the limit as its query timeout. The server applies it to
    a write only when its `TIMEOUT_MAX` setting is not zero. On each provider other
    than Neo4j, the call is also cancelled at the limit. The database can then complete
    a write that it already started.
    """
    session = graph_driver.session()
    try:
        if graph_driver.provider == GraphProvider.NEO4J:
            from neo4j import unit_of_work

            return await session.execute_write(unit_of_work(timeout=timeout)(func), *args, **kwargs)
        if hasattr(session, 'query_timeout_ms'):
            session.query_timeout_ms = max(1, int(timeout * 1000))  # type: ignore[attr-defined]
        return await asyncio.wait_for(session.execute_write(func, *args, **kwargs), timeout)
    finally:
        await session.close()


async def _entity_node_record(
    node: EntityNode, embedder: EmbedderClient, driver: GraphDriver
) -> dict[str, Any]:
    """Return the query parameters of one entity node for a bulk save or a bulk update."""
    if node.name_embedding is None:
        await node.generate_name_embedding(embedder)

    entity_data: dict[str, Any] = {
        'uuid': node.uuid,
        'name': node.name,
        'group_id': node.group_id,
        'summary': node.summary,
        'created_at': node.created_at,
        'name_embedding': node.name_embedding,
        'labels': list(set(node.labels + ['Entity'])),
    }

    for k, v in (node.attributes or {}).items():
        if k not in entity_data:
            entity_data[k] = v

    return entity_data


async def _entity_edge_record(
    edge: EntityEdge, embedder: EmbedderClient, driver: GraphDriver
) -> dict[str, Any]:
    """Return the query parameters of one entity edge for a bulk save or a bulk update."""
    if edge.fact_embedding is None:
        await edge.generate_embedding(embedder)
    edge_data: dict[str, Any] = {
        'uuid': edge.uuid,
        'source_node_uuid': edge.source_node_uuid,
        'target_node_uuid': edge.target_node_uuid,
        'name': edge.name,
        'fact': edge.fact,
        'group_id': edge.group_id,
        'episodes': edge.episodes,
        'created_at': edge.created_at,
        'expired_at': edge.expired_at,
        'valid_at': edge.valid_at,
        'invalid_at': edge.invalid_at,
        'reference_time': edge.reference_time,
        'fact_embedding': edge.fact_embedding,
    }

    # Merge attributes without overwriting explicit edge fields.
    # Attributes may contain stale string versions of typed fields
    # (e.g. reference_time as ISO string) that would replace the
    # datetime values set above.
    for k, v in (edge.attributes or {}).items():
        if k not in edge_data:
            edge_data[k] = v

    return edge_data


# The operations of `graph_operations_interface` that the commit and the enrichment of
# concurrent ingestion call. These operations have no Cypher fallback.
INGESTION_INTERFACE_OPERATIONS = (
    'episodic_node_save_bulk',
    'node_save_bulk',
    'episodic_edge_save_bulk',
    'edge_save_bulk',
    'node_update_bulk',
    'edge_update_bulk',
)


def missing_ingestion_interface_operations(driver: GraphDriver) -> list[str]:
    """Return the operations of concurrent ingestion that the interface of the driver lacks.

    An operation that the interface class does not override is missing. The result is
    empty for a driver with no `graph_operations_interface`.
    """
    interface = driver.graph_operations_interface
    if not interface:
        return []
    return sorted(
        name
        for name in INGESTION_INTERFACE_OPERATIONS
        if getattr(type(interface), name, None) in (None, getattr(GraphOperationsInterface, name))
    )


async def update_entity_edges_tx(
    tx: GraphDriverSession,
    entity_edges: list[EntityEdge],
    embedder: EmbedderClient,
    driver: GraphDriver,
):
    """Update entity edges that exist. An edge that does not exist is not created."""
    if not entity_edges:
        return
    edges = [await _entity_edge_record(edge, embedder, driver) for edge in entity_edges]
    if driver.graph_operations_interface:
        await driver.graph_operations_interface.edge_update_bulk(None, driver, tx, edges)
        return
    query = get_entity_edge_update_bulk_query(driver.provider)
    await tx.run(query, entity_edges=edges)


async def update_entity_nodes_tx(
    tx: GraphDriverSession,
    entity_nodes: list[EntityNode],
    embedder: EmbedderClient,
    driver: GraphDriver,
):
    """Update entity nodes that exist. A node that does not exist is not created."""
    if not entity_nodes:
        return
    nodes = [await _entity_node_record(node, embedder, driver) for node in entity_nodes]
    if driver.graph_operations_interface:
        await driver.graph_operations_interface.node_update_bulk(None, driver, tx, nodes)
        return
    query = get_entity_node_update_bulk_query(driver.provider, nodes)
    await tx.run(query, nodes=nodes)


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

    nodes = [await _entity_node_record(node, embedder, driver) for node in entity_nodes]
    edges = [await _entity_edge_record(edge, embedder, driver) for edge in entity_edges]

    if driver.graph_operations_interface:
        await driver.graph_operations_interface.episodic_node_save_bulk(None, driver, tx, episodes)
        await driver.graph_operations_interface.node_save_bulk(None, driver, tx, nodes)
        await driver.graph_operations_interface.episodic_edge_save_bulk(
            None, driver, tx, [edge.model_dump() for edge in episodic_edges]
        )
        await driver.graph_operations_interface.edge_save_bulk(None, driver, tx, edges)

    else:
        await tx.run(get_episode_node_save_bulk_query(driver.provider), episodes=episodes)
        await tx.run(
            get_entity_node_save_bulk_query(driver.provider, nodes),
            nodes=nodes,
        )
        await tx.run(
            get_episodic_edge_save_bulk_query(driver.provider),
            episodic_edges=[edge.model_dump() for edge in episodic_edges],
        )
        await tx.run(
            get_entity_edge_save_bulk_query(driver.provider),
            entity_edges=edges,
        )


async def extract_nodes_and_edges_bulk(
    clients: GraphitiClients,
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]],
    edge_type_map: dict[tuple[str, str], list[str]],
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    edge_types: dict[str, type[BaseModel]] | None = None,
    custom_extraction_instructions: str | None = None,
    use_combined_extraction: bool = True,
    strict_edge_types: bool = False,
) -> tuple[list[list[EntityNode]], list[list[EntityEdge]]]:
    if use_combined_extraction:
        return await _extract_nodes_and_edges_bulk_combined(
            clients,
            episode_tuples,
            edge_type_map,
            entity_types=entity_types,
            excluded_entity_types=excluded_entity_types,
            edge_types=edge_types,
            strict_edge_types=strict_edge_types,
            custom_extraction_instructions=custom_extraction_instructions,
        )

    return await _extract_nodes_and_edges_bulk_separate(
        clients,
        episode_tuples,
        edge_type_map,
        entity_types=entity_types,
        excluded_entity_types=excluded_entity_types,
        edge_types=edge_types,
        strict_edge_types=strict_edge_types,
        custom_extraction_instructions=custom_extraction_instructions,
    )


async def _extract_nodes_and_edges_bulk_combined(
    clients: GraphitiClients,
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]],
    edge_type_map: dict[tuple[str, str], list[str]],
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    edge_types: dict[str, type[BaseModel]] | None = None,
    strict_edge_types: bool = False,
    custom_extraction_instructions: str | None = None,
) -> tuple[list[list[EntityNode]], list[list[EntityEdge]]]:
    """Combined extraction: single LLM call per episode for both nodes and edges."""
    from graphiti_core.utils.maintenance.combined_extraction import (
        extract_nodes_and_edges as extract_combined,
    )

    results = await semaphore_gather(
        *[
            extract_combined(
                clients,
                episode,
                previous_episodes,
                entity_types=entity_types,
                excluded_entity_types=excluded_entity_types,
                edge_type_map=edge_type_map,
                edge_types=edge_types,
                strict_edge_types=strict_edge_types,
                custom_extraction_instructions=custom_extraction_instructions,
            )
            for episode, previous_episodes in episode_tuples
        ],
        max_coroutines=getattr(clients, 'max_coroutines', None),
    )

    nodes_bulk = [nodes for nodes, _, _ in results]
    edges_bulk = [edges for _, edges, _ in results]
    return nodes_bulk, edges_bulk


async def _extract_nodes_and_edges_bulk_separate(
    clients: GraphitiClients,
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]],
    edge_type_map: dict[tuple[str, str], list[str]],
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    edge_types: dict[str, type[BaseModel]] | None = None,
    strict_edge_types: bool = False,
    custom_extraction_instructions: str | None = None,
) -> tuple[list[list[EntityNode]], list[list[EntityEdge]]]:
    extracted_results: list[tuple[list[EntityNode], dict[str, list[int]]]] = await semaphore_gather(
        *[
            extract_nodes(
                clients,
                episode,
                previous_episodes,
                entity_types=entity_types,
                excluded_entity_types=excluded_entity_types,
                custom_extraction_instructions=custom_extraction_instructions,
            )
            for episode, previous_episodes in episode_tuples
        ],
        max_coroutines=getattr(clients, 'max_coroutines', None),
    )
    extracted_nodes_bulk = [nodes for nodes, _ in extracted_results]

    edge_results: list[tuple[list[EntityEdge], list[EntityNode]]] = await semaphore_gather(
        *[
            extract_edges(
                clients,
                episode,
                extracted_nodes_bulk[i],
                previous_episodes,
                edge_type_map=edge_type_map,
                group_id=episode.group_id,
                edge_types=edge_types,
                strict_edge_types=strict_edge_types,
                custom_extraction_instructions=custom_extraction_instructions,
            )
            for i, (episode, previous_episodes) in enumerate(episode_tuples)
        ],
        max_coroutines=getattr(clients, 'max_coroutines', None),
    )

    extracted_edges_bulk: list[list[EntityEdge]] = []
    for i, (edges, materialized_nodes) in enumerate(edge_results):
        extracted_edges_bulk.append(edges)
        extracted_nodes_bulk[i].extend(materialized_nodes)

    return extracted_nodes_bulk, extracted_edges_bulk


async def dedupe_nodes_bulk(
    clients: GraphitiClients,
    extracted_nodes: list[list[EntityNode]],
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]],
    entity_types: dict[str, type[BaseModel]] | None = None,
    identity_properties: dict[str, list[str]] | None = None,
    trace: 'ResolutionTrace | None' = None,
) -> tuple[dict[str, list[EntityNode]], dict[str, str]]:
    """Resolve entity duplicates across an in-memory batch using a two-pass strategy.

    1. Run :func:`resolve_extracted_nodes` for every episode in parallel so each batch item is
       reconciled against the live graph just like the non-batch flow.
    2. Re-run the deterministic similarity heuristics across the union of resolved nodes to catch
       duplicates that only co-occur inside this batch, emitting a canonical UUID map that callers
       can apply to edges and persistence.

    When `trace` is given, the first pass records the candidates that each extracted node
    saw and the stored nodes that an extracted node resolved to.
    """

    trace_kwargs: dict[str, Any] = {'trace': trace} if trace is not None else {}

    async def resolve_nodes(
        nodes: list[EntityNode],
        episode: EpisodicNode,
        previous_episodes: list[EpisodicNode],
    ):
        if identity_properties:
            return await resolve_extracted_nodes(
                clients,
                nodes,
                episode,
                previous_episodes,
                entity_types,
                identity_properties=identity_properties,
                **trace_kwargs,
            )
        return await resolve_extracted_nodes(
            clients,
            nodes,
            episode,
            previous_episodes,
            entity_types,
            **trace_kwargs,
        )

    first_pass_results = await semaphore_gather(
        *[
            resolve_nodes(
                nodes,
                episode_tuples[i][0],
                episode_tuples[i][1],
            )
            for i, nodes in enumerate(extracted_nodes)
        ],
        max_coroutines=getattr(clients, 'max_coroutines', None),
    )

    episode_resolutions: list[tuple[str, list[EntityNode]]] = []
    per_episode_uuid_maps: list[dict[str, str]] = []
    duplicate_pairs: list[tuple[str, str]] = []

    for (resolved_nodes, uuid_map, duplicates), (episode, _) in zip(
        first_pass_results, episode_tuples, strict=True
    ):
        episode_resolutions.append((episode.uuid, resolved_nodes))
        per_episode_uuid_maps.append(uuid_map)
        duplicate_pairs.extend((source.uuid, target.uuid) for source, target in duplicates)

    canonical_nodes: dict[str, EntityNode] = {}
    for _, resolved_nodes in episode_resolutions:
        for node in resolved_nodes:
            # NOTE: this loop is O(n^2) in the number of nodes inside the batch because we rebuild
            # the MinHash index for the accumulated canonical pool each time. The LRU-backed
            # shingle cache keeps the constant factors low for typical batch sizes (≤ CHUNK_SIZE),
            # but if batches grow significantly we should switch to an incremental index or chunked
            # processing.
            if not canonical_nodes:
                canonical_nodes[node.uuid] = node
                continue

            existing_candidates = list(canonical_nodes.values())
            if identity_properties:
                # Attribute and summary hydration uses all episodes that mention an identity-typed
                # node; name and similarity matches use the first episode (#2156).
                identity_type = _identity_type_for_node(node, identity_properties)
                identity_key = _identity_key_for_node(node, identity_properties)
                if identity_type is not None and identity_key is not None:
                    identity_match = next(
                        (
                            candidate
                            for candidate in existing_candidates
                            if _identity_type_for_node(candidate, identity_properties)
                            == identity_type
                            and _identity_key_for_node(candidate, identity_properties)
                            == identity_key
                        ),
                        None,
                    )
                    if identity_match is not None:
                        for key, value in node.attributes.items():
                            identity_match.attributes.setdefault(key, value)
                        if identity_match.uuid != node.uuid:
                            duplicate_pairs.append((node.uuid, identity_match.uuid))
                        continue

            normalized = _normalize_string_exact(node.name)
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

    union_pairs: list[tuple[str, str]] = []
    for uuid_map in per_episode_uuid_maps:
        union_pairs.extend(uuid_map.items())
    union_pairs.extend(duplicate_pairs)

    compressed_map: dict[str, str] = _build_directed_uuid_map(union_pairs)

    nodes_by_episode: dict[str, list[EntityNode]] = {}
    for episode_uuid, resolved_nodes in episode_resolutions:
        deduped_nodes: list[EntityNode] = []
        seen: set[str] = set()
        for node in resolved_nodes:
            canonical_uuid = compressed_map.get(node.uuid, node.uuid)
            if canonical_uuid in seen:
                continue
            seen.add(canonical_uuid)
            canonical_node = canonical_nodes.get(canonical_uuid)
            if canonical_node is None:
                logger.error(
                    'Canonical node %s missing during batch dedupe; falling back to %s',
                    canonical_uuid,
                    node.uuid,
                )
                canonical_node = node
            deduped_nodes.append(canonical_node)

        nodes_by_episode[episode_uuid] = deduped_nodes

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
        *[create_entity_edge_embeddings(embedder, edges) for edges in extracted_edges],
        max_coroutines=getattr(clients, 'max_coroutines', None),
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
                # Approximate BM25 by checking for word overlaps (this is faster than creating many in-memory indices)
                # This approach will cast a wider net than BM25, which is ideal for this use case
                if (
                    edge.source_node_uuid != existing_edge.source_node_uuid
                    or edge.target_node_uuid != existing_edge.target_node_uuid
                ):
                    continue

                edge_words = set(edge.fact.lower().split())
                existing_edge_words = set(existing_edge.fact.lower().split())
                has_overlap = not edge_words.isdisjoint(existing_edge_words)
                if has_overlap:
                    candidates.append(existing_edge)
                    continue

                # Check for semantic similarity even if there is no overlap
                similarity = np.dot(
                    normalize_l2(edge.fact_embedding or []),
                    normalize_l2(existing_edge.fact_embedding or []),
                )
                if similarity >= min_score:
                    candidates.append(existing_edge)

            dedupe_tuples.append((episode_tuples[i][0], edge, candidates))

    resolver_kwargs = {'clients': clients} if uses_prompt_routing(clients) else {}
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
                **resolver_kwargs,
            )
            for episode, edge, candidates in dedupe_tuples
        ],
        max_coroutines=getattr(clients, 'max_coroutines', None),
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

    for edges in extracted_edges:
        for edge in edges:
            canonical_uuid = compressed_map.get(edge.uuid, edge.uuid)
            if canonical_uuid != edge.uuid:
                canonical_edge = edge_uuid_map[canonical_uuid]
                for episode_uuid in edge.episodes:
                    if episode_uuid not in canonical_edge.episodes:
                        canonical_edge.episodes.append(episode_uuid)

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


def resolve_edge_pointers(edges: list[E], uuid_map: dict[str, str]) -> list[E]:
    """Repoint edge endpoints through ``uuid_map``, dropping edges that collapse.

    Endpoints are rewritten in place. The returned list omits edges whose two
    endpoints were distinct before remapping and resolve to the same node —
    dedup can map an endpoint onto the node that is already the other endpoint.
    Extracted self-referencing edges (source already equals target) are kept.
    Use the return value rather than the list passed in.
    """
    surviving: list[E] = []
    for edge in edges:
        source_uuid = edge.source_node_uuid
        target_uuid = edge.target_node_uuid
        edge.source_node_uuid = uuid_map.get(source_uuid, source_uuid)
        edge.target_node_uuid = uuid_map.get(target_uuid, target_uuid)

        if source_uuid != target_uuid and edge.source_node_uuid == edge.target_node_uuid:
            logger.info(
                'Dropping self-edge for node %s (endpoints resolved to the same node)',
                edge.source_node_uuid,
            )
            continue
        surviving.append(edge)

    return surviving
