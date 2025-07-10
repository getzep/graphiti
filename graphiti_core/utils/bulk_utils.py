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
import typing
from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field
from typing_extensions import Any

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession
from graphiti_core.edges import Edge, EntityEdge, EpisodicEdge, create_entity_edge_embeddings
from graphiti_core.embedder import EmbedderClient
from graphiti_core.graph_queries import (
    get_entity_edge_save_bulk_query,
    get_entity_node_save_bulk_query,
)
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import DEFAULT_DATABASE, normalize_l2, semaphore_gather
from graphiti_core.models.edges.edge_db_queries import (
    EPISODIC_EDGE_SAVE_BULK,
)
from graphiti_core.models.nodes.node_db_queries import (
    EPISODIC_NODE_SAVE_BULK,
)
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode, create_entity_node_embeddings
from graphiti_core.utils.maintenance.edge_operations import (
    extract_edges,
    resolve_extracted_edge,
)
from graphiti_core.utils.maintenance.graph_data_operations import (
    EPISODE_WINDOW_LEN,
    retrieve_episodes,
)
from graphiti_core.utils.maintenance.node_operations import (
    extract_nodes,
    resolve_extracted_nodes,
)

logger = logging.getLogger(__name__)

CHUNK_SIZE = 10


class RawEpisode(BaseModel):
    name: str
    uuid: str | None = Field(default=None)
    content: str
    source_description: str
    source: EpisodeType
    reference_time: datetime


async def retrieve_previous_episodes_bulk(
    driver: GraphDriver, episodes: list[EpisodicNode]
) -> list[tuple[EpisodicNode, list[EpisodicNode]]]:
    previous_episodes_list = await semaphore_gather(
        *[
            retrieve_episodes(
                driver, episode.valid_at, last_n=EPISODE_WINDOW_LEN, group_ids=[episode.group_id]
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
    session = driver.session(database=DEFAULT_DATABASE)
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
    nodes: list[dict[str, Any]] = []
    for node in entity_nodes:
        if node.name_embedding is None:
            await node.generate_name_embedding(embedder)
        entity_data: dict[str, Any] = {
            'uuid': node.uuid,
            'name': node.name,
            'name_embedding': node.name_embedding,
            'group_id': node.group_id,
            'summary': node.summary,
            'created_at': node.created_at,
        }

        entity_data.update(node.attributes or {})
        entity_data['labels'] = list(set(node.labels + ['Entity']))
        nodes.append(entity_data)

    edges: list[dict[str, Any]] = []
    for edge in entity_edges:
        if edge.fact_embedding is None:
            await edge.generate_embedding(embedder)
        edge_data: dict[str, Any] = {
            'uuid': edge.uuid,
            'source_node_uuid': edge.source_node_uuid,
            'target_node_uuid': edge.target_node_uuid,
            'name': edge.name,
            'fact': edge.fact,
            'fact_embedding': edge.fact_embedding,
            'group_id': edge.group_id,
            'episodes': edge.episodes,
            'created_at': edge.created_at,
            'expired_at': edge.expired_at,
            'valid_at': edge.valid_at,
            'invalid_at': edge.invalid_at,
        }

        edge_data.update(edge.attributes or {})
        edges.append(edge_data)

    await tx.run(EPISODIC_NODE_SAVE_BULK, episodes=episodes)
    entity_node_save_bulk = get_entity_node_save_bulk_query(nodes, driver.provider)
    await tx.run(entity_node_save_bulk, nodes=nodes)
    await tx.run(
        EPISODIC_EDGE_SAVE_BULK, episodic_edges=[edge.model_dump() for edge in episodic_edges]
    )
    entity_edge_save_bulk = get_entity_edge_save_bulk_query(driver.provider)
    await tx.run(entity_edge_save_bulk, entity_edges=edges)


async def extract_nodes_and_edges_bulk(
    clients: GraphitiClients,
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]],
    edge_type_map: dict[tuple[str, str], list[str]],
    entity_types: dict[str, BaseModel] | None = None,
    excluded_entity_types: list[str] | None = None,
    edge_types: dict[str, BaseModel] | None = None,
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
    entity_types: dict[str, BaseModel] | None = None,
) -> tuple[dict[str, list[EntityNode]], dict[str, str]]:
    embedder = clients.embedder
    min_score = 0.8

    # generate embeddings
    await semaphore_gather(
        *[create_entity_node_embeddings(embedder, nodes) for nodes in extracted_nodes]
    )

    # Find similar results
    dedupe_tuples: list[tuple[list[EntityNode], list[EntityNode]]] = []
    for i, nodes_i in enumerate(extracted_nodes):
        existing_nodes: list[EntityNode] = []
        for j, nodes_j in enumerate(extracted_nodes):
            if i == j:
                continue
            existing_nodes += nodes_j

        candidates_i: list[EntityNode] = []
        for node in nodes_i:
            for existing_node in existing_nodes:
                # Approximate BM25 by checking for word overlaps (this is faster than creating many in-memory indices)
                # This approach will cast a wider net than BM25, which is ideal for this use case
                node_words = set(node.name.lower().split())
                existing_node_words = set(existing_node.name.lower().split())
                has_overlap = not node_words.isdisjoint(existing_node_words)
                if has_overlap:
                    candidates_i.append(existing_node)
                    continue

                # Check for semantic similarity even if there is no overlap
                similarity = np.dot(
                    normalize_l2(node.name_embedding or []),
                    normalize_l2(existing_node.name_embedding or []),
                )
                if similarity >= min_score:
                    candidates_i.append(existing_node)

        dedupe_tuples.append((nodes_i, candidates_i))

    # Determine Node Resolutions
    bulk_node_resolutions: list[
        tuple[list[EntityNode], dict[str, str], list[tuple[EntityNode, EntityNode]]]
    ] = await semaphore_gather(
        *[
            resolve_extracted_nodes(
                clients,
                dedupe_tuple[0],
                episode_tuples[i][0],
                episode_tuples[i][1],
                entity_types,
                existing_nodes_override=dedupe_tuples[i][1],
            )
            for i, dedupe_tuple in enumerate(dedupe_tuples)
        ]
    )

    # Collect all duplicate pairs sorted by uuid
    duplicate_pairs: list[tuple[EntityNode, EntityNode]] = []
    for _, _, duplicates in bulk_node_resolutions:
        for duplicate in duplicates:
            n, m = duplicate
            if n.uuid < m.uuid:
                duplicate_pairs.append((n, m))
            else:
                duplicate_pairs.append((m, n))

    # Build full deduplication map
    duplicate_map: dict[str, str] = {}
    for value, key in duplicate_pairs:
        if key.uuid in duplicate_map:
            existing_value = duplicate_map[key.uuid]
            duplicate_map[key.uuid] = value.uuid if value.uuid < existing_value else existing_value
        else:
            duplicate_map[key.uuid] = value.uuid

    # Now we compress the duplicate_map, so that 3 -> 2 and 2 -> becomes 3 -> 1 (sorted by uuid)
    compressed_map: dict[str, str] = compress_uuid_map(duplicate_map)

    node_uuid_map: dict[str, EntityNode] = {
        node.uuid: node for nodes in extracted_nodes for node in nodes
    }

    nodes_by_episode: dict[str, list[EntityNode]] = {}
    for i, nodes in enumerate(extracted_nodes):
        episode = episode_tuples[i][0]

        nodes_by_episode[episode.uuid] = [
            node_uuid_map[compressed_map.get(node.uuid, node.uuid)] for node in nodes
        ]

    return nodes_by_episode, compressed_map


async def dedupe_edges_bulk(
    clients: GraphitiClients,
    extracted_edges: list[list[EntityEdge]],
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]],
    _entities: list[EntityNode],
    edge_types: dict[str, BaseModel],
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
        for j, edges_j in enumerate(extracted_edges):
            if i == j:
                continue
            existing_edges += edges_j

        for edge in edges_i:
            candidates: list[EntityEdge] = []
            for existing_edge in existing_edges:
                # Approximate BM25 by checking for word overlaps (this is faster than creating many in-memory indices)
                # This approach will cast a wider net than BM25, which is ideal for this use case
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

    bulk_edge_resolutions: list[
        tuple[EntityEdge, EntityEdge, list[EntityEdge]]
    ] = await semaphore_gather(
        *[
            resolve_extracted_edge(
                clients.llm_client, edge, candidates, candidates, episode, edge_types
            )
            for episode, edge, candidates in dedupe_tuples
        ]
    )

    duplicate_pairs: list[tuple[EntityEdge, EntityEdge]] = []
    for i, (_, _, duplicates) in enumerate(bulk_edge_resolutions):
        episode, edge, candidates = dedupe_tuples[i]
        for duplicate in duplicates:
            if edge.uuid < duplicate.uuid:
                duplicate_pairs.append((edge, duplicate))
            else:
                duplicate_pairs.append((duplicate, edge))

    # Build full deduplication map
    duplicate_map: dict[str, str] = {}
    for value, key in duplicate_pairs:
        if key.uuid in duplicate_map:
            existing_value = duplicate_map[key.uuid]
            duplicate_map[key.uuid] = value.uuid if value.uuid < existing_value else existing_value
        else:
            duplicate_map[key.uuid] = value.uuid

    # Now we compress the duplicate_map, so that 3 -> 2 and 2 -> becomes 3 -> 1 (sorted by uuid)
    compressed_map: dict[str, str] = compress_uuid_map(duplicate_map)

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


def compress_uuid_map(uuid_map: dict[str, str]) -> dict[str, str]:
    compressed_map = {}

    def find_min_uuid(start: str) -> str:
        path = []
        visited = set()
        curr = start

        while curr in uuid_map and curr not in visited:
            visited.add(curr)
            path.append(curr)
            curr = uuid_map[curr]

        # Also include the last resolved value (could be outside the map)
        path.append(curr)

        # Resolve to lex smallest UUID in the path
        min_uuid = min(path)

        # Assign all UUIDs in the path to the min_uuid
        for node in path:
            compressed_map[node] = min_uuid

        return min_uuid

    for key in uuid_map:
        if key not in compressed_map:
            find_min_uuid(key)

    return compressed_map


E = typing.TypeVar('E', bound=Edge)


def resolve_edge_pointers(edges: list[E], uuid_map: dict[str, str]):
    for edge in edges:
        source_uuid = edge.source_node_uuid
        target_uuid = edge.target_node_uuid
        edge.source_node_uuid = uuid_map.get(source_uuid, source_uuid)
        edge.target_node_uuid = uuid_map.get(target_uuid, target_uuid)

    return edges
