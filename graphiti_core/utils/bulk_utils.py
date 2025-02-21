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
from collections import defaultdict
from datetime import datetime
from math import ceil

from neo4j import AsyncDriver, AsyncManagedTransaction
from numpy import dot, sqrt
from pydantic import BaseModel
from typing_extensions import Any

from graphiti_core.edges import Edge, EntityEdge, EpisodicEdge
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.models.edges.edge_db_queries import (
    ENTITY_EDGE_SAVE_BULK,
    EPISODIC_EDGE_SAVE_BULK,
)
from graphiti_core.models.nodes.node_db_queries import (
    ENTITY_NODE_SAVE_BULK,
    EPISODIC_NODE_SAVE_BULK,
)
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import get_relevant_edges, get_relevant_nodes
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.edge_operations import (
    build_episodic_edges,
    dedupe_edge_list,
    dedupe_extracted_edges,
    extract_edges,
)
from graphiti_core.utils.maintenance.graph_data_operations import (
    EPISODE_WINDOW_LEN,
    retrieve_episodes,
)
from graphiti_core.utils.maintenance.node_operations import (
    dedupe_extracted_nodes,
    dedupe_node_list,
    extract_nodes,
)
from graphiti_core.utils.maintenance.temporal_operations import extract_edge_dates

logger = logging.getLogger(__name__)

CHUNK_SIZE = 10


class RawEpisode(BaseModel):
    name: str
    content: str
    source_description: str
    source: EpisodeType
    reference_time: datetime


async def retrieve_previous_episodes_bulk(
    driver: AsyncDriver, episodes: list[EpisodicNode]
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
    driver: AsyncDriver,
    episodic_nodes: list[EpisodicNode],
    episodic_edges: list[EpisodicEdge],
    entity_nodes: list[EntityNode],
    entity_edges: list[EntityEdge],
):
    async with driver.session() as session:
        await session.execute_write(
            add_nodes_and_edges_bulk_tx, episodic_nodes, episodic_edges, entity_nodes, entity_edges
        )


async def add_nodes_and_edges_bulk_tx(
    tx: AsyncManagedTransaction,
    episodic_nodes: list[EpisodicNode],
    episodic_edges: list[EpisodicEdge],
    entity_nodes: list[EntityNode],
    entity_edges: list[EntityEdge],
):
    episodes = [dict(episode) for episode in episodic_nodes]
    for episode in episodes:
        episode['source'] = str(episode['source'].value)
    nodes: list[dict[str, Any]] = []
    for node in entity_nodes:
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

    await tx.run(EPISODIC_NODE_SAVE_BULK, episodes=episodes)
    await tx.run(ENTITY_NODE_SAVE_BULK, nodes=nodes)
    await tx.run(EPISODIC_EDGE_SAVE_BULK, episodic_edges=[dict(edge) for edge in episodic_edges])
    await tx.run(ENTITY_EDGE_SAVE_BULK, entity_edges=[dict(edge) for edge in entity_edges])


async def extract_nodes_and_edges_bulk(
    llm_client: LLMClient, episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]]
) -> tuple[list[EntityNode], list[EntityEdge], list[EpisodicEdge]]:
    extracted_nodes_bulk = await semaphore_gather(
        *[
            extract_nodes(llm_client, episode, previous_episodes)
            for episode, previous_episodes in episode_tuples
        ]
    )

    episodes, previous_episodes_list = (
        [episode[0] for episode in episode_tuples],
        [episode[1] for episode in episode_tuples],
    )

    extracted_edges_bulk = await semaphore_gather(
        *[
            extract_edges(
                llm_client,
                episode,
                extracted_nodes_bulk[i],
                previous_episodes_list[i],
                episode.group_id,
            )
            for i, episode in enumerate(episodes)
        ]
    )

    episodic_edges: list[EpisodicEdge] = []
    for i, episode in enumerate(episodes):
        episodic_edges += build_episodic_edges(extracted_nodes_bulk[i], episode, episode.created_at)

    nodes: list[EntityNode] = []
    for extracted_nodes in extracted_nodes_bulk:
        nodes += extracted_nodes

    edges: list[EntityEdge] = []
    for extracted_edges in extracted_edges_bulk:
        edges += extracted_edges

    return nodes, edges, episodic_edges


async def dedupe_nodes_bulk(
    driver: AsyncDriver,
    llm_client: LLMClient,
    extracted_nodes: list[EntityNode],
) -> tuple[list[EntityNode], dict[str, str]]:
    # Compress nodes
    nodes, uuid_map = node_name_match(extracted_nodes)

    compressed_nodes, compressed_map = await compress_nodes(llm_client, nodes, uuid_map)

    node_chunks = [nodes[i : i + CHUNK_SIZE] for i in range(0, len(nodes), CHUNK_SIZE)]

    existing_nodes_chunks: list[list[EntityNode]] = list(
        await semaphore_gather(
            *[get_relevant_nodes(driver, SearchFilters(), node_chunk) for node_chunk in node_chunks]
        )
    )

    results: list[tuple[list[EntityNode], dict[str, str]]] = list(
        await semaphore_gather(
            *[
                dedupe_extracted_nodes(llm_client, node_chunk, existing_nodes_chunks[i])
                for i, node_chunk in enumerate(node_chunks)
            ]
        )
    )

    final_nodes: list[EntityNode] = []
    for result in results:
        final_nodes.extend(result[0])
        partial_uuid_map = result[1]
        compressed_map.update(partial_uuid_map)

    return final_nodes, compressed_map


async def dedupe_edges_bulk(
    driver: AsyncDriver, llm_client: LLMClient, extracted_edges: list[EntityEdge]
) -> list[EntityEdge]:
    # First compress edges
    compressed_edges = await compress_edges(llm_client, extracted_edges)

    edge_chunks = [
        compressed_edges[i : i + CHUNK_SIZE] for i in range(0, len(compressed_edges), CHUNK_SIZE)
    ]

    relevant_edges_chunks: list[list[EntityEdge]] = list(
        await semaphore_gather(
            *[get_relevant_edges(driver, edge_chunk, None, None) for edge_chunk in edge_chunks]
        )
    )

    resolved_edge_chunks: list[list[EntityEdge]] = list(
        await semaphore_gather(
            *[
                dedupe_extracted_edges(llm_client, edge_chunk, relevant_edges_chunks[i])
                for i, edge_chunk in enumerate(edge_chunks)
            ]
        )
    )

    edges = [edge for edge_chunk in resolved_edge_chunks for edge in edge_chunk]
    return edges


def node_name_match(nodes: list[EntityNode]) -> tuple[list[EntityNode], dict[str, str]]:
    uuid_map: dict[str, str] = {}
    name_map: dict[str, EntityNode] = {}
    for node in nodes:
        if node.name in name_map:
            uuid_map[node.uuid] = name_map[node.name].uuid
            continue

        name_map[node.name] = node

    return [node for node in name_map.values()], uuid_map


async def compress_nodes(
    llm_client: LLMClient, nodes: list[EntityNode], uuid_map: dict[str, str]
) -> tuple[list[EntityNode], dict[str, str]]:
    # We want to first compress the nodes by deduplicating nodes across each of the episodes added in bulk
    if len(nodes) == 0:
        return nodes, uuid_map

    # Our approach involves us deduplicating chunks of nodes in parallel.
    # We want n chunks of size n so that n ** 2 == len(nodes).
    # We want chunk sizes to be at least 10 for optimizing LLM processing time
    chunk_size = max(int(sqrt(len(nodes))), CHUNK_SIZE)

    # First calculate similarity scores between nodes
    similarity_scores: list[tuple[int, int, float]] = [
        (i, j, dot(n.name_embedding or [], m.name_embedding or []))
        for i, n in enumerate(nodes)
        for j, m in enumerate(nodes[:i])
    ]

    # We now sort by semantic similarity
    similarity_scores.sort(key=lambda score_tuple: score_tuple[2])

    # initialize our chunks based on chunk size
    node_chunks: list[list[EntityNode]] = [[] for _ in range(ceil(len(nodes) / chunk_size))]

    # Draft the most similar nodes into the same chunk
    while len(similarity_scores) > 0:
        i, j, _ = similarity_scores.pop()
        # determine if any of the nodes have already been drafted into a chunk
        n = nodes[i]
        m = nodes[j]
        # make sure the shortest chunks get preference
        node_chunks.sort(reverse=True, key=lambda chunk: len(chunk))

        n_chunk = max([i if n in chunk else -1 for i, chunk in enumerate(node_chunks)])
        m_chunk = max([i if m in chunk else -1 for i, chunk in enumerate(node_chunks)])

        # both nodes already in a chunk
        if n_chunk > -1 and m_chunk > -1:
            continue

        # n has a chunk and that chunk is not full
        elif n_chunk > -1 and len(node_chunks[n_chunk]) < chunk_size:
            # put m in the same chunk as n
            node_chunks[n_chunk].append(m)

        # m has a chunk and that chunk is not full
        elif m_chunk > -1 and len(node_chunks[m_chunk]) < chunk_size:
            # put n in the same chunk as m
            node_chunks[m_chunk].append(n)

        # neither node has a chunk or the chunk is full
        else:
            # add both nodes to the shortest chunk
            node_chunks[-1].extend([n, m])

    results = await semaphore_gather(
        *[dedupe_node_list(llm_client, chunk) for chunk in node_chunks]
    )

    extended_map = dict(uuid_map)
    compressed_nodes: list[EntityNode] = []
    for node_chunk, uuid_map_chunk in results:
        compressed_nodes += node_chunk
        extended_map.update(uuid_map_chunk)

    # Check if we have removed all duplicates
    if len(compressed_nodes) == len(nodes):
        compressed_uuid_map = compress_uuid_map(extended_map)
        return compressed_nodes, compressed_uuid_map

    return await compress_nodes(llm_client, compressed_nodes, extended_map)


async def compress_edges(llm_client: LLMClient, edges: list[EntityEdge]) -> list[EntityEdge]:
    if len(edges) == 0:
        return edges
    # We only want to dedupe edges that are between the same pair of nodes
    # We build a map of the edges based on their source and target nodes.
    edge_chunks = chunk_edges_by_nodes(edges)

    results = await semaphore_gather(
        *[dedupe_edge_list(llm_client, chunk) for chunk in edge_chunks]
    )

    compressed_edges: list[EntityEdge] = []
    for edge_chunk in results:
        compressed_edges += edge_chunk

    # Check if we have removed all duplicates
    if len(compressed_edges) == len(edges):
        return compressed_edges

    return await compress_edges(llm_client, compressed_edges)


def compress_uuid_map(uuid_map: dict[str, str]) -> dict[str, str]:
    # make sure all uuid values aren't mapped to other uuids
    compressed_map = {}
    for key, uuid in uuid_map.items():
        curr_value = uuid
        while curr_value in uuid_map:
            curr_value = uuid_map[curr_value]

        compressed_map[key] = curr_value
    return compressed_map


E = typing.TypeVar('E', bound=Edge)


def resolve_edge_pointers(edges: list[E], uuid_map: dict[str, str]):
    for edge in edges:
        source_uuid = edge.source_node_uuid
        target_uuid = edge.target_node_uuid
        edge.source_node_uuid = uuid_map.get(source_uuid, source_uuid)
        edge.target_node_uuid = uuid_map.get(target_uuid, target_uuid)

    return edges


async def extract_edge_dates_bulk(
    llm_client: LLMClient,
    extracted_edges: list[EntityEdge],
    episode_pairs: list[tuple[EpisodicNode, list[EpisodicNode]]],
) -> list[EntityEdge]:
    edges: list[EntityEdge] = []
    # confirm that all of our edges have at least one episode
    for edge in extracted_edges:
        if edge.episodes is not None and len(edge.episodes) > 0:
            edges.append(edge)

    episode_uuid_map: dict[str, tuple[EpisodicNode, list[EpisodicNode]]] = {
        episode.uuid: (episode, previous_episodes) for episode, previous_episodes in episode_pairs
    }

    results = await semaphore_gather(
        *[
            extract_edge_dates(
                llm_client,
                edge,
                episode_uuid_map[edge.episodes[0]][0],  # type: ignore
                episode_uuid_map[edge.episodes[0]][1],  # type: ignore
            )
            for edge in edges
        ]
    )

    for i, result in enumerate(results):
        valid_at = result[0]
        invalid_at = result[1]
        edge = edges[i]

        edge.valid_at = valid_at
        edge.invalid_at = invalid_at
        if edge.invalid_at:
            edge.expired_at = utc_now()

    return edges


def chunk_edges_by_nodes(edges: list[EntityEdge]) -> list[list[EntityEdge]]:
    # We only want to dedupe edges that are between the same pair of nodes
    # We build a map of the edges based on their source and target nodes.
    edge_chunk_map: dict[str, list[EntityEdge]] = defaultdict(list)
    for edge in edges:
        # We drop loop edges
        if edge.source_node_uuid == edge.target_node_uuid:
            continue

        # Keep the order of the two nodes consistent, we want to be direction agnostic during edge resolution
        pointers = [edge.source_node_uuid, edge.target_node_uuid]
        pointers.sort()

        edge_chunk_map[pointers[0] + pointers[1]].append(edge)

    edge_chunks = [chunk for chunk in edge_chunk_map.values()]

    return edge_chunks
