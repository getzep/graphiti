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
import typing
from datetime import datetime

from neo4j import AsyncDriver
from numpy import dot
from pydantic import BaseModel

from graphiti_core.edges import Edge, EntityEdge, EpisodicEdge
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search.search_utils import get_relevant_edges, get_relevant_nodes
from graphiti_core.utils import retrieve_episodes
from graphiti_core.utils.maintenance.edge_operations import (
    build_episodic_edges,
    dedupe_edge_list,
    dedupe_extracted_edges,
    extract_edges,
)
from graphiti_core.utils.maintenance.graph_data_operations import EPISODE_WINDOW_LEN
from graphiti_core.utils.maintenance.node_operations import (
    dedupe_extracted_nodes,
    dedupe_node_list,
    extract_nodes,
)

CHUNK_SIZE = 15


class RawEpisode(BaseModel):
    name: str
    content: str
    source_description: str
    source: EpisodeType
    reference_time: datetime


async def retrieve_previous_episodes_bulk(
    driver: AsyncDriver, episodes: list[EpisodicNode]
) -> list[tuple[EpisodicNode, list[EpisodicNode]]]:
    previous_episodes_list = await asyncio.gather(
        *[
            retrieve_episodes(driver, episode.valid_at, last_n=EPISODE_WINDOW_LEN)
            for episode in episodes
        ]
    )
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]] = [
        (episode, previous_episodes_list[i]) for i, episode in enumerate(episodes)
    ]

    return episode_tuples


async def extract_nodes_and_edges_bulk(
    llm_client: LLMClient, episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]]
) -> tuple[list[EntityNode], list[EntityEdge], list[EpisodicEdge]]:
    extracted_nodes_bulk = await asyncio.gather(
        *[
            extract_nodes(llm_client, episode, previous_episodes)
            for episode, previous_episodes in episode_tuples
        ]
    )

    episodes, previous_episodes_list = (
        [episode[0] for episode in episode_tuples],
        [episode[1] for episode in episode_tuples],
    )

    extracted_edges_bulk = await asyncio.gather(
        *[
            extract_edges(llm_client, episode, extracted_nodes_bulk[i], previous_episodes_list[i])
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

    existing_nodes = await get_relevant_nodes(compressed_nodes, driver)

    nodes, partial_uuid_map, _ = await dedupe_extracted_nodes(
        llm_client, compressed_nodes, existing_nodes
    )

    compressed_map.update(partial_uuid_map)

    return nodes, compressed_map


async def dedupe_edges_bulk(
    driver: AsyncDriver, llm_client: LLMClient, extracted_edges: list[EntityEdge]
) -> list[EntityEdge]:
    # Compress edges
    compressed_edges = await compress_edges(llm_client, extracted_edges)

    existing_edges = await get_relevant_edges(compressed_edges, driver)

    edges = await dedupe_extracted_edges(llm_client, compressed_edges, existing_edges)

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
    if len(nodes) == 0:
        return nodes, uuid_map

    anchor = nodes[0]
    nodes.sort(key=lambda node: dot(anchor.name_embedding or [], node.name_embedding or []))

    node_chunks = [nodes[i : i + CHUNK_SIZE] for i in range(0, len(nodes), CHUNK_SIZE)]

    results = await asyncio.gather(*[dedupe_node_list(llm_client, chunk) for chunk in node_chunks])

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

    anchor = edges[0]
    edges.sort(
        key=lambda embedding: dot(anchor.fact_embedding or [], embedding.fact_embedding or [])
    )

    edge_chunks = [edges[i : i + CHUNK_SIZE] for i in range(0, len(edges), CHUNK_SIZE)]

    results = await asyncio.gather(*[dedupe_edge_list(llm_client, chunk) for chunk in edge_chunks])

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
