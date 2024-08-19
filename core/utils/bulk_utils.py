import asyncio
from datetime import datetime

from neo4j import AsyncDriver
from pydantic import BaseModel

from core.edges import EpisodicEdge, EntityEdge
from core.llm_client import LLMClient
from core.nodes import EpisodicNode, EntityNode
from core.utils import retrieve_episodes
from core.utils.maintenance.edge_operations import extract_edges, build_episodic_edges
from core.utils.maintenance.graph_data_operations import EPISODE_WINDOW_LEN
from core.utils.maintenance.node_operations import extract_nodes, dedupe_nodes

CHUNK_SIZE = 20


class BulkEpisode(BaseModel):
    name: str
    content: str
    source_description: str
    episode_type: str
    reference_time: datetime


async def retrieve_previous_episodes_bulk(
    driver: AsyncDriver, episodes: list[EpisodicNode]
) -> list[tuple[EpisodicNode, list[EpisodicNode]]]:
    episode_pairs: list[tuple[EpisodicNode, list[EpisodicNode]]] = zip(
        episodes,
        await asyncio.gather(
            *[
                retrieve_episodes(driver, episode.valid_at, last_n=EPISODE_WINDOW_LEN)
                for episode in episodes
            ]
        ),
    )

    return episode_pairs


async def extract_nodes_and_edges_bulk(
    llm_client: LLMClient, episode_pairs: list[tuple[EpisodicNode, list[EpisodicNode]]]
) -> list[tuple[list[EntityNode], list[EntityEdge], list[EpisodicEdge]]]:
    extracted_nodes_bulk = await asyncio.gather(
        *[
            extract_nodes(llm_client, episode, previous_episodes)
            for episode, previous_episodes in episode_pairs
        ]
    )

    episodes, previous_episodes_list = zip(*episode_pairs)

    triplets: list[tuple[EpisodicNode, list[EpisodicNode], list[EntityNode]]] = zip(
        episodes, previous_episodes_list, extracted_nodes_bulk
    )

    extracted_edges_bulk = await asyncio.gather(
        *[
            extract_edges(llm_client, episode, extracted_nodes, previous_episodes)
            for episode, previous_episodes, extracted_nodes in triplets
        ]
    )

    episodic_edges_bulk: list[list[EpisodicEdge]] = [
        build_episodic_edges(extracted_nodes, episode, episode.created_at)
        for episode, _, extracted_nodes in triplets
    ]

    nodes_and_edges_bulk: list[
        tuple[list[EntityNode], list[EntityEdge], list[EpisodicEdge]]
    ] = zip(extracted_nodes_bulk, extracted_edges_bulk, episodic_edges_bulk)

    return nodes_and_edges_bulk


async def compress_nodes_bulk(
    llm_client: LLMClient, nodes: list[EntityNode], uuid_map: dict[str, str]
):
    node_chunks = [nodes[i : i + CHUNK_SIZE] for i in range(0, len(nodes), CHUNK_SIZE)]

    results = await asyncio.gather(
        *[dedupe_nodes(llm_client, chunk) for chunk in node_chunks]
    )

    compressed_nodes: list[EntityNode] = []
    for node_chunk, uuid_map_chunk in results:
        compressed_nodes += node_chunk
