from typing import Tuple

from core.edges import EpisodicEdge, EntityEdge, Edge
from core.nodes import EntityNode, EpisodicNode, Node


async def bfs(
    nodes: list[Node], edges: list[Edge], k: int
) -> Tuple[list[EntityNode], list[EntityEdge]]: ...


# Breadth first search over nodes and edges with desired depth


async def similarity_search(
    query: str, embedder
) -> Tuple[list[EntityNode], list[EntityEdge]]: ...


# vector similarity search over embedded facts


async def fulltext_search(
    query: str,
) -> Tuple[list[EntityNode], list[EntityEdge]]: ...


# fulltext search over names and summary


def build_episodic_edges(
    entity_nodes: list[EntityNode], episode: EpisodicNode
) -> list[EpisodicEdge]:
    edges: list[EpisodicEdge] = []

    for node in entity_nodes:
        edges.append(
            EpisodicEdge(
                source_node=episode,
                target_node=node,
                created_at=episode.created_at,
            )
        )

    return edges
