from typing import Tuple

from core.edges import EpisodicEdge, SemanticEdge, Edge
from core.nodes import SemanticNode, EpisodicNode, Node


async def bfs(
    nodes: list[Node], edges: list[Edge], k: int
) -> Tuple[list[SemanticNode], list[SemanticEdge]]: ...


# Breadth first search over nodes and edges with desired depth


async def similarity_search(
    query: str, embedder
) -> Tuple[list[SemanticNode], list[SemanticEdge]]: ...


# vector similarity search over embedded facts


async def fulltext_search(
    query: str,
) -> Tuple[list[SemanticNode], list[SemanticEdge]]: ...


# fulltext search over names and summary


def build_episodic_edges(
    semantic_nodes: list[SemanticNode], episode: EpisodicNode
) -> list[EpisodicEdge]:
    edges: list[EpisodicEdge] = []

    for node in semantic_nodes:
        edges.append(
            EpisodicEdge(
                source_node=episode,
                target_node=node,
                transaction_from=episode.transaction_from,
            )
        )

    return edges
