import json
from typing import List
from datetime import datetime

from core.nodes import EntityNode, EpisodicNode
from core.edges import EpisodicEdge, EntityEdge
import logging

from core.prompts import prompt_library
from core.llm_client import LLMClient

logger = logging.getLogger(__name__)


def build_episodic_edges(
    semantic_nodes: List[EntityNode],
    episode: EpisodicNode,
    transaction_from: datetime,
) -> List[EpisodicEdge]:
    edges: List[EpisodicEdge] = []

    for node in semantic_nodes:
        edge = EpisodicEdge(
            source_node=episode, target_node=node, created_at=transaction_from
        )
        edges.append(edge)

    return edges


async def extract_new_edges(
    llm_client: LLMClient,
    episode: EpisodicNode,
    new_nodes: list[EntityNode],
    relevant_schema: dict[str, any],
    previous_episodes: list[EpisodicNode],
) -> list[EntityEdge]:
    # Prepare context for LLM
    context = {
        "episode_content": episode.content,
        "episode_timestamp": (
            episode.valid_at.isoformat() if episode.valid_at else None
        ),
        "relevant_schema": json.dumps(relevant_schema, indent=2),
        "new_nodes": [
            {"name": node.name, "summary": node.summary} for node in new_nodes
        ],
        "previous_episodes": [
            {
                "content": ep.content,
                "timestamp": ep.valid_at.isoformat() if ep.valid_at else None,
            }
            for ep in previous_episodes
        ],
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_edges.v1(context)
    )
    new_edges_data = llm_response.get("new_edges", [])

    # Convert the extracted data into EntityEdge objects
    new_edges = []
    for edge_data in new_edges_data:
        source_node = next(
            (node for node in new_nodes if node.name == edge_data["source_node"]),
            None,
        )
        target_node = next(
            (node for node in new_nodes if node.name == edge_data["target_node"]),
            None,
        )

        # If source or target is not in new_nodes, check if it's an existing node
        if source_node is None and edge_data["source_node"] in relevant_schema["nodes"]:
            existing_node_data = relevant_schema["nodes"][edge_data["source_node"]]
            source_node = EntityNode(
                uuid=existing_node_data["uuid"],
                name=edge_data["source_node"],
                labels=[existing_node_data["label"]],
                summary="",
                created_at=datetime.now(),
            )
        if target_node is None and edge_data["target_node"] in relevant_schema["nodes"]:
            existing_node_data = relevant_schema["nodes"][edge_data["target_node"]]
            target_node = EntityNode(
                uuid=existing_node_data["uuid"],
                name=edge_data["target_node"],
                labels=[existing_node_data["label"]],
                summary="",
                created_at=datetime.now(),
            )

        if (
            source_node
            and target_node
            and not (
                source_node.name.startswith("Message")
                or target_node.name.startswith("Message")
            )
        ):
            valid_at = (
                datetime.fromisoformat(edge_data["valid_at"])
                if edge_data["valid_at"]
                else episode.valid_at or datetime.now()
            )
            invalid_at = (
                datetime.fromisoformat(edge_data["invalid_at"])
                if edge_data["invalid_at"]
                else None
            )

            new_edge = EntityEdge(
                source_node=source_node,
                target_node=target_node,
                name=edge_data["relation_type"],
                fact=edge_data["fact"],
                episodes=[episode.uuid],
                created_at=datetime.now(),
                valid_at=valid_at,
                invalid_at=invalid_at,
            )
            new_edges.append(new_edge)
            logger.info(
                f"Created new edge: {new_edge.name} from {source_node.name} (UUID: {source_node.uuid}) to {target_node.name} (UUID: {target_node.uuid})"
            )

    return new_edges
