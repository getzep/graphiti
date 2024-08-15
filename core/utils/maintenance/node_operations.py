from datetime import datetime

from core.nodes import SemanticNode, EpisodicNode
import logging
from core.llm_client import LLMClient

from core.prompts import prompt_library

logger = logging.getLogger(__name__)


async def extract_new_nodes(
    llm_client: LLMClient,
    episode: EpisodicNode,
    relevant_schema: dict[str, any],
    previous_episodes: list[EpisodicNode],
) -> list[SemanticNode]:
    # Prepare context for LLM
    existing_nodes = [
        {"name": node_name, "label": node_info["label"], "uuid": node_info["uuid"]}
        for node_name, node_info in relevant_schema["nodes"].items()
    ]

    context = {
        "episode_content": episode.content,
        "episode_timestamp": (
            episode.valid_from.isoformat() if episode.valid_from else None
        ),
        "existing_nodes": existing_nodes,
        "previous_episodes": [
            {
                "content": ep.content,
                "timestamp": ep.valid_from.isoformat() if ep.valid_from else None,
            }
            for ep in previous_episodes
        ],
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.v1(context)
    )
    new_nodes_data = llm_response.get("new_nodes", [])

    # Convert the extracted data into SemanticNode objects
    new_nodes = []
    for node_data in new_nodes_data:
        # Check if the node already exists
        if not any(
            existing_node["name"] == node_data["name"]
            for existing_node in existing_nodes
        ):
            new_node = SemanticNode(
                name=node_data["name"],
                labels=node_data["labels"],
                summary=node_data["summary"],
                transaction_from=datetime.now(),
            )
            new_nodes.append(new_node)
            logger.info(f"Created new node: {new_node.name} (UUID: {new_node.uuid})")
        else:
            logger.info(f"Node {node_data['name']} already exists, skipping creation.")

    return new_nodes
