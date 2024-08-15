import json
from typing import TypedDict, Protocol

from .models import Message, PromptVersion, PromptFunction


class Prompt(Protocol):
    v1: PromptVersion


class Versions(TypedDict):
    v1: PromptFunction


def v1(context: dict[str, any]) -> list[Message]:
    return [
        Message(
            role="system",
            content="You are a helpful assistant that extracts graph nodes from provided context.",
        ),
        Message(
            role="user",
            content=f"""
        Given the following context, extract new semantic nodes that need to be added to the knowledge graph:
    
        Existing Nodes:
        {json.dumps(context['existing_nodes'], indent=2)}
    
        Previous Episodes:
        {json.dumps([ep['content'] for ep in context['previous_episodes']], indent=2)}
    
        New Episode:
        Content: {context["episode_content"]}
        Timestamp: {context['episode_timestamp']}
    
        Extract new semantic nodes based on the content of the current episode, while considering the existing nodes and context from previous episodes.
    
        Guidelines:
        1. Only extract new nodes that don't already exist in the graph structure.
        2. Focus on entities, concepts, or actors that are central to the current episode.
        3. Avoid creating nodes for relationships or actions (these will be handled as edges later).
        4. Provide a brief but informative summary for each node.
        5. If a node seems to represent an existing concept but with updated information, don't create a new node. This will be handled by edge updates.
        6. Do not create nodes for episodic content (like Message 1 or Message 2).
    
        Respond with a JSON object in the following format:
        {{
            "new_nodes": [
                {{
                    "name": "Unique identifier for the node",
                    "labels": ["Semantic", "OptionalAdditionalLabel"],
                    "summary": "Brief summary of the node's role or significance"
                }}
            ]
        }}
    
        If no new nodes need to be added, return an empty list for "new_nodes".
        """,
        ),
    ]


versions: Versions = {
    "v1": v1,
}
