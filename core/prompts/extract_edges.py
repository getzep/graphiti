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
            content="You are a helpful assistant that extracts graph edges from provided context.",
        ),
        Message(
            role="user",
            content=f"""
        Given the following context, extract new semantic edges (relationships) that need to be added to the knowledge graph:
    
        Current Graph Structure:
        {context['relevant_schema']}
    
        New Nodes:
        {json.dumps(context['new_nodes'], indent=2)}
    
        New Episode:
        Content: {context['episode_content']}
        Timestamp: {context['episode_timestamp']}
    
        Previous Episodes:
        {json.dumps([ep['content'] for ep in context['previous_episodes']], indent=2)}
    
        Extract new semantic edges based on the content of the current episode, considering the existing graph structure, new nodes, and context from previous episodes.
    
        Guidelines:
        1. Create edges only between semantic nodes (not episodic nodes like messages).
        2. Each edge should represent a clear relationship between two semantic nodes.
        3. The relation_type should be a concise, all-caps description of the relationship (e.g., LOVES, IS_FRIENDS_WITH, WORKS_FOR).
        4. Provide a more detailed fact describing the relationship.
        5. If a relationship seems to update an existing one, create a new edge with the updated information.
        6. Consider temporal aspects of relationships when relevant.
        7. Do not create edges involving episodic nodes (like Message 1 or Message 2).
        8. Use existing nodes from the current graph structure when appropriate.
    
        Respond with a JSON object in the following format:
        {{
            "new_edges": [
                {{
                    "relation_type": "RELATION_TYPE_IN_CAPS",
                    "source_node": "Name of the source semantic node",
                    "target_node": "Name of the target semantic node",
                    "fact": "Detailed description of the relationship",
                    "valid_at": "YYYY-MM-DDTHH:MM:SSZ or null if not explicitly mentioned",
                    "invalid_at": "YYYY-MM-DDTHH:MM:SSZ or null if ongoing or not explicitly mentioned"
                }}
            ]
        }}
    
        If no new edges need to be added, return an empty list for "new_edges".
        """,
        ),
    ]


versions: Versions = {
    "v1": v1,
}
