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
            content="You are a helpful assistant that de-duplicates nodes from node lists.",
        ),
        Message(
            role="user",
            content=f"""
        Given the following context, deduplicate nodes from a list of new nodes given a list of existing nodes:

        Existing Nodes:
        {json.dumps(context['existing_nodes'], indent=2)}

        New Nodes:
        {json.dumps(context['extracted_nodes'], indent=2)}
        
        Task:
        1. start with the list of nodes from New Nodes
        2. If any node in New Nodes is a duplicate of a node in Existing Nodes, replace the new node with the existing
            node in the list
        3. Respond with the resulting list of nodes

        Guidelines:
        1. Use both the name and summary of nodes to determine if they are duplicates, 
            duplicate nodes may have different names

        Respond with a JSON object in the following format:
        {{
            "new_nodes": [
                {{
                    "name": "Unique identifier for the node",
                    "summary": "Brief summary of the node's role or significance"
                }}
            ]
        }}
        """,
        ),
    ]


versions: Versions = {"v1": v1}
