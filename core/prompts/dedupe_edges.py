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
            content="You are a helpful assistant that de-duplicates relationship from edge lists.",
        ),
        Message(
            role="user",
            content=f"""
        Given the following context, deduplicate edges from a list of new edges given a list of existing edges:

        Existing Edges:
        {json.dumps(context['existing_edges'], indent=2)}

        New Edges:
        {json.dumps(context['extracted_edges'], indent=2)}

        Task:
        1. start with the list of edges from New Edges
        2. If any edge in New Edges is a duplicate of an edge in Existing Edges, replace the new edge with the existing
            edge in the list
        3. Respond with the resulting list of edges

        Guidelines:
        1. Use both the name and fact of edges to determine if they are duplicates, 
            duplicate edges may have different names

        Respond with a JSON object in the following format:
        {{
            "new_edges": [
                {{
                    "name": "Unique identifier for the edge",
                    "fact": "one sentence description of the fact"
                }}
            ]
        }}
        """,
        ),
    ]


versions: Versions = {"v1": v1}
