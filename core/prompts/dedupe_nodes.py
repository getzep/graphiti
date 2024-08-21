import json
from typing import TypedDict, Protocol

from .models import Message, PromptVersion, PromptFunction


class Prompt(Protocol):
    v1: PromptVersion
    v2: PromptVersion
    node_list: PromptVersion


class Versions(TypedDict):
    v1: PromptFunction
    v2: PromptFunction
    node_list: PromptVersion


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
                }}
            ]
        }}
        """,
        ),
    ]


def v2(context: dict[str, any]) -> list[Message]:
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
        If any node in New Nodes is a duplicate of a node in Existing Nodes, add their names to the output list

        Guidelines:
        1. Use both the name and summary of nodes to determine if they are duplicates, 
            duplicate nodes may have different names
        2. In the output, name should always be the name of the New Node that is a duplicate. duplicate_of should be
            the name of the Existing Node.

        Respond with a JSON object in the following format:
        {{
            "duplicates": [
                {{
                    "name": "name of the new node",
                    "duplicate_of": "name of the existing node"
                }}
            ]
        }}
        """,
        ),
    ]


def node_list(context: dict[str, any]) -> list[Message]:
    return [
        Message(
            role="system",
            content="You are a helpful assistant that de-duplicates nodes from node lists.",
        ),
        Message(
            role="user",
            content=f"""
        Given the following context, deduplicate a list of nodes:

        Nodes:
        {json.dumps(context['nodes'], indent=2)}

        Task:
        1. Group nodes together such that all duplicate nodes are in the same list of names
        2. All dupolicate names should be grouped together in the same list

        Guidelines:
        1. Each name from the list of nodes should appear EXACTLY once in your response
        2. If a node has no duplicates, it should appear in the response in a list of only one name

        Respond with a JSON object in the following format:
        {{
            "nodes": [
                {{
                    "names": ["myNode", "node that is a duplicate of myNode"],
                }}
            ]
        }}
        """,
        ),
    ]


versions: Versions = {"v1": v1, "v2": v2, "node_list": node_list}
