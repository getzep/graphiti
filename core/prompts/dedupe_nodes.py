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
        If any node in New Nodes is a duplicate of a node in Existing Nodes, add their uuids to the output list

        Guidelines:
        1. Use both the name and summary of nodes to determine if they are duplicates, 
            duplicate nodes may have different names
        2. In the output, uuid should always be the uuid of the New Node that is a duplicate. duplicate_of should be
            the uuid of the Existing Node.
        3. Do not confuse a name that is a number (like a year) with the uuid of the node

        Respond with a JSON object in the following format:
        {{
            "duplicates": [
                {{
                    "uuid": "Unique identifier for the node",
                    "duplicate_of": "uuid of the existing node"
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
        1. If any of the nodes in the list are duplicates of each other, group those nodes together in a list
        3. Respond with the resulting list of duplicate node lists

        Guidelines:
        1. Use both the name and summary of nodes to determine if they are duplicates, 
            duplicate nodes may have different names
        2. Each uuid from the list of nodes should appear EXACTLY once in your response
        3. make sure the updated summary is brief

        Respond with a JSON object in the following format:
        {{
            "nodes": [
                {{
                    "uuid": "a97ed1e188834a59b93968fc75e43e71",
                    "duplicate_uuids":  ["uuid of node that is a duplicate of a97ed1e188834a59b93968fc75e43e71"],
                    "summary": "brief summary that combines information from the summaries of the node and its duplicates"
                }}
            ]
        }}
        """,
        ),
    ]


versions: Versions = {"v1": v1, "v2": v2, "node_list": node_list}
