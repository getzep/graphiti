"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
from typing import Any, Protocol, TypedDict

from .models import Message, PromptFunction, PromptVersion


class Prompt(Protocol):
    v1: PromptVersion
    v2: PromptVersion
    node_list: PromptVersion


class Versions(TypedDict):
    v1: PromptFunction
    v2: PromptFunction
    node_list: PromptVersion


def v1(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that de-duplicates nodes from node lists.',
        ),
        Message(
            role='user',
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
        3. when deduplicating nodes, synthesize their summaries into a short new summary that contains the relevant information
            of the summaries of the new and existing nodes
        4. Respond with the resulting list of nodes

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


def v2(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that de-duplicates nodes from node lists.',
        ),
        Message(
            role='user',
            content=f"""
        Given the following context, deduplicate nodes from a list of new nodes given a list of existing nodes:

        Existing Nodes:
        {json.dumps(context['existing_nodes'], indent=2)}

        New Nodes:
        {json.dumps(context['extracted_nodes'], indent=2)}
        Important:
        If a node in the new nodes is describing the same entity as a node in the existing nodes, mark it as a duplicate!!!
        Task:
        If any node in New Nodes is a duplicate of a node in Existing Nodes, add their names to the output list
        When finding duplicates nodes, synthesize their summaries into a short new summary that contains the 
        relevant information of the summaries of the new and existing nodes.

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
                    "duplicate_of": "name of the existing node",
                    "summary": "Brief summary of the node's role or significance. Takes information from the new and existing nodes"
                }}
            ]
        }}
        """,
        ),
    ]


def node_list(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that de-duplicates nodes from node lists.',
        ),
        Message(
            role='user',
            content=f"""
        Given the following context, deduplicate a list of nodes:

        Nodes:
        {json.dumps(context['nodes'], indent=2)}

        Task:
        1. Group nodes together such that all duplicate nodes are in the same list of names
        2. All duplicate names should be grouped together in the same list
        3. Also return a new summary that synthesizes the summary into a new short summary

        Guidelines:
        1. Each name from the list of nodes should appear EXACTLY once in your response
        2. If a node has no duplicates, it should appear in the response in a list of only one name

        Respond with a JSON object in the following format:
        {{
            "nodes": [
                {{
                    "names": ["myNode", "node that is a duplicate of myNode"],
                    "summary": "Brief summary of the node summaries that appear in the list of names."
                }}
            ]
        }}
        """,
        ),
    ]


versions: Versions = {'v1': v1, 'v2': v2, 'node_list': node_list}
