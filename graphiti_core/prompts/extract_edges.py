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


class Versions(TypedDict):
    v1: PromptFunction
    v2: PromptFunction


def v1(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts graph edges from provided context.',
        ),
        Message(
            role='user',
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


def v2(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts graph edges from provided context.',
        ),
        Message(
            role='user',
            content=f"""
        Given the following context, extract edges (relationships) that need to be added to the knowledge graph:
        Nodes:
        {json.dumps(context['nodes'], indent=2)}

        

        Episodes:
        {json.dumps([ep['content'] for ep in context['previous_episodes']], indent=2)}
        {context['episode_content']} <-- New Episode
        

        Extract entity edges based on the content of the current episode, the given nodes, and context from previous episodes.

        Guidelines:
        1. Create edges only between the provided nodes.
        2. Each edge should represent a clear relationship between two DISTINCT nodes.
        3. The relation_type should be a concise, all-caps description of the relationship (e.g., LOVES, IS_FRIENDS_WITH, WORKS_FOR).
        4. Provide a more detailed fact describing the relationship.
        5. Consider temporal aspects of relationships when relevant.
        6. Avoid using the same node as the source and target of a relationship

        Respond with a JSON object in the following format:
        {{
            "edges": [
                {{
                    "relation_type": "RELATION_TYPE_IN_CAPS",
                    "source_node_uuid": "uuid of the source entity node",
                    "target_node_uuid": "uuid of the target entity node",
                    "fact": "brief description of the relationship",
                    "valid_at": "YYYY-MM-DDTHH:MM:SSZ or null if not explicitly mentioned",
                    "invalid_at": "YYYY-MM-DDTHH:MM:SSZ or null if ongoing or not explicitly mentioned"
                }}
            ]
        }}

        If no edges need to be added, return an empty list for "edges".
        """,
        ),
    ]


versions: Versions = {'v1': v1, 'v2': v2}
