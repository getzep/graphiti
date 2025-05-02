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

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion


class NodeDuplicate(BaseModel):
    duplicate_node_id: int = Field(
        ...,
        description='id of the duplicate node. If no duplicate nodes are found, default to -1.',
    )
    name: str = Field(..., description='Name of the entity.')


class Prompt(Protocol):
    node: PromptVersion
    node_list: PromptVersion


class Versions(TypedDict):
    node: PromptFunction
    node_list: PromptFunction


def node(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that de-duplicates entities from entity lists.',
        ),
        Message(
            role='user',
            content=f"""
        <PREVIOUS MESSAGES>
        {json.dumps([ep for ep in context['previous_episodes']], indent=2)}
        </PREVIOUS MESSAGES>
        <CURRENT MESSAGE>
        {context['episode_content']}
        </CURRENT MESSAGE>
        <NEW ENTITY>
        {json.dumps(context['extracted_node'], indent=2)}
        </NEW ENTITY>
        <ENTITY TYPE DESCRIPTION>
        {json.dumps(context['entity_type_description'], indent=2)}
        </ENTITY TYPE DESCRIPTION>

        <EXISTING ENTITIES>
        {json.dumps(context['existing_nodes'], indent=2)}
        </EXISTING ENTITIES>
        
        Given the above EXISTING ENTITIES and their attributes, MESSAGE, and PREVIOUS MESSAGES; Determine if the NEW ENTITY extracted from the conversation
        is a duplicate entity of one of the EXISTING ENTITIES.
        
        The ENTITY TYPE DESCRIPTION gives more insight into what the entity type means for the NEW ENTITY.

        Task:
        If the NEW ENTITY represents a duplicate entity of any entity in EXISTING ENTITIES, set duplicate_entity_id to the
        id of the EXISTING ENTITY that is the duplicate. If the NEW ENTITY is not a duplicate of any of the EXISTING ENTITIES,
        duplicate_entity_id should be set to -1.
        
        Also return the most complete name for the entity.
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
        1. Group nodes together such that all duplicate nodes are in the same list of uuids
        2. All duplicate uuids should be grouped together in the same list
        3. Also return a new summary that synthesizes the summary into a new short summary

        Guidelines:
        1. Each uuid from the list of nodes should appear EXACTLY once in your response
        2. If a node has no duplicates, it should appear in the response in a list of only one uuid

        Respond with a JSON object in the following format:
        {{
            "nodes": [
                {{
                    "uuids": ["5d643020624c42fa9de13f97b1b3fa39", "node that is a duplicate of 5d643020624c42fa9de13f97b1b3fa39"],
                    "summary": "Brief summary of the node summaries that appear in the list of names."
                }}
            ]
        }}
        """,
        ),
    ]


versions: Versions = {'node': node, 'node_list': node_list}
