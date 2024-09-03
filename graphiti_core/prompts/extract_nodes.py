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
    extract_json: PromptVersion


class Versions(TypedDict):
    v1: PromptFunction
    v2: PromptFunction
    extract_json: PromptFunction


def v1(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts graph nodes from provided context.',
        ),
        Message(
            role='user',
            content=f"""
        Given the following context, extract new entity nodes that need to be added to the knowledge graph:

        Previous Episodes:
        {json.dumps([ep['content'] for ep in context['previous_episodes']], indent=2)}

        New Episode:
        Content: {context["episode_content"]}

        Extract new entity nodes based on the content of the current episode, while considering the context from previous episodes.

        Guidelines:
        1. Focus on entities, concepts, or actors that are central to the current episode.
        2. Avoid creating nodes for relationships or actions (these will be handled as edges later).
        3. Provide a brief but informative summary for each node.
        4. Be as explicit as possible in your node names, using full names and avoiding abbreviations.

        Respond with a JSON object in the following format:
        {{
            "new_nodes": [
                {{
                    "name": "Unique identifier for the node",
                    "labels": ["Entity", "OptionalAdditionalLabel"],
                    "summary": "Brief summary of the node's role or significance"
                }}
            ]
        }}

        If no new nodes need to be added, return an empty list for "new_nodes".
        """,
        ),
    ]


def v2(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that extracts entity nodes from conversational text. Your primary task is to identify and extract the speaker and other significant entities mentioned in the conversation."""

    user_prompt = f"""
Given the following conversation, extract entity nodes from the CURRENT MESSAGE that are explicitly or implicitly mentioned:

Conversation:
{json.dumps([ep['content'] for ep in context['previous_episodes']], indent=2)}
<CURRENT MESSAGE>
{context["episode_content"]}

Guidelines:
1. ALWAYS extract the speaker/actor as the first node. The speaker is the part before the colon in each line of dialogue.
2. Extract other significant entities, concepts, or actors mentioned in the conversation.
3. Provide concise but informative summaries for each extracted node.
4. Avoid creating nodes for relationships or actions.
5. Avoid creating nodes for temporal information like dates, times or years (these will be added to edges later).
6. Be as explicit as possible in your node names, using full names and avoiding abbreviations.

Respond with a JSON object in the following format:
{{
    "extracted_nodes": [
        {{
            "name": "Unique identifier for the node (use the speaker's name for speaker nodes)",
            "labels": ["Entity", "Speaker" for speaker nodes, "OptionalAdditionalLabel"],
            "summary": "Brief summary of the node's role or significance"
        }}
    ]
}}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that extracts entity nodes from conversational text. 
    Your primary task is to identify and extract relevant entities from JSON files"""

    user_prompt = f"""
Given the following source description, extract relevant entity nodes from the provided JSON:

Source Description:
{context["source_description"]}

JSON:
{context["episode_content"]}

Guidelines:
1. Always try to extract an entities that the JSON represents. This will often be something like a "name" or "user field
2. Do NOT extract any properties that contain dates

Respond with a JSON object in the following format:
{{
    "extracted_nodes": [
        {{
            "name": "Unique identifier for the node (use the speaker's name for speaker nodes)",
            "labels": ["Entity", "Speaker" for speaker nodes, "OptionalAdditionalLabel"],
            "summary": "Brief summary of the node's role or significance"
        }}
    ]
}}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


versions: Versions = {'v1': v1, 'v2': v2, 'extract_json': extract_json}
