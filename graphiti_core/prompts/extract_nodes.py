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


class ExtractedNodes(BaseModel):
    extracted_node_names: list[str] = Field(..., description='Name of the extracted entity')


class MissedEntities(BaseModel):
    missed_entities: list[str] = Field(..., description="Names of entities that weren't extracted")


class EntityClassificationTriple(BaseModel):
    uuid: str = Field(description='UUID of the entity')
    name: str = Field(description='Name of the entity')
    entity_type: str | None = Field(
        default=None, description='Type of the entity. Must be one of the provided types or None'
    )


class EntityClassification(BaseModel):
    entity_classifications: list[EntityClassificationTriple] = Field(
        ..., description='List of entities classification triples.'
    )


class Prompt(Protocol):
    extract_message: PromptVersion
    extract_json: PromptVersion
    extract_text: PromptVersion
    reflexion: PromptVersion
    classify_nodes: PromptVersion


class Versions(TypedDict):
    extract_message: PromptFunction
    extract_json: PromptFunction
    extract_text: PromptFunction
    reflexion: PromptFunction
    classify_nodes: PromptFunction


def extract_message(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that extracts entity nodes from conversational messages. Your primary task is to identify and extract the speaker and other significant entities mentioned in the conversation."""

    user_prompt = f"""
<PREVIOUS MESSAGES>
{json.dumps([ep for ep in context['previous_episodes']], indent=2)}
</PREVIOUS MESSAGES>
<CURRENT MESSAGE>
{context["episode_content"]}
</CURRENT MESSAGE>

{context['custom_prompt']}

Given the above conversation, extract entity nodes from the CURRENT MESSAGE that are explicitly or implicitly mentioned:

Guidelines:
1. ALWAYS extract the speaker/actor as the first node. The speaker is the part before the colon in each line of dialogue.
2. Extract other significant entities, concepts, or actors mentioned in the CURRENT MESSAGE.
3. DO NOT create nodes for relationships or actions.
4. DO NOT create nodes for temporal information like dates, times or years (these will be added to edges later).
5. Be as explicit as possible in your node names, using full names.
6. DO NOT extract entities mentioned only in PREVIOUS MESSAGES, those messages are only to provide context.
7. Extract preferences as their own nodes
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that extracts entity nodes from JSON. 
    Your primary task is to identify and extract relevant entities from JSON files"""

    user_prompt = f"""
<SOURCE DESCRIPTION>:
{context["source_description"]}
</SOURCE DESCRIPTION>
<JSON>
{context["episode_content"]}
</JSON>

{context['custom_prompt']}

Given the above source description and JSON, extract relevant entity nodes from the provided JSON:

Guidelines:
1. Always try to extract an entities that the JSON represents. This will often be something like a "name" or "user field
2. Do NOT extract any properties that contain dates
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that extracts entity nodes from text. Your primary task is to identify and extract the speaker and other significant entities mentioned in the provided text."""

    user_prompt = f"""
<TEXT>
{context["episode_content"]}
</TEXT>

{context['custom_prompt']}

Given the above text, extract entity nodes from the TEXT that are explicitly or implicitly mentioned:

Guidelines:
1. Extract significant entities, concepts, or actors mentioned in the conversation.
2. Avoid creating nodes for relationships or actions.
3. Avoid creating nodes for temporal information like dates, times or years (these will be added to edges later).
4. Be as explicit as possible in your node names, using full names and avoiding abbreviations.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which entities have not been extracted from the given context"""

    user_prompt = f"""
<PREVIOUS MESSAGES>
{json.dumps([ep for ep in context['previous_episodes']], indent=2)}
</PREVIOUS MESSAGES>
<CURRENT MESSAGE>
{context["episode_content"]}
</CURRENT MESSAGE>

<EXTRACTED ENTITIES>
{context["extracted_entities"]}
</EXTRACTED ENTITIES>

Given the above previous messages, current message, and list of extracted entities; determine if any entities haven't been
extracted.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def classify_nodes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that classifies entity nodes given the context from which they were extracted"""

    user_prompt = f"""
    <PREVIOUS MESSAGES>
    {json.dumps([ep for ep in context['previous_episodes']], indent=2)}
    </PREVIOUS MESSAGES>
    <CURRENT MESSAGE>
    {context["episode_content"]}
    </CURRENT MESSAGE>
    
    <EXTRACTED ENTITIES>
    {context['extracted_entities']}
    </EXTRACTED ENTITIES>
    
    <ENTITY TYPES>
    {context['entity_types']}
    </ENTITY TYPES>
    
    Given the above conversation, extracted entities, and provided entity types and their descriptions, classify the extracted entities.
    
    Guidelines:
    1. Each entity must have exactly one type
    2. Only use the provided ENTITY TYPES as types, do not use additional types to classify entities.
    3. If none of the provided entity types accurately classify an extracted node, the type should be set to None
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


versions: Versions = {
    'extract_message': extract_message,
    'extract_json': extract_json,
    'extract_text': extract_text,
    'reflexion': reflexion,
    'classify_nodes': classify_nodes,
}
