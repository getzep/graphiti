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

from abc import ABC, abstractmethod
from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field

from .models import ChatPrompt, Message, PromptFunction, PromptVersion
from .prompt_helpers import to_prompt_json
from .versions import PromptTemplate, render_template


class NodeDuplicate(BaseModel):
    id: int = Field(..., description='integer id of the entity')
    name: str = Field(
        ...,
        description='Name of the entity. Should be the most complete and descriptive name of the entity. Do not include any JSON formatting in the Entity name such as {}.',
    )
    duplicate_candidate_id: int = Field(
        ...,
        description='candidate_id of the matching EXISTING ENTITY, or -1 if no duplicate exists.',
    )


class NodeResolutions(BaseModel):
    entity_resolutions: list[NodeDuplicate] = Field(..., description='List of resolved nodes')


class Prompt(Protocol):
    node: PromptVersion
    node_list: PromptVersion
    nodes: PromptVersion


class Versions(TypedDict):
    node: PromptFunction
    node_list: PromptFunction
    nodes: PromptFunction


NODE_SYSTEM_PROMPT = """You are an entity deduplication assistant. NEVER fabricate entity names or mark distinct entities as duplicates."""
NODE_USER_TEMPLATE = """
<PREVIOUS MESSAGES>
{previous_episodes}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{episode_content}
</CURRENT MESSAGE>

<NEW ENTITY>
{extracted_node}
</NEW ENTITY>

<ENTITY TYPE DESCRIPTION>
{entity_type_description}
</ENTITY TYPE DESCRIPTION>

<EXISTING ENTITIES>
{existing_nodes}
</EXISTING ENTITIES>

Entities should only be considered duplicates if they refer to the *same real-world object or concept*.
Semantic Equivalence: if a descriptive label in EXISTING ENTITIES clearly refers to a named entity in context, treat them as duplicates.

NEVER mark entities as duplicates if:
- They are related but distinct.
- They have similar names or purposes but refer to separate instances or concepts.

Task:
1. Compare the NEW ENTITY against each EXISTING ENTITY (identified by `candidate_id`).
2. If it refers to the same real-world object or concept, return the `candidate_id` of that match.
3. Return `duplicate_candidate_id = -1` when there is no match or you are unsure.

<EXAMPLE>
NEW ENTITY: "Sam" (Person)
EXISTING ENTITIES: [{"candidate_id": 0, "name": "Sam", "entity_types": ["Person"], "summary": "Sam enjoys hiking and photography"}]
Result: duplicate_candidate_id = 0 (same person referenced in conversation)

NEW ENTITY: "NYC"
EXISTING ENTITIES: [{"candidate_id": 0, "name": "New York City", "entity_types": ["Location"]}, {"candidate_id": 1, "name": "New York Knicks", "entity_types": ["Organization"]}]
Result: duplicate_candidate_id = 0 (same location, abbreviated name)

NEW ENTITY: "Java" (programming language)
EXISTING ENTITIES: [{"candidate_id": 0, "name": "Java", "entity_types": ["Location"], "summary": "An island in Indonesia"}]
Result: duplicate_candidate_id = -1 (same name but distinct real-world things)

NEW ENTITY: "Marco's car"
EXISTING ENTITIES: [{"candidate_id": 0, "name": "Marco's vehicle", "entity_types": ["Entity"], "summary": "Marco drives a red sedan."}]
Result: duplicate_candidate_id = 0 (synonym — "car" and "vehicle" refer to the same thing, same possessor)
</EXAMPLE>
"""


def node(context: dict[str, Any]) -> list[Message]:
    return [
        Message(role='system', content=NODE_SYSTEM_PROMPT),
        Message(
            role='user',
            content=render_template(
                NODE_USER_TEMPLATE,
                previous_episodes=to_prompt_json(context['previous_episodes']),
                episode_content=context['episode_content'],
                extracted_node=to_prompt_json(context['extracted_node']),
                entity_type_description=to_prompt_json(context['entity_type_description']),
                existing_nodes=to_prompt_json(context['existing_nodes']),
            ),
        ),
    ]


NODES_SYSTEM_PROMPT = NODE_SYSTEM_PROMPT
NODES_USER_TEMPLATE = """
<PREVIOUS MESSAGES>
{previous_episodes}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{episode_content}
</CURRENT MESSAGE>

<ENTITIES>
{extracted_nodes}
</ENTITIES>

<EXISTING ENTITIES>
{existing_nodes}
</EXISTING ENTITIES>

Each of the above ENTITIES was extracted from the CURRENT MESSAGE.
For each entity, determine if it is a duplicate of any EXISTING ENTITY.
Entities should only be considered duplicates if they refer to the *same real-world object or concept*.

NEVER mark entities as duplicates if:
- They are related but distinct.
- They have similar names or purposes but refer to separate instances or concepts.

Task:
ENTITIES contains {entity_count} entities with IDs 0 through {last_entity_id}.
Your response MUST include EXACTLY {entity_count} resolutions with IDs 0 through {last_entity_id}. Do not skip or add IDs.

For every entity, provide:
- `id`: integer id from ENTITIES
- `name`: the best full name for the entity (preserve the original name unless a duplicate has a more complete name)
- `duplicate_candidate_id`: the `candidate_id` of the EXISTING ENTITY that is the best duplicate match, or -1 if there is no duplicate

<EXAMPLE>
ENTITY: "Sam" (Person)
EXISTING ENTITIES: [{"candidate_id": 0, "name": "Sam", "entity_types": ["Person"], "summary": "Sam enjoys hiking and photography"}]
Result: duplicate_candidate_id = 0 (same person referenced in conversation)

ENTITY: "NYC"
EXISTING ENTITIES: [{"candidate_id": 0, "name": "New York City", "entity_types": ["Location"]}, {"candidate_id": 1, "name": "New York Knicks", "entity_types": ["Organization"]}]
Result: duplicate_candidate_id = 0 (same location, abbreviated name)

ENTITY: "Java" (programming language)
EXISTING ENTITIES: [{"candidate_id": 0, "name": "Java", "entity_types": ["Location"], "summary": "An island in Indonesia"}]
Result: duplicate_candidate_id = -1 (same name but distinct real-world things)

ENTITY: "Marco's car"
EXISTING ENTITIES: [{"candidate_id": 0, "name": "Marco's vehicle", "entity_types": ["Entity"], "summary": "Marco drives a red sedan."}]
Result: duplicate_candidate_id = 0 (synonym — "car" and "vehicle" refer to the same thing, same possessor)
</EXAMPLE>
"""


def nodes(context: dict[str, Any]) -> list[Message]:
    entity_count = len(context['extracted_nodes'])
    return [
        Message(role='system', content=NODES_SYSTEM_PROMPT),
        Message(
            role='user',
            content=render_template(
                NODES_USER_TEMPLATE,
                previous_episodes=to_prompt_json(context['previous_episodes']),
                episode_content=context['episode_content'],
                extracted_nodes=to_prompt_json(context['extracted_nodes']),
                existing_nodes=to_prompt_json(context['existing_nodes']),
                entity_count=str(entity_count),
                last_entity_id=str(entity_count - 1),
            ),
        ),
    ]


NODE_LIST_SYSTEM_PROMPT = (
    'You are an entity deduplication assistant that groups duplicate nodes by UUID.'
)
NODE_LIST_USER_TEMPLATE = """
Given the following context, deduplicate a list of nodes:

<NODES>
{nodes}
</NODES>

Task:
1. Group nodes together such that all duplicate nodes are in the same list of uuids.
2. All duplicate uuids should be grouped together in the same list.
3. Also return a new summary that synthesizes the summaries into a new short summary.

Guidelines:
1. Each uuid from the list of nodes should appear EXACTLY once in your response.
2. If a node has no duplicates, it should appear in the response in a list of only one uuid.

<EXAMPLE>
Input nodes:
[
  {"uuid": "a1", "name": "NYC", "summary": "New York City"},
  {"uuid": "b2", "name": "New York City", "summary": "The city of New York"},
  {"uuid": "c3", "name": "Los Angeles", "summary": "City in California"}
]

Result:
[
  {"uuids": ["a1", "b2"], "summary": "New York City, also known as NYC"},
  {"uuids": ["c3"], "summary": "City in California"}
]
</EXAMPLE>
"""


def node_list(context: dict[str, Any]) -> list[Message]:
    return [
        Message(role='system', content=NODE_LIST_SYSTEM_PROMPT),
        Message(
            role='user',
            content=render_template(
                NODE_LIST_USER_TEMPLATE, nodes=to_prompt_json(context['nodes'])
            ),
        ),
    ]


NODE_TEMPLATE = PromptTemplate((NODE_SYSTEM_PROMPT, NODE_USER_TEMPLATE))
NODES_TEMPLATE = PromptTemplate((NODES_SYSTEM_PROMPT, NODES_USER_TEMPLATE))
NODE_LIST_TEMPLATE = PromptTemplate((NODE_LIST_SYSTEM_PROMPT, NODE_LIST_USER_TEMPLATE))
node.__prompt_template__ = NODE_TEMPLATE
nodes.__prompt_template__ = NODES_TEMPLATE
node_list.__prompt_template__ = NODE_LIST_TEMPLATE

versions: Versions = {'node': node, 'node_list': node_list, 'nodes': nodes}


class DedupeNodesPrompts(ABC):
    @abstractmethod
    def node(self, context: dict[str, Any]) -> ChatPrompt: ...

    @abstractmethod
    def node_list(self, context: dict[str, Any]) -> ChatPrompt: ...

    @abstractmethod
    def nodes(self, context: dict[str, Any]) -> ChatPrompt: ...


class DefaultDedupeNodesPrompts(DedupeNodesPrompts):
    def node(self, context: dict[str, Any]) -> ChatPrompt:
        return ChatPrompt.from_messages(node(context))

    def node_list(self, context: dict[str, Any]) -> ChatPrompt:
        return ChatPrompt.from_messages(node_list(context))

    def nodes(self, context: dict[str, Any]) -> ChatPrompt:
        return ChatPrompt.from_messages(nodes(context))
