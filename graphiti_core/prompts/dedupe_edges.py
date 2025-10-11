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

from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion
from .prompt_helpers import to_prompt_json


class EdgeDuplicate(BaseModel):
    duplicate_facts: list[int] = Field(
        ...,
        description='List of idx values of any duplicate facts. If no duplicate facts are found, default to empty list.',
    )
    contradicted_facts: list[int] = Field(
        ...,
        description='List of idx values of facts that should be invalidated. If no facts should be invalidated, the list should be empty.',
    )
    fact_type: str = Field(..., description='One of the provided fact types or DEFAULT')


class UniqueFact(BaseModel):
    uuid: str = Field(..., description='unique identifier of the fact')
    fact: str = Field(..., description='fact of a unique edge')


class UniqueFacts(BaseModel):
    unique_facts: list[UniqueFact]


class Prompt(Protocol):
    edge: PromptVersion
    edge_list: PromptVersion
    resolve_edge: PromptVersion


class Versions(TypedDict):
    edge: PromptFunction
    edge_list: PromptFunction
    resolve_edge: PromptFunction


def edge(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that de-duplicates edges from edge lists.',
        ),
        Message(
            role='user',
            content=f"""
        Given the following context, determine whether the New Edge represents any of the edges in the list of Existing Edges.

        <EXISTING EDGES>
        {to_prompt_json(context['related_edges'])}
        </EXISTING EDGES>

        <NEW EDGE>
        {to_prompt_json(context['extracted_edges'])}
        </NEW EDGE>

        Task:
        If the New Edges represents the same factual information as any edge in Existing Edges, return the id of the duplicate fact
            as part of the list of duplicate_facts.
        If the NEW EDGE is not a duplicate of any of the EXISTING EDGES, return an empty list.

        Guidelines:
        1. The facts do not need to be completely identical to be duplicates, they just need to express the same information.
        """,
        ),
    ]


def edge_list(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that de-duplicates edges from edge lists.',
        ),
        Message(
            role='user',
            content=f"""
        Given the following context, find all of the duplicates in a list of facts:

        Facts:
        {to_prompt_json(context['edges'])}

        Task:
        If any facts in Facts is a duplicate of another fact, return a new fact with one of their uuid's.

        Guidelines:
        1. identical or near identical facts are duplicates
        2. Facts are also duplicates if they are represented by similar sentences
        3. Facts will often discuss the same or similar relation between identical entities
        4. The final list should have only unique facts. If 3 facts are all duplicates of each other, only one of their
            facts should be in the response
        """,
        ),
    ]


def resolve_edge(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that de-duplicates facts from fact lists and determines which existing '
            'facts are contradicted by the new fact.',
        ),
        Message(
            role='user',
            content=f"""
        Task:
        You will receive TWO separate lists of facts. Each list uses 'idx' as its index field, starting from 0.

        1. DUPLICATE DETECTION:
           - If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those idx values in duplicate_facts.
           - Facts with similar information that contain key differences should NOT be marked as duplicates.
           - Return idx values from EXISTING FACTS.
           - If no duplicates, return an empty list for duplicate_facts.

        2. FACT TYPE CLASSIFICATION:
           - Given the predefined FACT TYPES, determine if the NEW FACT should be classified as one of these types.
           - Return the fact type as fact_type or DEFAULT if NEW FACT is not one of the FACT TYPES.

        3. CONTRADICTION DETECTION:
           - Based on FACT INVALIDATION CANDIDATES and NEW FACT, determine which facts the new fact contradicts.
           - Return idx values from FACT INVALIDATION CANDIDATES.
           - If no contradictions, return an empty list for contradicted_facts.

        IMPORTANT:
        - duplicate_facts: Use ONLY 'idx' values from EXISTING FACTS
        - contradicted_facts: Use ONLY 'idx' values from FACT INVALIDATION CANDIDATES
        - These are two separate lists with independent idx ranges starting from 0

        Guidelines:
        1. Some facts may be very similar but will have key differences, particularly around numeric values in the facts.
            Do not mark these facts as duplicates.

        <FACT TYPES>
        {context['edge_types']}
        </FACT TYPES>

        <EXISTING FACTS>
        {context['existing_edges']}
        </EXISTING FACTS>

        <FACT INVALIDATION CANDIDATES>
        {context['edge_invalidation_candidates']}
        </FACT INVALIDATION CANDIDATES>

        <NEW FACT>
        {context['new_edge']}
        </NEW FACT>
        """,
        ),
    ]


versions: Versions = {'edge': edge, 'edge_list': edge_list, 'resolve_edge': resolve_edge}
