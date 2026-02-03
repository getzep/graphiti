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


class EdgeDuplicate(BaseModel):
    duplicate_facts: list[int] = Field(
        ...,
        description='List of idx values of any duplicate facts. If no duplicate facts are found, default to empty list.',
    )
    contradicted_facts: list[int] = Field(
        ...,
        description='List of idx values of facts that should be invalidated. If no facts should be invalidated, the list should be empty.',
    )


class Prompt(Protocol):
    resolve_edge: PromptVersion


class Versions(TypedDict):
    resolve_edge: PromptFunction


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

        2. CONTRADICTION DETECTION:
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


versions: Versions = {'resolve_edge': resolve_edge}
