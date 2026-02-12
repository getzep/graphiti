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
        description='List of idx values of duplicate facts (only from EXISTING FACTS range). Empty list if none.',
    )
    contradicted_facts: list[int] = Field(
        ...,
        description='List of idx values of contradicted facts (from full idx range). Empty list if none.',
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
        You will receive TWO lists of facts with CONTINUOUS idx numbering across both lists.
        EXISTING FACTS are indexed first, followed by FACT INVALIDATION CANDIDATES.

        1. DUPLICATE DETECTION:
           - If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those idx values in duplicate_facts.
           - Facts with similar information that contain key differences should NOT be marked as duplicates.
           - If no duplicates, return an empty list for duplicate_facts.

        2. CONTRADICTION DETECTION:
           - Determine which facts the NEW FACT contradicts from either list.
           - A fact from EXISTING FACTS can be both a duplicate AND contradicted (e.g., semantically the same but the new fact updates/supersedes it).
           - Return all contradicted idx values in contradicted_facts.
           - If no contradictions, return an empty list for contradicted_facts.

        IMPORTANT:
        - duplicate_facts: ONLY idx values from EXISTING FACTS (cannot include FACT INVALIDATION CANDIDATES)
        - contradicted_facts: idx values from EITHER list (EXISTING FACTS or FACT INVALIDATION CANDIDATES)
        - The idx values are continuous across both lists (INVALIDATION CANDIDATES start where EXISTING FACTS end)

        Guidelines:
        1. Some facts may be very similar but will have key differences, particularly around numeric values.
           Do not mark these as duplicates.

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
