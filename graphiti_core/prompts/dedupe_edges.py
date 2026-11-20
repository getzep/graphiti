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
from .versions import PromptTemplate, render_template


class EdgeDuplicate(BaseModel):
    duplicate_facts: list[int] = Field(
        ...,
        description='List of idx values of duplicate facts (only from EXISTING FACTS range). Empty list if none.',
    )
    contradicted_facts: list[int] = Field(
        ...,
        description='List of idx values of contradicted facts (from full idx range). Empty list if none.',
    )


class EdgeDuplicates(BaseModel):
    duplicate_facts: list[int] = Field(
        ..., description='List of duplicate idx values from EXISTING FACTS. Empty list if none.'
    )


class EdgeContradictions(BaseModel):
    contradicted_facts: list[int] = Field(
        ...,
        description='List of contradicted idx values from the full idx range. Empty list if none.',
    )


class Prompt(Protocol):
    resolve_edge: PromptVersion
    resolve_duplicates: PromptVersion
    resolve_contradictions: PromptVersion


class Versions(TypedDict):
    resolve_edge: PromptFunction
    resolve_duplicates: PromptFunction
    resolve_contradictions: PromptFunction


RESOLVE_EDGE_SYSTEM_PROMPT = """You are a fact deduplication assistant. NEVER mark facts as contradicted unless the new fact explicitly makes the old fact false."""
RESOLVE_EDGE_USER_TEMPLATE = """
NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers.

Only mark a fact as contradicted when the NEW FACT and the existing fact cannot both be true.
Do NOT treat additional activity, progress, status, or project-update facts as contradictions of role,
ownership, responsibility, membership, or assignment facts. A person doing work on a project is
compatible with that person remaining responsible for, owning, leading, or being the DRI for the project.

IMPORTANT constraints:
- duplicate_facts: ONLY idx values from EXISTING FACTS (NEVER include FACT INVALIDATION CANDIDATES)
- contradicted_facts: idx values from EITHER list (EXISTING FACTS or FACT INVALIDATION CANDIDATES)
- The idx values are continuous across both lists (INVALIDATION CANDIDATES start where EXISTING FACTS end)

<EXISTING FACTS>
{existing_edges}
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
{edge_invalidation_candidates}
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
{new_edge}
</NEW FACT>

You will receive TWO lists of facts with CONTINUOUS idx numbering across both lists.
EXISTING FACTS are indexed first, followed by FACT INVALIDATION CANDIDATES.

1. DUPLICATE DETECTION:
   - If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those idx values in duplicate_facts.
   - If no duplicates, return an empty list for duplicate_facts.

2. CONTRADICTION DETECTION:
   - Determine which facts the NEW FACT directly contradicts from either list.
   - A contradiction requires explicit evidence that the old fact stopped being true or was replaced,
     such as clear termination, replacement, transfer, or mutually exclusive current-state language.
   - Do NOT infer contradiction from recency alone, episode dates, meeting dates, weekly snapshots,
     section headings such as "updates" or "status changes", or a new event involving the same entities.
   - Do NOT mark a role/ownership/responsibility/assignment fact as contradicted merely because the
     NEW FACT describes work performed, updates shipped, documents created, progress made, blockers,
     next steps, or collaboration involving the same person and project.
   - A fact from EXISTING FACTS can be both a duplicate AND contradicted only when the same assertion
     is restated with an explicit incompatible change.
   - Return all contradicted idx values in contradicted_facts.
   - If no contradictions, return an empty list for contradicted_facts.

<EXAMPLE>
EXISTING FACT: idx=0, "Alice joined Acme Corp in 2020"
NEW FACT: "Alice joined Acme Corp in 2020"
Result: duplicate_facts=[0], contradicted_facts=[] (identical factual information)

EXISTING FACT: idx=1, "Alice works at Acme Corp as a software engineer"
NEW FACT: "Alice works at Acme Corp as a senior engineer"
Result: duplicate_facts=[], contradicted_facts=[1] (same relationship but updated title — contradiction, NOT a duplicate)

EXISTING FACT: idx=2, "Bob ran 5 miles on Tuesday"
NEW FACT: "Bob ran 3 miles on Wednesday"
Result: duplicate_facts=[], contradicted_facts=[] (different events on different days — neither duplicate nor contradiction)
</EXAMPLE>
"""


def resolve_edge(context: dict[str, Any]) -> list[Message]:
    return [
        Message(role='system', content=RESOLVE_EDGE_SYSTEM_PROMPT),
        Message(
            role='user',
            content=render_template(
                RESOLVE_EDGE_USER_TEMPLATE,
                existing_edges=context['existing_edges'],
                edge_invalidation_candidates=context['edge_invalidation_candidates'],
                new_edge=context['new_edge'],
            ),
        ),
    ]


RESOLVE_EDGE_TEMPLATE = PromptTemplate((RESOLVE_EDGE_SYSTEM_PROMPT, RESOLVE_EDGE_USER_TEMPLATE))
resolve_edge.__prompt_template__ = RESOLVE_EDGE_TEMPLATE

DUPLICATES_SYSTEM_PROMPT = 'You are a fact deduplication assistant. Return only duplicate facts.'
DUPLICATES_USER_TEMPLATE = """
NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers.

<EXISTING FACTS>
{existing_edges}
</EXISTING FACTS>

<NEW FACT>
{new_edge}
</NEW FACT>

If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those idx values in duplicate_facts.
duplicate_facts must ONLY contain idx values from EXISTING FACTS. If no duplicates, return an empty list.
Do not return contradiction decisions.
"""
CONTRADICTIONS_SYSTEM_PROMPT = RESOLVE_EDGE_SYSTEM_PROMPT
CONTRADICTIONS_USER_TEMPLATE = """
Only mark a fact as contradicted when the NEW FACT and the existing fact cannot both be true.
Do NOT treat additional activity, progress, status, or project-update facts as contradictions of role,
ownership, responsibility, membership, or assignment facts. A person doing work on a project is
compatible with that person remaining responsible for, owning, leading, or being the DRI for the project.

<EXISTING FACTS>
{existing_edges}
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
{edge_invalidation_candidates}
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
{new_edge}
</NEW FACT>

The idx values are continuous across both lists. INVALIDATION CANDIDATES start where EXISTING FACTS end.
Determine which facts the NEW FACT directly contradicts from either list.
A contradiction requires explicit evidence that the old fact stopped being true or was replaced,
such as clear termination, replacement, transfer, or mutually exclusive current-state language.
Do NOT infer contradiction from recency alone, episode dates, meeting dates, weekly snapshots,
section headings such as "updates" or "status changes", or a new event involving the same entities.
Do NOT mark a role/ownership/responsibility/assignment fact as contradicted merely because the
NEW FACT describes work performed, updates shipped, documents created, progress made, blockers,
next steps, or collaboration involving the same person and project.
A restated assertion is contradicted only when it has an explicit incompatible change.
Return all contradicted idx values in contradicted_facts. If no contradictions, return an empty list.
Do not return duplicate decisions.
"""


def resolve_duplicates(context: dict[str, Any]) -> list[Message]:
    return [
        Message(role='system', content=DUPLICATES_SYSTEM_PROMPT),
        Message(
            role='user',
            content=render_template(
                DUPLICATES_USER_TEMPLATE,
                existing_edges=context['existing_edges'],
                new_edge=context['new_edge'],
            ),
        ),
    ]


def resolve_contradictions(context: dict[str, Any]) -> list[Message]:
    return [
        Message(role='system', content=CONTRADICTIONS_SYSTEM_PROMPT),
        Message(
            role='user',
            content=render_template(
                CONTRADICTIONS_USER_TEMPLATE,
                existing_edges=context['existing_edges'],
                edge_invalidation_candidates=context['edge_invalidation_candidates'],
                new_edge=context['new_edge'],
            ),
        ),
    ]


resolve_duplicates.__prompt_template__ = PromptTemplate(
    (DUPLICATES_SYSTEM_PROMPT, DUPLICATES_USER_TEMPLATE)
)
resolve_contradictions.__prompt_template__ = PromptTemplate(
    (CONTRADICTIONS_SYSTEM_PROMPT, CONTRADICTIONS_USER_TEMPLATE)
)

versions: Versions = {
    'resolve_edge': resolve_edge,
    'resolve_duplicates': resolve_duplicates,
    'resolve_contradictions': resolve_contradictions,
}


class DedupeEdgesPrompts(ABC):
    @abstractmethod
    def resolve_edge(self, context: dict[str, Any]) -> ChatPrompt: ...

    def resolve_duplicates(self, context: dict[str, Any]) -> ChatPrompt:
        return ChatPrompt.from_messages(resolve_duplicates(context))

    def resolve_contradictions(self, context: dict[str, Any]) -> ChatPrompt:
        return ChatPrompt.from_messages(resolve_contradictions(context))


class DefaultDedupeEdgesPrompts(DedupeEdgesPrompts):
    def resolve_edge(self, context: dict[str, Any]) -> ChatPrompt:
        return ChatPrompt.from_messages(resolve_edge(context))
