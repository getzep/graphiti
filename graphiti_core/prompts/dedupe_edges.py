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
    duplicate_facts: list[str] = Field(
        ...,
        description='List of E-prefixed ids of duplicate facts, e.g. ["E0", "E3"]. '
        'Only from EXISTING FACTS. Empty list if none.',
    )
    contradicted_facts: list[str] = Field(
        ...,
        description='List of E- or I-prefixed ids of contradicted facts, e.g. ["E1", "I2"]. '
        'Empty list if none.',
    )


class Prompt(Protocol):
    resolve_edge: PromptVersion


class Versions(TypedDict):
    resolve_edge: PromptFunction


def resolve_edge(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a fact deduplication assistant. '
            'NEVER mark facts with key differences as duplicates.',
        ),
        Message(
            role='user',
            content=f"""
NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers.

IMPORTANT constraints:
- duplicate_facts: ONLY E-prefixed ids from EXISTING FACTS (NEVER include FACT INVALIDATION CANDIDATES)
- contradicted_facts: E- or I-prefixed ids from EITHER list (EXISTING FACTS or FACT INVALIDATION CANDIDATES)
- E and I each number from 0 independently (e.g. E0, E1, ... and I0, I1, ...)

<EXISTING FACTS>
{context['existing_edges']}
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
{context['edge_invalidation_candidates']}
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
{context['new_edge']}
</NEW FACT>

You will receive TWO lists of facts with independent per-list numbering.
EXISTING FACTS are numbered E0, E1, ... and FACT INVALIDATION CANDIDATES are numbered I0, I1, ...

1. DUPLICATE DETECTION:
   - If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those E-prefixed ids in duplicate_facts.
   - If no duplicates, return an empty list for duplicate_facts.

2. CONTRADICTION DETECTION:
   - Determine which facts the NEW FACT contradicts from either list.
   - A fact from EXISTING FACTS can be both a duplicate AND contradicted (e.g., semantically the same but the new fact updates/supersedes it).
   - Return all contradicted E- or I-prefixed ids in contradicted_facts.
   - If no contradictions, return an empty list for contradicted_facts.

<EXAMPLE>
EXISTING FACT: id=E0, "Alice joined Acme Corp in 2020"
NEW FACT: "Alice joined Acme Corp in 2020"
Result: duplicate_facts=["E0"], contradicted_facts=[] (identical factual information)

EXISTING FACT: id=E1, "Alice works at Acme Corp as a software engineer"
NEW FACT: "Alice works at Acme Corp as a senior engineer"
Result: duplicate_facts=[], contradicted_facts=["E1"] (same relationship but updated title — contradiction, NOT a duplicate)

EXISTING FACT: id=E2, "Bob ran 5 miles on Tuesday"
FACT INVALIDATION CANDIDATE: id=I0, "Bob runs 5 miles weekly"
NEW FACT: "Bob ran 3 miles on Wednesday"
Result: duplicate_facts=[], contradicted_facts=[] (different events on different days — neither duplicate nor contradiction)

EXISTING FACT: id=E3, "Alice is employed at Acme Corp"
FACT INVALIDATION CANDIDATE: id=I1, "Alice resigned from Acme Corp in 2024"
NEW FACT: "Alice returned to Acme Corp in 2026"
Result: duplicate_facts=[], contradicted_facts=["I1"] (outdated invalidation candidate is contradicted by the new fact)
</EXAMPLE>
""",
        ),
    ]


versions: Versions = {'resolve_edge': resolve_edge}
