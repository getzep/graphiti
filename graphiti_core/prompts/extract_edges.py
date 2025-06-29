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


class Edge(BaseModel):
    relation_type: str = Field(..., description='FACT_PREDICATE_IN_SCREAMING_SNAKE_CASE')
    source_entity_id: int = Field(..., description='The id of the source entity of the fact.')
    target_entity_id: int = Field(..., description='The id of the target entity of the fact.')
    fact: str = Field(..., description='')
    valid_at: str | None = Field(
        None,
        description='The date and time when the relationship described by the edge fact became true or was established. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ)',
    )
    invalid_at: str | None = Field(
        None,
        description='The date and time when the relationship described by the edge fact stopped being true or ended. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ)',
    )


class ExtractedEdges(BaseModel):
    edges: list[Edge]


class MissingFacts(BaseModel):
    missing_facts: list[str] = Field(..., description="facts that weren't extracted")


class Prompt(Protocol):
    edge: PromptVersion
    reflexion: PromptVersion
    extract_attributes: PromptVersion


class Versions(TypedDict):
    edge: PromptFunction
    reflexion: PromptFunction
    extract_attributes: PromptFunction


def edge(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are an expert fact extractor that extracts fact triples from text. '
            '1. Extracted fact triples should also be extracted with relevant date information.'
            '2. Treat the CURRENT TIME as the time the CURRENT MESSAGE was sent. All temporal information should be extracted relative to this time.',
        ),
        Message(
            role='user',
            content=f"""
<PREVIOUS_MESSAGES>
{json.dumps([ep for ep in context['previous_episodes']], indent=2)}
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
{context['episode_content']}
</CURRENT_MESSAGE>

<ENTITIES>
{context['nodes']} 
</ENTITIES>

<REFERENCE_TIME>
{context['reference_time']}  # ISO 8601 (UTC); used to resolve relative time mentions
</REFERENCE_TIME>

<FACT TYPES>
{context['edge_types']}
</FACT TYPES>

# TASK
Extract all factual relationships between the given ENTITIES based on the CURRENT MESSAGE.
Only extract facts that:
- involve two DISTINCT ENTITIES from the ENTITIES list,
- are clearly stated or unambiguously implied in the CURRENT MESSAGE,
    and can be represented as edges in a knowledge graph.
- The FACT TYPES provide a list of the most important types of facts, make sure to extract facts of these types
- The FACT TYPES are not an exhaustive list, extract all facts from the message even if they do not fit into one
    of the FACT TYPES
- The FACT TYPES each contain their fact_type_signature which represents the source and target entity types.

You may use information from the PREVIOUS MESSAGES only to disambiguate references or support continuity.


{context['custom_prompt']}

# EXTRACTION RULES

1. Only emit facts where both the subject and object match IDs in ENTITIES.
2. Each fact must involve two **distinct** entities.
3. Use a SCREAMING_SNAKE_CASE string as the `relation_type` (e.g., FOUNDED, WORKS_AT).
4. Do not emit duplicate or semantically redundant facts.
5. The `fact_text` should quote or closely paraphrase the original source sentence(s).
6. Use `REFERENCE_TIME` to resolve vague or relative temporal expressions (e.g., "last week").
7. Do **not** hallucinate or infer temporal bounds from unrelated events.

# DATETIME RULES

- Use ISO 8601 with “Z” suffix (UTC) (e.g., 2025-04-30T00:00:00Z).
- If the fact is ongoing (present tense), set `valid_at` to REFERENCE_TIME.
- If a change/termination is expressed, set `invalid_at` to the relevant timestamp.
- Leave both fields `null` if no explicit or resolvable time is stated.
- If only a date is mentioned (no time), assume 00:00:00.
- If only a year is mentioned, use January 1st at 00:00:00.
        """,
        ),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which facts have not been extracted from the given context"""

    user_prompt = f"""
<PREVIOUS MESSAGES>
{json.dumps([ep for ep in context['previous_episodes']], indent=2)}
</PREVIOUS MESSAGES>
<CURRENT MESSAGE>
{context['episode_content']}
</CURRENT MESSAGE>

<EXTRACTED ENTITIES>
{context['nodes']}
</EXTRACTED ENTITIES>

<EXTRACTED FACTS>
{context['extracted_facts']}
</EXTRACTED FACTS>

Given the above MESSAGES, list of EXTRACTED ENTITIES entities, and list of EXTRACTED FACTS; 
determine if any facts haven't been extracted.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts fact properties from the provided text.',
        ),
        Message(
            role='user',
            content=f"""

        <MESSAGE>
        {json.dumps(context['episode_content'], indent=2)}
        </MESSAGE>
        <REFERENCE TIME>
        {context['reference_time']}
        </REFERENCE TIME>

        Given the above MESSAGE, its REFERENCE TIME, and the following FACT, update any of its attributes based on the information provided
        in MESSAGE. Use the provided attribute descriptions to better understand how each attribute should be determined.

        Guidelines:
        1. Do not hallucinate entity property values if they cannot be found in the current context.
        2. Only use the provided MESSAGES and FACT to set attribute values.

        <FACT>
        {context['fact']}
        </FACT>
        """,
        ),
    ]


versions: Versions = {
    'edge': edge,
    'reflexion': reflexion,
    'extract_attributes': extract_attributes,
}
