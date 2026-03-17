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


class Edge(BaseModel):
    relation_type: str = Field(..., description='FACT_PREDICATE_IN_SCREAMING_SNAKE_CASE')
    source_entity_id: int = Field(
        ..., description='The id of the source entity from the ENTITIES list'
    )
    target_entity_id: int = Field(
        ..., description='The id of the target entity from the ENTITIES list'
    )
    fact: str = Field(
        ...,
        description='A natural language description of the relationship between the entities, paraphrased from the source text',
    )
    reasoning: str = Field(
        ...,
        description='Brief reasoning (1-2 sentences) explaining why this relationship was extracted '
        'and why the chosen relation_type accurately represents the relationship described in the text.',
    )
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


class FactCorrection(BaseModel):
    """Correction for a misextracted fact."""
    original_fact: str = Field(..., description='The original fact text that needs correction')
    issue: str = Field(
        ...,
        description='Type of issue: "wrong_relation_type", "wrong_direction", "nonexistent_relationship"'
    )
    corrected_relation_type: str | None = Field(
        None,
        description='The correct relation_type if issue is "wrong_relation_type"'
    )
    reason: str = Field(..., description='Brief explanation of why this fact is incorrect')


class MissingFacts(BaseModel):
    missing_facts: list[str] = Field(
        default_factory=list,
        description="Facts that weren't extracted but should be"
    )
    facts_to_remove: list[str] = Field(
        default_factory=list,
        description='Facts that should be REMOVED because they are incorrect '
        '(e.g., relationship does not exist, hallucinated, or misinterpreted)',
    )
    facts_to_correct: list[FactCorrection] = Field(
        default_factory=list,
        description='Facts with wrong relation_type or direction that need correction',
    )


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
<FACT TYPES>
{context['edge_types']}
</FACT TYPES>

<PREVIOUS_MESSAGES>
{to_prompt_json([ep for ep in context['previous_episodes']])}
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
{context['episode_content']}
</CURRENT_MESSAGE>

<ENTITIES>
{to_prompt_json(context['nodes'])}
</ENTITIES>

<REFERENCE_TIME>
{context['reference_time']}  # ISO 8601 (UTC); used to resolve relative time mentions
</REFERENCE_TIME>

# TASK
Extract all factual relationships between the given ENTITIES based on the CURRENT MESSAGE.
Only extract facts that:
- involve two DISTINCT ENTITIES from the ENTITIES list,
- are clearly stated or unambiguously implied in the CURRENT MESSAGE,
    and can be represented as edges in a knowledge graph.
- Facts should include entity names rather than pronouns whenever possible.
- The FACT TYPES provide a list of the most important types of facts, make sure to extract facts of these types
- The FACT TYPES are not an exhaustive list, extract all facts from the message even if they do not fit into one
    of the FACT TYPES
- The FACT TYPES each contain a fact_type_signature list showing ALL valid (source, target) entity type pairs for that relation.

You may use information from the PREVIOUS MESSAGES only to disambiguate references or support continuity.


{context['custom_prompt']}

# EXTRACTION RULES

1. **Entity ID Validation**: `source_entity_id` and `target_entity_id` must use only the `id` values from the ENTITIES list provided above.
   - **CRITICAL**: Using IDs not in the list will cause the edge to be rejected
2. Each fact must involve two **distinct** entities.
3. Use a SCREAMING_SNAKE_CASE string as the `relation_type` (e.g., FOUNDED, WORKS_AT).
4. Do not emit duplicate or semantically redundant facts.
5. The `fact` should closely paraphrase the original source sentence(s). Do not verbatim quote the original text.
6. Use `REFERENCE_TIME` to resolve vague or relative temporal expressions (e.g., "last week").
7. Do **not** hallucinate or infer temporal bounds from unrelated events.
8. **Reasoning Requirement**: For each extracted edge, provide a brief `reasoning` (1-2 sentences) explaining:
   - Why this relationship was extracted from the text
   - Why the chosen relation_type accurately represents the relationship
   - This forces careful consideration before extraction.

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
    sys_prompt = """You are an AI assistant that reviews fact/relationship extraction quality. You perform both:
1. **Positive reflexion**: Identify facts that SHOULD have been extracted but weren't
2. **Negative reflexion**: Identify facts that are INCORRECT and should be removed or corrected

A fact should be REMOVED if:
- The relationship does not actually exist between the entities
- The fact was hallucinated or misinterpreted from the text
- The fact is redundant with another extracted fact

A fact should be CORRECTED if:
- The relation_type is wrong (e.g., WORKS_AT instead of MANAGES)
- The direction is reversed (source and target swapped)"""

    user_prompt = f"""
<PREVIOUS MESSAGES>
{to_prompt_json([ep for ep in context['previous_episodes']])}
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

Review the fact extraction quality and provide:

1. **missing_facts**: Facts that should have been extracted but weren't
   - Describe each missing fact in natural language

2. **facts_to_remove**: Facts from EXTRACTED FACTS that should be REMOVED
   - List the exact fact text that should be removed
   - Only include facts that are clearly incorrect or hallucinated

3. **facts_to_correct**: Facts with wrong relation_type or direction
   - original_fact: the exact fact text
   - issue: "wrong_relation_type" or "wrong_direction" or "nonexistent_relationship"
   - corrected_relation_type: the correct type (only for wrong_relation_type)
   - reason: why this correction is needed

Return empty lists if the extraction quality is good.
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
        {to_prompt_json(context['episode_content'])}
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
