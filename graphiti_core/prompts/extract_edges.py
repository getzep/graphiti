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
    source_entity_name: str = Field(
        ..., description='The name of the source entity from the ENTITIES list'
    )
    target_entity_name: str = Field(
        ..., description='The name of the target entity from the ENTITIES list'
    )
    relation_type: str = Field(
        ...,
        description='The type of relationship between the entities, in SCREAMING_SNAKE_CASE (e.g., WORKS_AT, LIVES_IN, IS_FRIENDS_WITH)',
    )
    fact: str = Field(
        ...,
        description='A natural language description of the relationship between the entities, paraphrased from the source text',
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


class Prompt(Protocol):
    edge: PromptVersion
    extract_attributes: PromptVersion


class Versions(TypedDict):
    edge: PromptFunction
    extract_attributes: PromptFunction


def edge(context: dict[str, Any]) -> list[Message]:
    edge_types_section = ''
    if context.get('edge_types'):
        edge_types_section = f"""
<FACT_TYPES>
{to_prompt_json(context['edge_types'])}
</FACT_TYPES>
"""

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
{edge_types_section}
# TASK
Extract all factual relationships between the given ENTITIES based on the CURRENT MESSAGE.
Only extract facts that:
- involve two DISTINCT ENTITIES from the ENTITIES list,
- are clearly stated or unambiguously implied in the CURRENT MESSAGE,
    and can be represented as edges in a knowledge graph.
- Facts should include entity names rather than pronouns whenever possible.

You may use information from the PREVIOUS MESSAGES only to disambiguate references or support continuity.


{context['custom_extraction_instructions']}

# EXTRACTION RULES

1. **Entity Name Validation**: `source_entity_name` and `target_entity_name` must use only the `name` values from the ENTITIES list provided above.
   - **CRITICAL**: Using names not in the list will cause the edge to be rejected
2. Each fact must involve two **distinct** entities.
3. Do not emit duplicate or semantically redundant facts.
4. The `fact` should closely paraphrase the original source sentence(s). Do not verbatim quote the original text.
5. Use `REFERENCE_TIME` to resolve vague or relative temporal expressions (e.g., "last week").
6. Do **not** hallucinate or infer temporal bounds from unrelated events.

# RELATION TYPE RULES

- If FACT_TYPES are provided and the relationship matches one of the types (considering the entity type signature), use that fact_type_name as the `relation_type`.
- Otherwise, derive a `relation_type` from the relationship predicate in SCREAMING_SNAKE_CASE (e.g., WORKS_AT, LIVES_IN, IS_FRIENDS_WITH).

# DATETIME RULES

- Use ISO 8601 with "Z" suffix (UTC) (e.g., 2025-04-30T00:00:00Z).
- If the fact is ongoing (present tense), set `valid_at` to REFERENCE_TIME.
- If a change/termination is expressed, set `invalid_at` to the relevant timestamp.
- Leave both fields `null` if no explicit or resolvable time is stated.
- If only a date is mentioned (no time), assume 00:00:00.
- If only a year is mentioned, use January 1st at 00:00:00.
        """,
        ),
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
        Given the following FACT, its REFERENCE TIME, and any EXISTING ATTRIBUTES, extract or update
        attributes based on the information explicitly stated in the fact. Use the provided attribute
        descriptions to understand how each attribute should be determined.

        Guidelines:
        1. Do not hallucinate attribute values if they cannot be found explicitly in the fact.
        2. Only use information stated in the FACT to set attribute values.
        3. Use REFERENCE TIME to resolve any relative temporal expressions in the fact.
        4. Preserve existing attribute values unless the fact explicitly provides new information.

        <FACT>
        {context['fact']}
        </FACT>

        <REFERENCE TIME>
        {context['reference_time']}
        </REFERENCE TIME>

        <EXISTING ATTRIBUTES>
        {to_prompt_json(context['existing_attributes'])}
        </EXISTING ATTRIBUTES>
        """,
        ),
    ]


versions: Versions = {
    'edge': edge,
    'extract_attributes': extract_attributes,
}
