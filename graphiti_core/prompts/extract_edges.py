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
    episode_indices: list[int] = Field(
        default_factory=lambda: [0],
        description='List of episode numbers (0-indexed) that this fact was derived from. '
        'When processing a single episode, this should be [0].',
    )


class ExtractedEdges(BaseModel):
    edges: list[Edge] = Field(default_factory=list)


class EdgeTimestamps(BaseModel):
    """Temporal bounds extracted from a fact."""

    valid_at: str | None = Field(
        None,
        description='When the fact became true. ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z)',
    )
    invalid_at: str | None = Field(
        None,
        description='When the fact stopped being true. ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z)',
    )


class BatchEdgeTimestamps(BaseModel):
    """Temporal bounds for a batch of facts."""

    timestamps: list[EdgeTimestamps] = Field(
        ..., description='Timestamps for each fact, in the same order as the input facts'
    )


class Prompt(Protocol):
    edge: PromptVersion
    extract_attributes: PromptVersion
    extract_timestamps: PromptVersion
    extract_timestamps_batch: PromptVersion


class Versions(TypedDict):
    edge: PromptFunction
    extract_attributes: PromptFunction
    extract_timestamps: PromptFunction
    extract_timestamps_batch: PromptFunction


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
            '1. Extracted fact triples should also be extracted with relevant date information. '
            '2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. '
            "Use each episode's timestamp to resolve temporal references within that episode. "
            'REFERENCE_TIME is a fallback for when no per-episode timestamp is available.',
        ),
        Message(
            role='user',
            content=f"""
<PREVIOUS_MESSAGES>
{to_prompt_json(context['previous_episodes'])}
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
2. Each fact must involve two **distinct** entities — `source_entity_name` and `target_entity_name` NEVER refer to the same entity.
3. Prefer facts that involve two distinct entities from the ENTITIES list. When a sentence describes a specific, concrete detail about a single entity (a brand name, a specific item, a physical description, a quantity, a location, a named activity), do NOT drop it. Instead, look for a second entity in the ENTITIES list that the detail relates to and form a proper triple (e.g., Entity -> OWNS -> item-entity, Entity -> LIVES_IN -> place-entity, Entity -> HAS_ATTRIBUTE -> detail-entity). Only skip the fact when no second entity in the ENTITIES list can anchor the detail.
   - BAD: "Alice feels happy" (vague single-entity state with no concrete detail — what is Alice happy about?)
   - GOOD: "Alice feels happy about Bob's promotion" → Alice -> FEELS_HAPPY_ABOUT -> Bob's promotion
   - GOOD: "Nate plays games on a Gamecube" → Nate -> PLAYS_GAMES_ON -> Gamecube (when "Gamecube" is in ENTITIES)
   - GOOD: "Alice congratulated Bob" (relationship between two entities), "Alice lives in Paris" (relationship between entity and place)
4. Do not emit semantically redundant facts, even across episodes within the CURRENT_MESSAGE. However, if a later episode adds specific details to a previously stated fact (e.g., adding a brand name, a count, a color, a location, or any concrete attribute), extract the more detailed version as a NEW fact — it is NOT a duplicate. Only treat facts as duplicates when they convey the same specificity.
   - NOT a duplicate: "user plays video games" (Episode 0) vs. "user plays games on a Gamecube" (Episode 1) → extract the second, more detailed fact.
   - IS a duplicate: "user plays games on a Gamecube" (Episode 0) vs. "user plays Gamecube games" (Episode 1) → extract once, list both episodes in `episode_indices`.
5. The `fact` MUST preserve all specific details from the source text: proper nouns, brand names, product names, model numbers, quantities, counts, colors, materials, physical descriptions, specific items, named locations, and named activities. Paraphrase the sentence structure but NEVER generalize:
   - NEVER generalize "Gamecube" to "gaming console", "Ford Mustang" to "car", "wool coat" to "coat", "red and purple lighting" to "lighting", "cracked windshield" to "car damage", or "three screenplays" to "several screenplays".
   - Do not verbatim quote the original text, but every concrete noun, number, and descriptor in the source should survive into the `fact`.
6. Use `REFERENCE_TIME` to resolve vague or relative temporal expressions (e.g., "last week"). When the CURRENT_MESSAGE contains multiple episodes with per-episode timestamps, prefer the timestamp of the specific episode the fact originates from.
7. Do **not** hallucinate or infer temporal bounds from unrelated events.

# RELATION TYPE RULES

- If FACT_TYPES are provided and the relationship matches one of the types (considering the entity type signature), use that fact_type_name as the `relation_type`.
- Otherwise, derive a `relation_type` from the relationship predicate in SCREAMING_SNAKE_CASE (e.g., WORKS_AT, LIVES_IN, IS_FRIENDS_WITH).

# DATETIME RULES

- Use ISO 8601 with "Z" suffix (UTC) (e.g., 2025-04-30T00:00:00Z).
- If the fact is ongoing (present tense), set `valid_at` to the timestamp of the episode the fact originates from. If no per-episode timestamp is available, use REFERENCE_TIME.
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
            content='You are a fact attribute extraction specialist. NEVER hallucinate or infer values not explicitly stated.',
        ),
        Message(
            role='user',
            content=f"""
Given the following FACT, its REFERENCE TIME, and any EXISTING ATTRIBUTES, extract or update
attributes based on the information explicitly stated in the fact. Use the provided attribute
descriptions to understand how each attribute should be determined.

Guidelines:
1. NEVER hallucinate or infer attribute values — only use values explicitly stated in the FACT.
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


def extract_timestamps(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You extract temporal bounds from facts. NEVER hallucinate dates.',
        ),
        Message(
            role='user',
            content=f"""Given a FACT and its REFERENCE TIME, determine when the fact became true
(valid_at) and when it stopped being true (invalid_at).

Rules:
- Resolve relative expressions ("last week", "2 years ago", "yesterday") using REFERENCE TIME.
- If the fact is ongoing (present tense), set valid_at to REFERENCE TIME.
- If a change or end is expressed, set invalid_at to the relevant time.
- Leave both null if no time is stated or resolvable.
- If only a date is mentioned (no time), assume 00:00:00.
- Use ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z).
- Do NOT hallucinate or infer dates from unrelated events.

<FACT>
{context['fact']}
</FACT>

<REFERENCE TIME>
{context['reference_time']}
</REFERENCE TIME>
""",
        ),
    ]


def extract_timestamps_batch(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You extract temporal bounds from facts. NEVER hallucinate dates.',
        ),
        Message(
            role='user',
            content=f"""Given a list of FACTS with their REFERENCE TIMES, determine when each fact
became true (valid_at) and when it stopped being true (invalid_at).

Rules:
- Resolve relative expressions ("last week", "2 years ago", "yesterday") using each fact's REFERENCE TIME.
- If the fact is ongoing (present tense), set valid_at to its REFERENCE TIME.
- If a change or end is expressed, set invalid_at to the relevant time.
- Leave both null if no time is stated or resolvable.
- If only a date is mentioned (no time), assume 00:00:00.
- Use ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z).
- Do NOT hallucinate or infer dates from unrelated events.

Return one timestamps entry per fact, in the same order.

<FACTS>
{to_prompt_json(context['facts'])}
</FACTS>
""",
        ),
    ]


versions: Versions = {
    'edge': edge,
    'extract_attributes': extract_attributes,
    'extract_timestamps': extract_timestamps,
    'extract_timestamps_batch': extract_timestamps_batch,
}
