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

from .knowledge import KNOWLEDGE_SECTION_TEMPLATE, knowledge_section
from .models import ChatPrompt, Message, PromptFunction, PromptVersion
from .prompt_helpers import to_prompt_json
from .versions import PromptTemplate, render_template

EDGE_SYSTEM_TEMPLATE = """You are an expert fact extractor that extracts fact triples from text. 1. Extracted fact triples should also be extracted with relevant date information. 2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. Use each episode's timestamp to resolve temporal references within that episode. REFERENCE_TIME is a fallback for when no per-episode timestamp is available."""
EDGE_USER_PREFIX_TEMPLATE = """
<PREVIOUS_MESSAGES>
{previous_episodes}
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
{episode_content}
</CURRENT_MESSAGE>

<ENTITIES>
{nodes}
</ENTITIES>

<REFERENCE_TIME>
{reference_time}  # ISO 8601 (UTC); used to resolve relative time mentions
</REFERENCE_TIME>
"""
EDGE_TYPES_SECTION_TEMPLATE = """
<FACT_TYPES>
{edge_types}
</FACT_TYPES>

# RELATION TYPE RULES

{relation_type_rules}
"""
EDGE_TYPES_STRICT_RULES = """- Use one of the provided fact_type_name values as the relation_type.
- If a relationship does not match any FACT_TYPE, skip it. Do not derive a new relation_type."""
EDGE_TYPES_FLEXIBLE_RULES = """- If FACT_TYPES are provided and the relationship matches one of the types (considering the entity type signature), use that fact_type_name as the `relation_type`.
- Otherwise, derive a `relation_type` from the relationship predicate in SCREAMING_SNAKE_CASE (e.g., WORKS_AT, LIVES_IN, IS_FRIENDS_WITH)."""
EDGE_USER_SUFFIX_TEMPLATE = """
# TASK
Extract all factual relationships between the given ENTITIES based on the CURRENT MESSAGE.
Only extract facts that:
- involve two DISTINCT ENTITIES from the ENTITIES list,
- are clearly stated or unambiguously implied in the CURRENT MESSAGE,
    and can be represented as edges in a knowledge graph.
- Facts should include entity names rather than pronouns whenever possible.

You may use information from the PREVIOUS MESSAGES only to disambiguate references or support continuity.


{knowledge}

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

# DATETIME RULES

- Use ISO 8601 with "Z" suffix (UTC) (e.g., 2025-04-30T00:00:00Z).
- If the fact is ongoing (present tense), set `valid_at` to the timestamp of the episode the fact originates from. If no per-episode timestamp is available, use REFERENCE_TIME.
- If the fact is a future-tense plan, promise, scheduled event, or deadline, set `valid_at` to the episode timestamp / REFERENCE_TIME. Do NOT set `valid_at` to the future occurrence date.
- Preserve future occurrence dates and deadlines in the `fact` text. Do NOT use a soft deadline ("by March 2026", "by the end of Q3 2025") as `invalid_at`.
- Use a future date as `invalid_at` only when the fact describes an ongoing state that ends then (for example, "is unavailable until December 25, 2024" or "works at Acme until January 15, 2027").
- If a change/termination is expressed, set `invalid_at` to the relevant timestamp.
- Leave both fields `null` if no explicit or resolvable time is stated.
- If only a date is mentioned (no time), assume 00:00:00.
- If only a year is mentioned, use January 1st at 00:00:00.
- Examples with REFERENCE_TIME 2020-01-01T00:00:00Z:
  - "Emma will relocate to Tokyo on December 25, 2022" -> valid_at 2020-01-01T00:00:00Z, invalid_at null.
  - "Alice will complete the project by July 30, 2024" -> valid_at 2020-01-01T00:00:00Z, invalid_at null.
  - "The team will ship version 2.0 by the end of Q3 2025" -> valid_at 2020-01-01T00:00:00Z, invalid_at null.
        """
EXTRACT_ATTRIBUTES_SYSTEM_TEMPLATE = """You are a fact attribute extraction specialist. You ONLY emit attribute values that are explicitly stated in the FACT or already present in EXISTING ATTRIBUTES. You output strictly the JSON specified by the response schema — no reasoning, no explanation, no commentary in any field."""
EXTRACT_ATTRIBUTES_USER_TEMPLATE = """\
Given the following FACT, its REFERENCE TIME, and any EXISTING ATTRIBUTES, update the attributes.

HARD RULES — violating any of these is a failure:

1. Each attribute value MUST be one of:
   (a) a clean value copied or directly normalized from the FACT,
   (b) the existing value already in EXISTING ATTRIBUTES (preserved unchanged), or
   (c) null / omitted, when neither (a) nor (b) applies.

2. NEVER write reasoning, justification, or commentary into any field. Specifically:
   - NEVER include parenthetical explanations like "(implied by ...)", "(Context: ...)",
     "(not explicitly stated ...)", "(based on ...)".
   - NEVER include first-person or deliberative phrases like "I should...", "However...",
     "Sticking to...", "Since no...", "the instruction is to...", "must be kept...".
   - NEVER list alternatives or candidates inside one field ("X, or Y, or maybe Z").
   - NEVER explain why a value is null. If unknown, set the field to null and stop.

3. Each attribute schema description tells you the FORMAT a real value should take. The
   description text is NEVER itself a value. NEVER copy schema description text into the field.

4. The literal strings "null", "N/A", "Not specified", "unknown", "none", "not provided",
   or any sentence describing absence are NOT valid values. If no value is supported by
   the FACT, set the field to null (or omit it) — do not write a sentence.

5. Each attribute value must be a short, well-formed instance of the type the field
   describes. If you cannot produce a clean value of that type from the FACT, the field is null.
   A field has a value only when the FACT states the specific data that the field description
   asks for. Extract only that data. Copying or paraphrasing the whole FACT is never a value:
   the FACT is already stored on the edge. A value that repeats most of the FACT, or the FACT without its subject, is also a copy.
   If the FACT does not state that data, set the field to null.
   Example: for the field "reason" ("Why the customer switched plans") and the FACT "Dana switched
   to the annual plan because it costs less per month", the value is "it costs less per month".
   For the FACT "Dana switched to the annual plan", the value is null.

6. Use REFERENCE TIME to resolve any relative temporal expressions in the fact.

7. Preserve existing attribute values unless the FACT explicitly provides a new value.

<FACT>
{fact}
</FACT>

<REFERENCE TIME>
{reference_time}
</REFERENCE TIME>

<EXISTING ATTRIBUTES>
{existing_attributes}
</EXISTING ATTRIBUTES>
"""
EXTRACT_TIMESTAMPS_SYSTEM_TEMPLATE = (
    'You extract temporal bounds from facts. NEVER hallucinate dates.'
)
EXTRACT_TIMESTAMPS_USER_TEMPLATE = """Given a FACT and its REFERENCE TIME, determine when the fact became true
(valid_at) and when it stopped being true (invalid_at).

Rules:
- Resolve relative expressions ("last week", "2 years ago", "yesterday") using REFERENCE TIME.
- If the fact is ongoing (present tense), set valid_at to REFERENCE TIME.
- If the fact is a future-tense plan, promise, scheduled event, or deadline, set valid_at to REFERENCE TIME. Do NOT set valid_at to the future occurrence date.
- Preserve future occurrence dates and deadlines in the fact text. Do NOT use a soft deadline ("by March 2026", "by the end of Q3 2025") as invalid_at.
- Use a future date as invalid_at only when the fact describes an ongoing state that ends then (for example, "is unavailable until December 25, 2024" or "works at Acme until January 15, 2027").
- If a change or end is expressed, set invalid_at to the relevant time.
- Leave both null if no time is stated or resolvable.
- If only a date is mentioned (no time), assume 00:00:00.
- Use ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z).
- Do NOT hallucinate or infer dates from unrelated events.
- Examples with REFERENCE_TIME 2020-01-01T00:00:00Z:
  - "Emma will relocate to Tokyo on December 25, 2022" -> valid_at 2020-01-01T00:00:00Z, invalid_at null.
  - "Alice will complete the project by July 30, 2024" -> valid_at 2020-01-01T00:00:00Z, invalid_at null.

<FACT>
{fact}
</FACT>

<REFERENCE TIME>
{reference_time}
</REFERENCE TIME>
"""
EXTRACT_TIMESTAMPS_BATCH_SYSTEM_TEMPLATE = (
    'You extract temporal bounds from facts. NEVER hallucinate dates.'
)
EXTRACT_TIMESTAMPS_BATCH_USER_TEMPLATE = """Given a list of FACTS with their REFERENCE TIMES, determine when each fact
became true (valid_at) and when it stopped being true (invalid_at).

Rules:
- Resolve relative expressions ("last week", "2 years ago", "yesterday") using each fact's REFERENCE TIME.
- If the fact is ongoing (present tense), set valid_at to its REFERENCE TIME.
- If the fact is a future-tense plan, promise, scheduled event, or deadline, set valid_at to its REFERENCE TIME. Do NOT set valid_at to the future occurrence date.
- Preserve future occurrence dates and deadlines in the fact text. Do NOT use a soft deadline ("by March 2026", "by the end of Q3 2025") as invalid_at.
- Use a future date as invalid_at only when the fact describes an ongoing state that ends then (for example, "is unavailable until December 25, 2024" or "works at Acme until January 15, 2027").
- If a change or end is expressed, set invalid_at to the relevant time.
- Leave both null if no time is stated or resolvable.
- If only a date is mentioned (no time), assume 00:00:00.
- Use ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z).
- Do NOT hallucinate or infer dates from unrelated events.
- Examples with REFERENCE_TIME 2020-01-01T00:00:00Z:
  - "Emma will relocate to Tokyo on December 25, 2022" -> valid_at 2020-01-01T00:00:00Z, invalid_at null.
  - "Alice will complete the project by July 30, 2024" -> valid_at 2020-01-01T00:00:00Z, invalid_at null.
  - "The team will ship version 2.0 by the end of Q3 2025" -> valid_at 2020-01-01T00:00:00Z, invalid_at null.

Return one timestamps entry per fact, in the same order.

<FACTS>
{facts}
</FACTS>
"""
# One template for the untyped, flexible, and strict renders. A change to the
# strict rules also changes the version of untyped and flexible requests.
EDGE_PROMPT_TEMPLATE = PromptTemplate(
    (
        EDGE_SYSTEM_TEMPLATE,
        EDGE_USER_PREFIX_TEMPLATE,
        EDGE_TYPES_SECTION_TEMPLATE,
        EDGE_TYPES_STRICT_RULES,
        EDGE_TYPES_FLEXIBLE_RULES,
        EDGE_USER_SUFFIX_TEMPLATE,
        KNOWLEDGE_SECTION_TEMPLATE,
    )
)


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
        description='The date and time when the relationship or assertion described by the edge fact became true or was established. Future-tense plans/promises use the statement reference time, not the future occurrence date. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ)',
    )
    invalid_at: str | None = Field(
        None,
        description='The date and time when the relationship described by the edge fact stopped being true or ended. Soft deadlines are not invalid_at values. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ)',
    )
    episode_indices: list[int] = Field(
        default_factory=lambda: [0],
        description='List of episode numbers (0-indexed) that this fact was derived from. '
        'When processing a single episode, this should be [0].',
    )


class ExtractedEdges(BaseModel):
    edges: list[Edge]


class EdgeTimestamps(BaseModel):
    """Temporal bounds extracted from a fact."""

    valid_at: str | None = Field(
        None,
        description='When the fact became true. Future-tense plans/promises use the reference time, not the future occurrence date. ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z)',
    )
    invalid_at: str | None = Field(
        None,
        description='When the fact stopped being true. Soft deadlines are not invalid_at values. ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z)',
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
        relation_type_rules = (
            EDGE_TYPES_STRICT_RULES
            if context.get('strict_edge_types', False)
            else EDGE_TYPES_FLEXIBLE_RULES
        )
        edge_types_section = render_template(
            EDGE_TYPES_SECTION_TEMPLATE,
            edge_types=to_prompt_json(context['edge_types']),
            relation_type_rules=relation_type_rules,
        )

    return [
        Message(
            role='system',
            content=EDGE_SYSTEM_TEMPLATE,
        ),
        Message(
            role='user',
            content=(
                render_template(
                    EDGE_USER_PREFIX_TEMPLATE,
                    previous_episodes=to_prompt_json(context['previous_episodes']),
                    episode_content=context['episode_content'],
                    nodes=to_prompt_json(context['nodes']),
                    reference_time=context['reference_time'],
                )
                + edge_types_section
                + render_template(
                    EDGE_USER_SUFFIX_TEMPLATE,
                    knowledge=knowledge_section(context['knowledge']),
                )
            ),
        ),
    ]


edge.__prompt_template__ = EDGE_PROMPT_TEMPLATE


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content=EXTRACT_ATTRIBUTES_SYSTEM_TEMPLATE,
        ),
        Message(
            role='user',
            content=render_template(
                EXTRACT_ATTRIBUTES_USER_TEMPLATE,
                fact=context['fact'],
                reference_time=context['reference_time'],
                existing_attributes=to_prompt_json(context['existing_attributes']),
            ),
        ),
    ]


def extract_timestamps(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content=EXTRACT_TIMESTAMPS_SYSTEM_TEMPLATE,
        ),
        Message(
            role='user',
            content=render_template(
                EXTRACT_TIMESTAMPS_USER_TEMPLATE,
                fact=context['fact'],
                reference_time=context['reference_time'],
            ),
        ),
    ]


def extract_timestamps_batch(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content=EXTRACT_TIMESTAMPS_BATCH_SYSTEM_TEMPLATE,
        ),
        Message(
            role='user',
            content=render_template(
                EXTRACT_TIMESTAMPS_BATCH_USER_TEMPLATE,
                facts=to_prompt_json(context['facts']),
            ),
        ),
    ]


versions: Versions = {
    'edge': edge,
    'extract_attributes': extract_attributes,
    'extract_timestamps': extract_timestamps,
    'extract_timestamps_batch': extract_timestamps_batch,
}

extract_attributes.__prompt_template__ = PromptTemplate(
    (EXTRACT_ATTRIBUTES_SYSTEM_TEMPLATE, EXTRACT_ATTRIBUTES_USER_TEMPLATE)
)
extract_timestamps.__prompt_template__ = PromptTemplate(
    (EXTRACT_TIMESTAMPS_SYSTEM_TEMPLATE, EXTRACT_TIMESTAMPS_USER_TEMPLATE)
)
extract_timestamps_batch.__prompt_template__ = PromptTemplate(
    (EXTRACT_TIMESTAMPS_BATCH_SYSTEM_TEMPLATE, EXTRACT_TIMESTAMPS_BATCH_USER_TEMPLATE)
)


class ExtractEdgesPrompts(ABC):
    @abstractmethod
    def edge(self, context: dict[str, Any]) -> ChatPrompt: ...

    @abstractmethod
    def extract_attributes(self, context: dict[str, Any]) -> ChatPrompt: ...

    @abstractmethod
    def extract_timestamps(self, context: dict[str, Any]) -> ChatPrompt: ...

    @abstractmethod
    def extract_timestamps_batch(self, context: dict[str, Any]) -> ChatPrompt: ...


class DefaultExtractEdgesPrompts(ExtractEdgesPrompts):
    def edge(self, context: dict[str, Any]) -> ChatPrompt:
        return ChatPrompt.from_messages(edge(context))

    def extract_attributes(self, context: dict[str, Any]) -> ChatPrompt:
        return ChatPrompt.from_messages(extract_attributes(context))

    def extract_timestamps(self, context: dict[str, Any]) -> ChatPrompt:
        return ChatPrompt.from_messages(extract_timestamps(context))

    def extract_timestamps_batch(self, context: dict[str, Any]) -> ChatPrompt:
        return ChatPrompt.from_messages(extract_timestamps_batch(context))
