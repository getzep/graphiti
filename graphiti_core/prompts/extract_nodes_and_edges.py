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


class CombinedEntity(BaseModel):
    """Entity extracted by the combined node+edge extraction prompt."""

    name: str = Field(..., description='Name of the extracted entity')
    entity_type_id: int = Field(
        description='ID of the classified entity type. '
        'Must be one of the provided entity_type_id integers.',
    )


class CombinedFact(BaseModel):
    """Relationship fact extracted by the combined node+edge extraction prompt."""

    source_entity_name: str = Field(
        ..., description='The name of the source entity from the extracted entities list'
    )
    target_entity_name: str = Field(
        ..., description='The name of the target entity from the extracted entities list'
    )
    relation_type: str = Field(
        ...,
        description='The type of relationship between the entities, in SCREAMING_SNAKE_CASE '
        '(e.g., WORKS_AT, LIVES_IN, IS_FRIENDS_WITH)',
    )
    fact: str = Field(
        ...,
        description='A self-contained natural language description of the relationship, '
        'paraphrased from the source text with all specific details preserved',
    )
    episode_indices: list[int] = Field(
        default_factory=lambda: [0],
        description='List of episode numbers (0-indexed) that this fact was derived from. '
        'When processing a single episode, this should be [0].',
    )


class CombinedExtraction(BaseModel):
    """Combined node and edge extraction response."""

    extracted_entities: list[CombinedEntity] = Field(..., description='List of extracted entities')
    edges: list[CombinedFact] = Field(..., description='List of extracted relationship facts')


class Prompt(Protocol):
    extract_message: PromptVersion


class Versions(TypedDict):
    extract_message: PromptFunction


def extract_message(context: dict[str, Any]) -> list[Message]:
    sys_prompt = (
        'You are an expert knowledge graph extraction specialist for an AI agent memory system. '
        'You extract both entity nodes and relationship facts from conversations in a single pass. '
        'The extracted graph will be searched later by an AI agent to answer questions, personalize '
        'responses, and maintain long-term memory. The original conversation will NOT be available '
        'at retrieval time — only the entities and facts you extract will survive.'
    )

    user_prompt = f"""
ENTITY RULES:
1. Extract speakers and named entities explicitly mentioned in CURRENT MESSAGES.
2. Entity names must be at most 5 words. Use the most specific form mentioned.
3. When someone discusses their possession, project, pet, or creation, extract it
   as a SEPARATE possessive entity — not just the person, not just the bare noun:
   GOOD: "James's notebook", "Calvin's guitar", "Audrey's dogs", "Sam's cooking class"
   BAD: "notebook", "guitar", "dogs" (too generic) or just "James" (collapses detail)
4. Extract hobbies and activities as entities when someone engages in them:
   "video games", "watercolor painting", "VR gaming", "road cycling", "cooking"
5. Extract named/described objects ("Gamecube", "Ford Mustang", "wool coat") and
   places ("Riverside Park", "the gym", "the beach") — not bare generics ("car", "coat").
6. Do NOT extract: pronouns, vague abstractions (balance, growth, motivation),
   filler nouns (day, life, stuff, time), dates as entities, full sentences as names.
7. Each entity appears exactly ONCE. Classify using the ENTITY TYPES provided.
8. Only extract entities from CURRENT MESSAGES — PREVIOUS MESSAGES are context only.

FACT RULES:
1. source_entity_name and target_entity_name must match your extracted entity names.
2. When a fact involves two entities that are BOTH in your extracted entities list,
   you MUST use both as source and target — never collapse into a self-referencing fact:
   "Nate plays games on a Gamecube" → Nate -> PLAYS_GAMES_ON -> Gamecube
   "Sarah lives in San Francisco" → Sarah -> LIVES_IN -> San Francisco
   "James has a dog named Maximilian" → James -> HAS_PET -> Maximilian
   Only use a self-referencing fact when no second entity in your list fits.
   Self-referencing facts are still common and valuable — do NOT skip them:
   - Routines/health: "Deborah goes jogging every morning", "Evan has a knee injury"
   - Preferences/plans: "Nate's favorite game is Xenoblade Chronicles",
     "Jon said he would not quit on his dreams"
   - Emotions/states: "Sam feels he lacks motivation"
3. Facts must be SELF-CONTAINED — understandable without the original episode.
   Use entity names, not pronouns. Preserve specific details where possible.
4. Extract facts from EVERY episode — not just the first. Process each episode's
   CURRENT_MESSAGE independently. Set `episode_indices` to the 0-based episode
   number(s) each fact comes from (matching [Episode N] headers).
   If the SAME fact appears across multiple episodes, extract it ONCE and list ALL
   episode indices — do NOT emit duplicate facts with different episode numbers.
5. You MAY use PREVIOUS MESSAGES to resolve what the current message refers to.
   If the current message reacts to or confirms prior context, extract the full
   contextualized fact (e.g., "all the hard work paid off" → extract what paid off).
6. Extract liberally — when in doubt, extract the fact. Preferences, opinions,
   reactions, advice, plans, states, and experiences are all valuable. Only skip
   content-free utterances like "Hi!", "Bye!", "Thanks!".
7. Do not emit redundant facts across episodes. But if a later episode adds new
   details (brand, count, location), extract the more detailed version as a new fact.

<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

<PREVIOUS MESSAGES>
{to_prompt_json([ep for ep in context['previous_episodes']])}
</PREVIOUS MESSAGES>

<CURRENT MESSAGES>
{context['episode_content']}
</CURRENT MESSAGES>

{context['custom_extraction_instructions']}
"""

    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


versions: Versions = {
    'extract_message': extract_message,
}
