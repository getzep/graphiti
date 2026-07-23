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


def _build_edge_types_section(edge_types: list[dict] | None) -> str:
    """Build the optional FACT TYPES section for the combined prompt."""
    if not edge_types:
        return ''
    return f"""
<FACT TYPES>
{to_prompt_json(edge_types)}
</FACT TYPES>
RELATION TYPE RULES:
- When a relationship matches a FACT TYPE, use that fact_type_name as the relation_type.
- If no FACT TYPE fits, derive a relation_type in SCREAMING_SNAKE_CASE
  (e.g., WORKS_AT, LIVES_IN, IS_FRIENDS_WITH).
"""


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
6. Entity names must be NOUN PHRASES that refer to a discrete thing in the user's
   world. Do NOT extract any of the following — they belong inside fact text, as
   properties of another entity, or not at all:
   a. Pronouns, articles, and unresolved references: "the city", "this issue",
      "that rule", "the victim", "the game". Resolve to the actual referent or skip.
   b. Vague abstractions and filler nouns (balance, growth, motivation, day, life,
      stuff, time). Full sentences or propositions as names are also banned.
   c. Specific clock times and dates as entities ("5:54 am", "8:47 am",
      "January 15"). A time/date is a property of the event — keep it in fact text.
   d. Bare quantities, measurements, durations, prices, counts, or recipe amounts
      ("7-8 hours", "7.5 hours of daylight", "$150/hour", "10-pound dumbbells",
      "1 cup granulated white sugar", "4 cups water", "15-30 minutes"). Numbers
      are properties of the thing they measure — preserve them inside the fact.
   e. Geographic coordinates / latitudes / longitudes ("52 degrees north",
      "66°33' north of the equator"). They are attributes of the location entity.
   f. Multiple-choice answer labels, response-option tokens, or rubric/template
      scaffolding the assistant is constrained to emit ("Agree",
      "Strongly disagree", "the four answers", "the score", "scorr"). These are
      prompt-template tokens, not entities in the user's world.
   g. Imperative verb-phrase advice — bullet headers from how-to / tip lists
      ("Buy in bulk", "Cook in bulk", "Plan your meals", "Follow up on leads",
      "Personalize your outreach", "Shop sales", "Keep it simple"). Entity names
      are noun phrases. If the topic is worth keeping, name the topical noun
      ("bulk buying", "meal planning") and put the advice in the fact text.
   h. Quoted slogans, idioms, definitional phrases, or loaded adjective phrases
      from opinion statements ("from each according to his ability, to each
      according to his need", "the enemy of my enemy", "the perpetrator of a
      crime", "genuinely disadvantaged"). They are not retrievable referents.
7. Each entity appears exactly ONCE. Classify using the ENTITY TYPES provided.
8. Only extract entities from CURRENT MESSAGES — PREVIOUS MESSAGES are context only.
9. Skip didactic / tutorial scaffolding when the assistant is teaching a topic
   (Unix commands, astronomy, cooking technique, etc.). Tutorial example values
   ("/path/to/source", "file.txt", "<username>") and explanatory primitives are
   external didactic material, not facts about the user. The user's interest in
   the topic ("user is learning Unix commands") is one fact — do not turn each
   command, flag, or example path into its own node.

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
4. Extract facts from EVERY episode — not just the meatiest one. Conversational
   setup episodes still contain real facts that belong in the graph:
   - Narrative announcements: "something funny happened last night",
     "I have news to share", "you won't believe what happened"
   - Forward commitments: "I'll keep you posted", "I'll let you know how it
     goes", "I'll call you when I land"
   - Plans / intentions: "I'm calling them tomorrow", "I'm planning to start
     next week"
   - Emotional / state setup: "I've been feeling stressed lately",
     "I'm so excited about this"
   Process each episode's CURRENT_MESSAGE independently. Set `episode_indices`
   to the 0-based episode number(s) each fact comes from (matching
   [Episode N] headers). If the SAME fact appears across multiple episodes,
   extract it ONCE and list ALL episode indices — do NOT emit duplicate facts
   with different episode numbers.
5. You MAY use PREVIOUS MESSAGES to resolve what the current message refers to.
   If the current message reacts to or confirms prior context, extract the full
   contextualized fact (e.g., "all the hard work paid off" → extract what paid off).
6. Extract liberally — when in doubt, extract the fact. Preferences, opinions,
   reactions, advice, plans, states, and experiences are all valuable. Only skip
   content-free utterances like "Hi!", "Bye!", "Thanks!".
7. Do not emit redundant facts across episodes. But if a later episode adds new
   details (brand, count, location), extract the more detailed version as a new fact.
8. Prefer DIRECT speaker-to-target edges over fragmenting a relationship through
   descriptive intermediaries. When a person is/visits/wants/recommends a
   Location, performs at/works for an Organization, owns/uses/likes an Object,
   or feels strongly about a Topic, emit the edge directly from the person —
   do not route the relationship through scenery nouns, observed features, or
   ad-hoc descriptive entities.
   GOOD: Calvin -> TOOK_PHOTO_IN -> Tokyo
         Calvin -> PERFORMED_IN -> Tokyo
         Dave -> WANTS_TO_VISIT -> Tokyo
   BAD:  city lights -> LOCATED_IN -> Tokyo,
         night skyline -> LOCATED_IN -> Tokyo,
         insane crowd -> LOCATED_IN -> Tokyo
         (the descriptive scenery is observation about the location, not its
         own subject — Calvin's and Dave's actual relationships to Tokyo
         become unretrievable. Put the scenery details inside fact text.)
   The same applies for organizations and possessions: each speaker-relevant
   thing should have at least one edge with the speaker (or an entity they
   own/identify with) as source, not an observation noun.

OUTPUT DISCIPLINE:
- Entity `name` is a literal mention from CURRENT MESSAGES, ≤5 words. NEVER use a full
  sentence, action item, goal statement, or quoted aspiration as a name.
  BAD: "Establish a firm training/onboarding program",
       "Secure competitive advantage through IP",
       "Decide whether to expand into Europe next quarter".
  GOOD (terse-name fallback for the same source content): "training program",
       "competitive advantage", "European expansion" — extract the topical noun
       phrase, not the full proposition. Multi-word topic names like "watercolor
       painting" or "VR gaming" remain valid (see ENTITY RULE 4).
- The `fact` field is one self-contained sentence. NEVER include reasoning, hedging
  ("appears to", "implies", "suggests"), parenthetical commentary, or schema-description text.
- `relation_type` is SCREAMING_SNAKE_CASE letters/underscores only. NEVER spaces,
  punctuation, or sentences.
- Output ONLY the JSON specified by the response schema. No preamble, no trailing notes,
  no explanation of choices.

<NEGATIVE EXAMPLES>
Each example shows the source phrasing, what NOT to extract as an entity, and
what to keep instead. The skipped content still survives — inside fact text on
the surviving entity.

A) Multiple-choice / response-option scaffolding
   Source (assistant): "Reply with one of: Strongly disagree, Disagree, Agree,
   or Strongly agree."
   SKIP entities: "Agree", "Strongly disagree", "the four answers", "responses".
   KEEP: nothing — this is template instruction, not a fact about the user.

B) Specific clock times
   Source: "The sun rises around 8:47 am in Stockholm on the winter solstice."
   SKIP entities: "8:47 am", "2:48 pm".
   KEEP: "Stockholm", "winter solstice". The time stays in the fact text:
   "The sun rises around 8:47 am in Stockholm on the winter solstice."

C) Quantities / durations / prices / recipe amounts
   Source: "Berlin and London experience approximately 7.5 hours of daylight
   on the winter solstice."
   SKIP entities: "7.5 hours of daylight", "7-8 hours", "6 hours of daylight".
   KEEP: "Berlin", "London", "winter solstice". Duration stays inside the fact.

   Source: "Mix 1 cup granulated white sugar into 4 cups water to make nectar."
   SKIP entities: "1 cup granulated white sugar", "4 cups water".
   KEEP: "sugar-water nectar" (the recipe topic). The amounts stay in the fact.

D) Geographic coordinates
   Source: "Melbourne is located approximately 37 degrees south of the equator;
   Stockholm is around 59 degrees north."
   SKIP entities: "37 degrees south of the equator", "59 degrees north".
   KEEP: "Melbourne", "Stockholm", "equator". The latitude stays in the fact.

E) Imperative verb-phrase advice from tip lists
   Source (assistant): "Tips for saving money on groceries: Buy in bulk; Cook
   in bulk; Plan your meals; Shop sales; Use cashback apps."
   SKIP entities: "Buy in bulk", "Cook in bulk", "Plan your meals",
   "Shop sales", "Use cashback apps".
   KEEP: "saving money on groceries" (the topical noun phrase). Each tip lives
   inside a fact attached to that topic, not as its own node.

F) Quoted slogans / idioms / loaded phrases
   Source (user): "I believe in 'from each according to his ability, to each
   according to his need' — the rich are too highly taxed though."
   SKIP entities: "from each according to his ability...", "the enemy of my
   enemy", "the rich", "genuinely disadvantaged".
   KEEP: the User entity. The belief and opinion go into facts in plain
   language (e.g. user -> BELIEVES_IN -> Marxist distribution principle).

G) Direct speaker-to-target edges (no fragmenting through scenery)
   Source: Calvin: "I took that pic in Tokyo last night. The skyline was
   stunning! [...]"  Dave: "Wow, the night skyline really pops with those
   city lights. I gotta take a trip there soon!"  Calvin (later): "Touring
   with Frank Ocean last week was wild. Tokyo was unreal — the crowd was
   insane."
   SKIP edge sources/targets: city lights -> Tokyo,
        night skyline -> Tokyo, insane crowd -> Tokyo.
   KEEP edges: Calvin -> TOOK_PHOTO_IN -> Tokyo
               Calvin -> PERFORMED_IN -> Tokyo
               Calvin -> TOURED_WITH -> Frank Ocean
               Dave -> WANTS_TO_VISIT -> Tokyo
   Descriptive scenery ("stunning skyline", "city lights pop", "insane
   crowd") goes inside the fact text on these direct edges, not as its
   own edges.
</NEGATIVE EXAMPLES>

<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>
{_build_edge_types_section(context.get('edge_types'))}
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
