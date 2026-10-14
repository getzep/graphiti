from abc import ABC, abstractmethod
from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field

from .models import ChatPrompt, Message, PromptFunction, PromptVersion
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
        ...,
        description=(
            'The name of the source entity. MUST exactly match an entity `name` in '
            'extracted_entities (including value-like endpoints such as versions).'
        ),
    )
    target_entity_name: str = Field(
        ...,
        description=(
            'The name of the target entity. MUST exactly match an entity `name` in '
            'extracted_entities (including value-like endpoints such as versions).'
        ),
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


def _build_edge_types_section(
    edge_types: list[dict] | None, strict_edge_types: bool = False
) -> str:
    """Build the optional FACT TYPES section for the combined prompt."""
    if not edge_types:
        return ''
    relation_type_rules = (
        '- Use one of the provided fact_type_name values as the relation_type.\n'
        '- If a relationship does not match any FACT TYPE, skip it. Do not derive a '
        'relation_type.'
        if strict_edge_types
        else '- When a relationship matches a FACT TYPE, use that fact_type_name as the '
        'relation_type.\n'
        '- If no FACT TYPE fits, derive a relation_type in SCREAMING_SNAKE_CASE\n'
        '  (e.g., WORKS_AT, LIVES_IN, IS_FRIENDS_WITH).'
    )
    return f"""
<FACT TYPES>
{to_prompt_json(edge_types)}
</FACT TYPES>
RELATION TYPE RULES:
{relation_type_rules}
"""


def _build_custom_instructions_section(custom_extraction_instructions: str | None) -> str:
    """Build the optional CUSTOM INSTRUCTIONS section for the combined prompt."""
    if not custom_extraction_instructions:
        return ''
    return f"""
<CUSTOM INSTRUCTIONS>
{custom_extraction_instructions}
</CUSTOM INSTRUCTIONS>
CUSTOM INSTRUCTION RULES:
- CUSTOM INSTRUCTIONS come from the owner of this graph. Apply them. When a CUSTOM INSTRUCTION conflicts with a rule above, the CUSTOM INSTRUCTIONS wins.
"""


_SYSTEM_PROMPT = 'You build the long-term memory of an AI agent. From each conversation you extract the entities and the relationship facts that the agent will need later to answer questions, personalize responses, and continue work. The original conversation will not be available later: only your entities and facts will remain, so each fact must stand alone.'


_C2_INSTRUCTION_TEXT = """# Goal
Turn CURRENT MESSAGES into a knowledge graph: a list of entities and a list of facts (edges) between them. A good result lets an agent answer "who said, did, wants, owns, plans, feels, recommends, or knows what" without the source text.

# What good output looks like
- Complete: every episode, post, and speaker with real content is covered, including the later ones in a long input. Preferences, opinions, plans, commitments, requests, questions, emotions, experiences, advice, and results all count. Skip only content-free turns ("Hi!", "Thanks!", "Bye!").
- Attributed: when a named person, author, account, or organization states, recommends, or does something, that actor is the source entity, and the fact text names the actor.
- Self-contained: each fact is one plain sentence that uses entity names (no pronouns), keeps the specific details (numbers, dates, prices, versions, names), and makes sense with no other context.
- Compact: one fact for one statement. Keep a list of items from one statement in one fact ("Marina plans to change into a sweater and trousers."), unless each item has its own details. Do not make a fact or an entity for each list item. Not redundant: If the same fact appears in more than one episode, emit it once and list all of its episode indices.

# Entities
- Speakers. Name each speaker with the most complete name available. When a message header has profile fields (first_name and last_name, given_name and family_name, user_alias, screen_name), use the full personal name, for example "Marisol Beckett". Use "User" only when no name is available.
- Other entities: named people, organizations, places, products, files, tickets, projects, events, documents, and named or described objects, plus hobbies and activities that someone does ("road cycling", "watercolor painting"). Use the most specific form mentioned, at most 5 words, as a noun phrase. For a person's possession, pet, project, or creation, use a possessive name ("Calvin's guitar", "Marina's sweater"), not the bare noun.
- Do not make entities from: pronouns or unresolved references ("the city", "this issue"); vague abstractions ("growth", "life", "time"); clock times, dates, quantities, prices, durations, measurements, or coordinates (keep these in the fact text, unless the value is itself an edge endpoint such as a version string); answer options or template tokens ("Agree", "Strongly disagree"); imperative tips ("Buy in bulk" becomes the topic "bulk buying"); quoted slogans or idioms; tutorial example values ("/path/to/source", "file.txt").
- Each entity appears once. Classify it with an entity_type_id from ENTITY TYPES. Extract entities only from CURRENT MESSAGES; use PREVIOUS MESSAGES and [CONTEXT EPISODE] messages only to resolve references.

# Facts
- source_entity_name and target_entity_name must be names from extracted_entities. If an endpoint is a value such as a version, add it as an entity.
- Connect the actor directly to the thing: "Calvin -> PERFORMED_IN -> Tokyo", "Mira Sen -> RECOMMENDS_METRICS_FOR -> Ember & Oak". Do not route a relationship through descriptive nouns ("city lights -> LOCATED_IN -> Tokyo"), and do not turn an actor's claim into a fact between two nouns that the actor mentioned.
- Use a self-referencing fact (source and target are the same entity) only when a statement is about one entity and no other extracted entity fits, for example a routine, a feeling, or a health state ("Evan has a knee injury.").
- When a fact involves two extracted entities, use both as source and target.
- Use PREVIOUS MESSAGES and [CONTEXT EPISODE] messages to resolve what the current message refers to, and write the resolved fact ("all the hard work paid off" becomes what paid off). [CONTEXT EPISODE] messages are conversation turns between the numbered episodes, shown so you can read the conversation in order.
- Set episode_indices to the 0-based [Episode N] numbers that the fact comes from. Messages headed [CONTEXT EPISODE] are not numbered episodes: do not extract facts from them and do not give them an episode index.
- Write each entity name and each fact in the dominant language of its source episode. Keep proper names as written. Write relation_type as an English SCREAMING_SNAKE_CASE label.
- A fact states what the source says. Do not add reasoning, hedges ("appears to", "suggests"), or commentary.

# Output
Return only the JSON object: {{"extracted_entities":[{{"name":"...","entity_type_id":0}}],"edges":[{{"source_entity_name":"...","target_entity_name":"...","relation_type":"...","fact":"...","episode_indices":[0]}}]}}."""


def _build_extract_user_prompt(context: dict[str, Any]) -> str:
    return f"""{_C2_INSTRUCTION_TEXT}

<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>
{_build_edge_types_section(context.get('edge_types'), context.get('strict_edge_types', False))}
{_build_custom_instructions_section(context.get('custom_extraction_instructions'))}
<PREVIOUS MESSAGES>
{to_prompt_json([ep for ep in context['previous_episodes']])}
</PREVIOUS MESSAGES>

<CURRENT MESSAGES>
{context['episode_content']}
</CURRENT MESSAGES>
"""


def extract_message(context: dict[str, Any]) -> list[Message]:
    return [
        Message(role='system', content=_SYSTEM_PROMPT),
        Message(role='user', content=_build_extract_user_prompt(context)),
    ]


versions: Versions = {
    'extract_message': extract_message,
}


class ExtractNodesAndEdgesPrompts(ABC):
    @abstractmethod
    def extract_message(self, context: dict[str, Any]) -> ChatPrompt: ...


class DefaultExtractNodesAndEdgesPrompts(ExtractNodesAndEdgesPrompts):
    def extract_message(self, context: dict[str, Any]) -> ChatPrompt:
        return ChatPrompt.from_messages(extract_message(context))
