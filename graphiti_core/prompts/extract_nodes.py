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

from pydantic import AliasChoices, BaseModel, Field

from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS

from .models import Message, PromptFunction, PromptVersion
from .prompt_helpers import to_prompt_json
from .snippets import summary_instructions

# Knowledge Graph Builder's Principles - fundamental guidelines for entity extraction
KNOWLEDGE_GRAPH_PRINCIPLES = """
## KNOWLEDGE GRAPH BUILDER'S PRINCIPLES (Must Follow)

You are building an enterprise knowledge graph. Every entity you extract will become a permanent node that other documents can reference and build relationships upon.

Before extracting any entity, verify it passes ALL four principles:

1. **Permanence Principle**: Only extract entities that have lasting value beyond this single document.
   Ask: "Will this entity still be meaningful and useful 6 months from now?"

2. **Connectivity Principle**: Only extract entities that can form meaningful relationships with other entities.
   Ask: "Can this entity connect to other concepts in a knowledge graph?"

3. **Independence Principle**: Only extract entities that are self-explanatory without the source document.
   Ask: "Would someone understand this entity name without reading the original text?"

4. **Domain Value Principle**: Only extract entities that represent real domain knowledge, not document artifacts.
   Ask: "Is this a concept a domain expert would recognize and care about?"

**EXTRACTION DECISION**: Apply a HIGH BAR for extraction.
- When uncertain about ANY principle, do NOT extract.
- Prefer PRECISION over RECALL - a smaller, high-quality knowledge graph is far more valuable than a large, noisy one.
- If an entity only makes sense within the context of a specific process, workflow, or document structure, it likely fails the Independence Principle.
- Generic terms, internal process steps, and transient concepts should NOT be extracted.
"""


class ExtractedEntity(BaseModel):
    name: str = Field(
        ...,
        description='Name of the extracted entity',
        validation_alias=AliasChoices('name', 'entity_name', 'entity'),
    )
    entity_type_id: int = Field(
        description='ID of the classified entity type. '
        'Must be one of the provided entity_type_id integers.',
    )
    reasoning: str = Field(
        ...,
        description='Brief reasoning (1-2 sentences) explaining why this entity was extracted '
        'and why the chosen entity_type_id is correct based on the ENTITY TYPES descriptions.',
    )


# Type scoring models for more deliberate classification decisions
class TopTypeCandidate(BaseModel):
    """Top candidate type with score and reasoning."""
    type_id: int = Field(..., description='The entity_type_id being scored')
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description='Confidence score from 0.0 to 1.0. '
                    '1.0 = perfect match, 0.0 = definitely not this type',
    )
    reasoning: str = Field(
        ...,
        description='Brief explanation of why this score was given',
    )


class ExtractedEntityWithScores(BaseModel):
    """Entity with top 3 candidate types (reduced token output)."""
    name: str = Field(
        ...,
        description='Name of the extracted entity',
        validation_alias=AliasChoices('name', 'entity_name', 'entity'),
    )
    top_candidates: list[TopTypeCandidate] = Field(
        ...,
        description='Top 3 most likely entity types with scores. '
                    'Sorted by score descending. Maximum 3 candidates.',
        min_length=1,
        max_length=3,
    )
    final_type_id: int = Field(
        ...,
        description='The entity_type_id of the best matching type (highest score)',
    )


class ExtractedEntitiesWithScores(BaseModel):
    """Collection of extracted entities with top candidate scores."""
    extracted_entities: list[ExtractedEntityWithScores] = Field(
        ...,
        description='List of extracted entities with top 3 type candidates',
        validation_alias=AliasChoices('extracted_entities', 'entities'),
    )


# Second-pass type resolution for ambiguous entities (multiple high scores)
class AmbiguousEntityInput(BaseModel):
    """Input for ambiguous entity type resolution."""
    name: str = Field(..., description='Name of the entity')
    candidate_type_ids: list[int] = Field(
        ...,
        description='List of candidate type_ids to choose from',
    )


class CandidateTypeScore(BaseModel):
    """Score for a candidate type in second-pass resolution."""
    type_id: int = Field(..., description='The entity_type_id being scored')
    score: float = Field(..., description='Confidence score from 0.0 to 1.0')
    reasoning: str = Field(..., description='Why this type received this score')


class ResolvedEntityType(BaseModel):
    """Resolved type for an ambiguous entity with scoring."""
    name: str = Field(..., description='Name of the entity')
    chosen_type_id: int = Field(..., description='The chosen entity_type_id (highest score)')
    reasoning: str = Field(
        ...,
        description='Explanation of why this type is the best fit',
    )
    candidate_scores: list[CandidateTypeScore] = Field(
        ...,
        description='Scores for each candidate type, sorted by score descending',
    )


class ResolvedEntityTypes(BaseModel):
    """Batch resolution results for ambiguous entities."""
    resolutions: list[ResolvedEntityType] = Field(
        ...,
        description='Type resolution for each ambiguous entity',
    )


# Keep old TypeScore as alias for backward compatibility
TypeScore = TopTypeCandidate


# Phase 2: Subtype refinement models (two-phase type recognition)
class RefinedEntityType(BaseModel):
    """Result of Phase 2 subtype refinement for a single entity."""

    entity_name: str = Field(..., description='Name of the entity being refined')
    refined_type: str | None = Field(
        None,
        description='The specific subtype name (e.g., "Ticket"), or null if no subtype matches well',
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description='Confidence score for the refinement decision',
    )
    reasoning: str = Field(..., description='Brief explanation for the decision')


class RefinedEntityTypes(BaseModel):
    """Batch result of Phase 2 subtype refinement."""

    refinements: list[RefinedEntityType] = Field(
        ...,
        description='Type refinement results for each entity',
    )


class ExtractedEntities(BaseModel):
    extracted_entities: list[ExtractedEntity] = Field(
        ...,
        description='List of extracted entities',
        validation_alias=AliasChoices('extracted_entities', 'entities'),
    )


class MissedEntities(BaseModel):
    missed_entities: list[str] = Field(
        default_factory=list,
        description="Names of entities that weren't extracted but should be"
    )
    entities_to_remove: list[str] = Field(
        default_factory=list,
        description='Names of extracted entities that should NOT be in the knowledge graph '
        '(e.g., too generic, transient concepts, document artifacts)',
    )
    entities_to_reclassify: list['EntityReclassification'] = Field(
        default_factory=list,
        description='Entities that were misclassified and should be assigned a different type',
    )


class EntityReclassification(BaseModel):
    name: str = Field(..., description='Name of the entity to reclassify')
    new_type: str = Field(
        ...,
        description='The correct entity type name from VALID ENTITY TYPES. Must be an exact match.',
    )
    reason: str = Field(..., description='Brief explanation of why this type is more appropriate')


class EntitiesToFilter(BaseModel):
    entities_to_remove: list[str] = Field(
        ...,
        description='Names of entities that should be removed from the knowledge graph',
    )
    entities_to_reclassify: list[EntityReclassification] = Field(
        default_factory=list,
        description='Entities that were misclassified and should be assigned a different type',
    )
    reasoning: str = Field(
        ...,
        description='Brief explanation of why these entities were flagged for removal or reclassification',
    )


# Step 1: Validate entity against its assigned type
class EntityValidationItem(BaseModel):
    name: str = Field(..., description='Name of the entity')
    is_valid: bool = Field(..., description='True if entity matches its assigned type definition')
    reason: str = Field(..., description='Brief explanation of why it matches or does not match')


class EntityValidationResult(BaseModel):
    validations: list[EntityValidationItem] = Field(
        ..., description='Validation results for each entity'
    )


class EntityClassificationTriple(BaseModel):
    uuid: str = Field(description='UUID of the entity')
    name: str = Field(description='Name of the entity')
    entity_type: str | None = Field(
        default=None,
        description='Type of the entity. Must be one of the provided types or None',
    )


class EntityClassification(BaseModel):
    entity_classifications: list[EntityClassificationTriple] = Field(
        ..., description='List of entities classification triples.'
    )


class EntitySummary(BaseModel):
    summary: str = Field(
        ...,
        description=f'Summary containing the important information about the entity. Under {MAX_SUMMARY_CHARS} characters.',
    )


# EasyOps: Batch summary extraction models
class EntitySummaryItem(BaseModel):
    """Summary for a single entity in batch extraction."""

    entity_id: int = Field(..., description='The ID of the entity from the input list')
    summary: str = Field(
        ...,
        description=f'Summary containing the important information about the entity. Under {MAX_SUMMARY_CHARS} characters.',
    )


class EntitySummaries(BaseModel):
    """Batch of entity summaries."""

    summaries: list[EntitySummaryItem] = Field(
        ..., description='List of summaries for each entity'
    )


class Prompt(Protocol):
    extract_message: PromptVersion
    extract_json: PromptVersion
    extract_text: PromptVersion
    extract_text_with_scores: PromptVersion
    extract_message_with_scores: PromptVersion
    reflexion: PromptVersion
    filter_entities: PromptVersion
    validate_entity_types: PromptVersion
    classify_nodes: PromptVersion
    extract_attributes: PromptVersion
    extract_summary: PromptVersion
    extract_summaries_bulk: PromptVersion  # EasyOps: batch summary extraction
    resolve_ambiguous_types: PromptVersion


class Versions(TypedDict):
    extract_message: PromptFunction
    extract_json: PromptFunction
    extract_text: PromptFunction
    extract_text_with_scores: PromptFunction
    extract_message_with_scores: PromptFunction
    reflexion: PromptFunction
    filter_entities: PromptFunction
    validate_entity_types: PromptFunction
    classify_nodes: PromptFunction
    extract_attributes: PromptFunction
    extract_summary: PromptFunction
    extract_summaries_bulk: PromptFunction  # EasyOps: batch summary extraction
    resolve_ambiguous_types: PromptFunction


def extract_message(context: dict[str, Any]) -> list[Message]:
    sys_prompt = f"""You are an AI assistant that extracts entity nodes from conversational messages for building an enterprise knowledge graph.
    Your primary task is to extract and classify the speaker and other significant entities mentioned in the conversation.

{KNOWLEDGE_GRAPH_PRINCIPLES}"""

    user_prompt = f"""
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

<PREVIOUS MESSAGES>
{to_prompt_json([ep for ep in context['previous_episodes']])}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{context['episode_content']}
</CURRENT MESSAGE>

Instructions:

You are given a conversation context and a CURRENT MESSAGE. Your task is to extract **entity nodes** mentioned **explicitly or implicitly** in the CURRENT MESSAGE.
Pronoun references such as he/she/they or this/that/those should be disambiguated to the names of the
reference entities. Only extract distinct entities from the CURRENT MESSAGE. Don't extract pronouns like you, me, he/she/they, we/us as entities.

1. **Speaker Extraction**: Always extract the speaker (the part before the colon `:` in each dialogue line) as the first entity node.
   - If the speaker is mentioned again in the message, treat both mentions as a **single entity**.

2. **Entity Identification**:
   - Extract all significant entities, concepts, or actors that are **explicitly or implicitly** mentioned in the CURRENT MESSAGE.
   - **Exclude** entities mentioned only in the PREVIOUS MESSAGES (they are for context only).

3. **Entity Classification**:
   - Use the descriptions in ENTITY TYPES to classify each extracted entity.
   - Assign the appropriate `entity_type_id` for each one.
   - **CRITICAL**: Carefully read the 【是】(IS) and 【不是】(IS NOT) examples in each entity type description to ensure correct classification.

4. **Reasoning Requirement**:
   - For each extracted entity, provide a brief `reasoning` (1-2 sentences) explaining:
     - Why you identified this as a significant entity
     - Why the chosen entity_type_id is correct based on the ENTITY TYPES descriptions
   - This forces careful consideration before classification.

5. **Exclusions**:
   - Do NOT extract entities representing relationships or actions.
   - Do NOT extract dates, times, or other temporal information—these will be handled separately.

6. **Formatting**:
   - Be **explicit and unambiguous** in naming entities (e.g., use full names when available).

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    sys_prompt = f"""You are an AI assistant that extracts entity nodes from JSON for building an enterprise knowledge graph.
    Your primary task is to extract and classify relevant entities from JSON files.

{KNOWLEDGE_GRAPH_PRINCIPLES}"""

    user_prompt = f"""
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

<SOURCE DESCRIPTION>:
{context['source_description']}
</SOURCE DESCRIPTION>
<JSON>
{context['episode_content']}
</JSON>

{context['custom_prompt']}

Given the above source description and JSON, extract relevant entities from the provided JSON.
For each entity extracted, also determine its entity type based on the provided ENTITY TYPES and their descriptions.
Indicate the classified entity type by providing its entity_type_id.

Guidelines:
1. Extract all entities that the JSON represents. This will often be something like a "name" or "user" field
2. Extract all entities mentioned in all other properties throughout the JSON structure
3. Do NOT extract any properties that contain dates
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:
    sys_prompt = f"""You are an AI assistant that extracts entity nodes from text for building an enterprise knowledge graph.
    Your primary task is to extract and classify significant entities mentioned in the provided text.

{KNOWLEDGE_GRAPH_PRINCIPLES}"""

    user_prompt = f"""
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

<TEXT>
{context['episode_content']}
</TEXT>

Given the above text, extract entities from the TEXT that are explicitly or implicitly mentioned.
For each entity extracted, also determine its entity type based on the provided ENTITY TYPES and their descriptions.
Indicate the classified entity type by providing its entity_type_id.

{context['custom_prompt']}

Guidelines:
1. Extract significant entities, concepts, or actors mentioned in the conversation.
2. Avoid creating nodes for relationships or actions.
3. Avoid creating nodes for temporal information like dates, times or years (these will be added to edges later).
4. Be as explicit as possible in your node names, using full names and avoiding abbreviations.
5. **CRITICAL**: Carefully read the 【是】(IS) and 【不是】(IS NOT) examples in each entity type description to ensure correct classification.
6. **Reasoning Requirement**: For each extracted entity, provide a brief `reasoning` (1-2 sentences) explaining:
   - Why you identified this as a significant entity
   - Why the chosen entity_type_id is correct based on the ENTITY TYPES descriptions
   - This forces careful consideration before classification.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text_with_scores(context: dict[str, Any]) -> list[Message]:
    """Extract entities with top 3 type candidates for efficient classification.

    This version only outputs the top 3 most likely types per entity,
    reducing token consumption while maintaining classification accuracy.
    """
    sys_prompt = f"""You are an AI assistant that extracts entity nodes from text for building an enterprise knowledge graph.
Your task is to extract significant entities and identify the TOP 3 most likely types for each.

{KNOWLEDGE_GRAPH_PRINCIPLES}

**TYPE SCORING PROCESS**:
For each entity you extract:
1. Quickly scan ALL entity types to identify the 3 most likely candidates
2. Score each of these 3 candidates from 0.0 to 1.0
3. Provide reasoning for each score
4. Set final_type_id to the highest-scoring candidate"""

    user_prompt = f"""
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

<TEXT>
{context['episode_content']}
</TEXT>

Extract entities from the text. For each entity, output ONLY the top 3 most likely types.

{context['custom_prompt']}

**SCORING GUIDELINES**:
- Score 0.9-1.0: Perfect match - entity clearly fits the type's IS examples
- Score 0.7-0.8: Strong match - entity likely fits this type
- Score 0.5-0.6: Partial match - some characteristics match
- Score 0.0-0.4: Weak match - poor fit but still a candidate

**OUTPUT REQUIREMENTS**:
For each extracted entity:
1. `top_candidates`: List exactly 3 candidates (or fewer if less than 3 types exist), sorted by score descending
   - Each candidate needs: type_id, score, reasoning (why this score)
2. `final_type_id`: The type_id with the highest score

**ENTITY EXTRACTION GUIDELINES**:
1. Extract significant entities, concepts, or actors mentioned in the text
2. Avoid creating nodes for relationships, actions, or temporal information
3. Be explicit in entity names, using full names when available
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_message_with_scores(context: dict[str, Any]) -> list[Message]:
    """Extract entities from messages with top 3 type candidates for efficient classification."""
    sys_prompt = f"""You are an AI assistant that extracts entity nodes from conversational messages for building an enterprise knowledge graph.
Your task is to extract entities and identify the TOP 3 most likely types for each.

{KNOWLEDGE_GRAPH_PRINCIPLES}

**TYPE SCORING PROCESS**:
For each entity you extract:
1. Quickly scan ALL entity types to identify the 3 most likely candidates
2. Score each of these 3 candidates from 0.0 to 1.0
3. Provide reasoning for each score
4. Set final_type_id to the highest-scoring candidate"""

    user_prompt = f"""
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

<PREVIOUS MESSAGES>
{to_prompt_json([ep for ep in context['previous_episodes']])}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{context['episode_content']}
</CURRENT MESSAGE>

Extract entities from the CURRENT MESSAGE. For each entity, output ONLY the top 3 most likely types.

{context['custom_prompt']}

**EXTRACTION RULES**:
1. Always extract the speaker as the first entity
2. Extract significant entities mentioned in CURRENT MESSAGE only
3. Disambiguate pronouns to actual entity names

**SCORING GUIDELINES**:
- Score 0.9-1.0: Perfect match - entity clearly fits the type's IS examples
- Score 0.7-0.8: Strong match - entity likely fits this type
- Score 0.5-0.6: Partial match - some characteristics match
- Score 0.0-0.4: Weak match - poor fit but still a candidate

**OUTPUT REQUIREMENTS**:
For each extracted entity:
1. `top_candidates`: List exactly 3 candidates (or fewer if less than 3 types exist), sorted by score descending
   - Each candidate needs: type_id, score, reasoning (why this score)
2. `final_type_id`: The type_id with the highest score
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that reviews entity extraction quality. You perform both:
1. **Positive reflexion**: Identify entities that SHOULD have been extracted but weren't
2. **Negative reflexion**: Identify entities that should NOT have been extracted or were misclassified

Review each extracted entity against the Knowledge Graph Builder's Principles:

1. **Permanence Principle**: Does it have lasting value beyond this document?
2. **Connectivity Principle**: Can it meaningfully connect to other entities?
3. **Independence Principle**: Is the name self-explanatory without the source text?
4. **Domain Value Principle**: Does it represent real domain knowledge, not document artifacts?

An entity should be REMOVED if it fails ANY of these principles AND cannot be reclassified to a valid type.

**CRITICAL**: Do NOT trust pre-assigned types blindly. You MUST re-validate the entity against the type's definition in VALID ENTITY TYPES. If the entity does not actually match its assigned type's criteria (especially the IS/IS NOT examples in the type description):
- If it matches a DIFFERENT valid type, RECLASSIFY it
- If it doesn't match ANY valid type, REMOVE it"""

    # Build entity types reference if available
    entity_types_ref = ''
    if context.get('entity_types'):
        entity_types_ref = '\n<VALID ENTITY TYPES>\n'
        for et in context['entity_types']:
            name = et.get('entity_type_name', et.get('name', ''))
            desc = et.get('entity_type_description', et.get('description', ''))
            if name and name != 'Entity':
                entity_types_ref += f"- {name}: {desc}\n"
        entity_types_ref += '</VALID ENTITY TYPES>\n'

    user_prompt = f"""
<PREVIOUS MESSAGES>
{to_prompt_json([ep for ep in context['previous_episodes']])}
</PREVIOUS MESSAGES>
<CURRENT MESSAGE>
{context['episode_content']}
</CURRENT MESSAGE>

<EXTRACTED ENTITIES>
{context['extracted_entities']}
</EXTRACTED ENTITIES>
{entity_types_ref}
Review the extraction quality and provide:

1. **missed_entities**: Names of significant entities mentioned in CURRENT MESSAGE that weren't extracted
   - Focus on entities with lasting value that can connect to other concepts

2. **entities_to_remove**: Names of extracted entities that should be REMOVED because they:
   - Are too generic or vague (e.g., "it", "the system", "this thing")
   - Are transient/temporary concepts with no lasting value
   - Are document artifacts, not real domain knowledge
   - Cannot meaningfully connect to other entities
   - MISCLASSIFIED and don't match ANY valid type

3. **entities_to_reclassify**: Entities with wrong type assignment
   - For each entity with a pre-assigned type, RE-VALIDATE against that type's definition
   - Check if it matches the type's IS examples (keep current type)
   - Check if it matches the type's IS NOT examples (needs reclassification)
   - Only include if the entity matches a DIFFERENT valid type
   - Provide: name, new_type (must be exact match from VALID ENTITY TYPES), reason

**Common Misclassifications to Watch For**:
- Metric IDs (like system_cpu_cores, memory_usage) are NOT Components - Components are deployable services
- Table columns or field names are NOT Models - Models are schema definitions
- Generic technical terms are NOT Features - Features have specific product functionality

Return empty lists if the extraction quality is good.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def filter_entities(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are a knowledge graph quality reviewer. Your task is to identify entities that should NOT be in an enterprise knowledge graph, or entities that were misclassified and need type correction.

Review each extracted entity against the Knowledge Graph Builder's Principles:

1. **Permanence Principle**: Does it have lasting value beyond this document?
2. **Connectivity Principle**: Can it meaningfully connect to other entities?
3. **Independence Principle**: Is the name self-explanatory without the source text?
4. **Domain Value Principle**: Does it represent real domain knowledge, not document artifacts?

An entity should be REMOVED if it fails ANY of these principles AND cannot be reclassified to a valid type.

**CRITICAL**: Entities may have a pre-assigned "type" from the extraction step. Do NOT trust this type blindly. You MUST re-validate the entity against the type's definition in VALID ENTITY TYPES. If the entity does not actually match its assigned type's criteria (especially the IS/IS NOT examples in the type description):
- If it matches a DIFFERENT valid type, RECLASSIFY it
- If it doesn't match ANY valid type, REMOVE it"""

    # Build entity types reference if available
    entity_types_ref = ''
    if context.get('entity_types'):
        entity_types_ref = '\n<VALID ENTITY TYPES>\n'
        for et in context['entity_types']:
            if et.get('entity_type_name') != 'Entity':  # Skip default type
                entity_types_ref += f"- {et.get('entity_type_name')}: {et.get('entity_type_description', '')}\n"
        entity_types_ref += '</VALID ENTITY TYPES>\n'

    # Format entities - now includes summary and type if available
    entities_json = to_prompt_json(context['extracted_entities'])

    user_prompt = f"""
<EXTRACTED ENTITIES>
{entities_json}
</EXTRACTED ENTITIES>

<SOURCE TEXT>
{context['episode_content']}
</SOURCE TEXT>
{entity_types_ref}
Review each extracted entity. The entity info includes name, and may include summary and type.
Use the summary to better understand what the entity represents.

**Decision Process**:
1. If the entity has a pre-assigned type, RE-VALIDATE it against that type's definition in VALID ENTITY TYPES:
   - Check if it matches the type's criteria (judgment standards)
   - Check if it matches the IS examples (should be kept with current type)
   - Check if it matches the IS NOT examples (needs reclassification or removal)
   - If MISCLASSIFIED but matches a DIFFERENT valid type → add to entities_to_reclassify
   - If MISCLASSIFIED and doesn't match ANY valid type → add to entities_to_remove
2. If the entity has no type or type is "Entity", check if it matches any valid type:
   - If it matches a valid type → add to entities_to_reclassify with the correct type
   - If it doesn't match any type, apply the four principles strictly

**Common Misclassifications to Watch For**:
- Metric IDs (like system_cpu_cores, system_memory_total) are NOT Components - Components are deployable services
- Table columns or field names are NOT CmdbModels - CmdbModels are model definitions
- Generic technical terms are NOT Features - Features have menu entries in the product

**Output Format**:
- entities_to_remove: Names of entities that fail quality checks and don't match any valid type
- entities_to_reclassify: Entities with wrong type that should be corrected (include name, new_type, reason)
- reasoning: Brief explanation of your decisions

Return empty lists if all entities pass the quality check with correct types.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def validate_entity_types(context: dict[str, Any]) -> list[Message]:
    """Step 1: Validate each entity against its assigned type definition only.

    This is a focused validation - each entity is checked against ONLY its current type.
    No episode content needed, just entity summary + type definition.
    """
    sys_prompt = f"""You are a type validator for a knowledge graph. Your task is to verify if each entity truly matches its assigned type definition.

For each entity, you will receive:
- Entity name and summary
- The FULL definition of its assigned type (including IS/IS NOT examples)

You must check if the entity matches the type's criteria.

{KNOWLEDGE_GRAPH_PRINCIPLES}"""

    # Build entities with their type definitions
    entities_with_defs = []
    for entity in context['entities']:
        entity_info = {
            'name': entity['name'],
            'summary': entity.get('summary', ''),
            'assigned_type': entity.get('type', 'Entity'),
            'type_definition': entity.get('type_definition', ''),
        }
        entities_with_defs.append(entity_info)

    entities_json = to_prompt_json(entities_with_defs)

    user_prompt = f"""
<ENTITIES TO VALIDATE>
{entities_json}
</ENTITIES TO VALIDATE>

For each entity, determine if it truly matches its assigned type based on the type_definition provided.

**Validation Rules**:
1. Read the type_definition carefully, especially the IS and IS NOT examples
2. Compare the entity's name and summary against the criteria
3. If the entity matches IS examples → is_valid = true
4. If the entity matches IS NOT examples → is_valid = false
5. If uncertain but the entity could reasonably fit the type → is_valid = true (preserve valuable entities)
6. Provide a clear reason for your decision

Return validation result for EVERY entity in the list.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def classify_nodes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that classifies entity nodes given the context from which they were extracted"""

    user_prompt = f"""
    <PREVIOUS MESSAGES>
    {to_prompt_json([ep for ep in context['previous_episodes']])}
    </PREVIOUS MESSAGES>
    <CURRENT MESSAGE>
    {context['episode_content']}
    </CURRENT MESSAGE>

    <EXTRACTED ENTITIES>
    {context['extracted_entities']}
    </EXTRACTED ENTITIES>

    <ENTITY TYPES>
    {context['entity_types']}
    </ENTITY TYPES>

    Given the above conversation, extracted entities, and provided entity types and their descriptions, classify the extracted entities.

    Guidelines:
    1. Each entity must have exactly one type
    2. Only use the provided ENTITY TYPES as types, do not use additional types to classify entities.
    3. If none of the provided entity types accurately classify an extracted node, the type should be set to None
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts entity properties from the provided text.',
        ),
        Message(
            role='user',
            content=f"""
        Given the MESSAGES and the following ENTITY, update any of its attributes based on the information provided
        in MESSAGES. Use the provided attribute descriptions to better understand how each attribute should be determined.

        Guidelines:
        1. Do not hallucinate entity property values if they cannot be found in the current context.
        2. Only use the provided MESSAGES and ENTITY to set attribute values.

        <MESSAGES>
        {to_prompt_json(context['previous_episodes'])}
        {to_prompt_json(context['episode_content'])}
        </MESSAGES>

        <ENTITY>
        {context['node']}
        </ENTITY>
        """,
        ),
    ]


def extract_summary(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts entity summaries from the provided text.',
        ),
        Message(
            role='user',
            content=f"""
        Given the MESSAGES and the ENTITY, update the summary that combines relevant information about the entity
        from the messages and relevant information from the existing summary.

        {summary_instructions}

        <MESSAGES>
        {to_prompt_json(context['previous_episodes'])}
        {to_prompt_json(context['episode_content'])}
        </MESSAGES>

        <ENTITY>
        {context['node']}
        </ENTITY>
        """,
        ),
    ]


# EasyOps: Batch summary extraction - process multiple entities in one LLM call
def extract_summaries_bulk(context: dict[str, Any]) -> list[Message]:
    """Extract summaries for multiple entities in a single LLM call.

    Context should contain:
    - entities: list of dicts with 'id', 'name', 'summary' (existing), 'entity_types', 'attributes'
    - episode_content: current episode content
    - previous_episodes: list of previous episode contents
    """
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts entity summaries from the provided text. '
            'You will process multiple entities at once and return a summary for each.',
        ),
        Message(
            role='user',
            content=f"""
Given the MESSAGES and the list of ENTITIES, update the summary for EACH entity.
Combine relevant information about each entity from the messages with any existing summary.

{summary_instructions}

<MESSAGES>
{to_prompt_json(context['previous_episodes'])}
{to_prompt_json(context['episode_content'])}
</MESSAGES>

<ENTITIES>
{to_prompt_json(context['entities'])}
</ENTITIES>

For each entity in the list above, provide a summary. Return a JSON object with a "summaries" array.
Each item in the array must have:
- "entity_id": the ID from the input entity
- "summary": the updated summary for that entity

Process ALL entities in the list. Do not skip any entity.
""",
        ),
    ]


def resolve_ambiguous_types(context: dict[str, Any]) -> list[Message]:
    """Resolve ambiguous entity types when multiple candidates have close scores.

    This is a second-pass resolution that requires scoring ALL candidate types,
    similar to the first pass, to ensure consistent and comparable decisions.
    """
    sys_prompt = """You are an AI assistant that helps classify entities into the most appropriate type.
For each entity, you will be given a list of candidate types. Your task is to SCORE each candidate and select the best one.

**SCORING PROCESS** (same as first-pass extraction):
1. Score EACH candidate type from 0.0 to 1.0
2. Provide reasoning for each score
3. Select the highest-scoring type as chosen_type_id

**SCORING GUIDELINES**:
- Score 0.9-1.0: Perfect match - entity clearly fits the type's IS examples
- Score 0.7-0.8: Strong match - entity likely fits this type
- Score 0.5-0.6: Partial match - some characteristics match
- Score 0.0-0.4: Weak match - poor fit

**DECISION CRITERIA**:
1. Consider the entity name and how it's used in the context
2. Match against each candidate type's definition, judgment standards, and IS/IS NOT examples
3. Choose the type that best captures the entity's primary role or nature

**CRITICAL: Each entity must have its own separate resolution with ALL candidate scores.**"""

    user_prompt = f"""
<ORIGINAL TEXT>
{context['episode_content']}
</ORIGINAL TEXT>

<ENTITIES TO CLASSIFY>
{context['ambiguous_entities']}
</ENTITIES TO CLASSIFY>

<CANDIDATE TYPES>
{context['candidate_types']}
</CANDIDATE TYPES>

For each entity listed above:
1. Score EACH candidate type (not just the best one)
2. Provide reasoning for each score
3. Set chosen_type_id to the highest-scoring candidate

**OUTPUT FORMAT**:
Return a `resolutions` array where EACH entity has ALL candidate scores:

Example for 1 entity with 2 candidates:
{{
  "resolutions": [
    {{
      "name": "Entity A",
      "chosen_type_id": 2,
      "reasoning": "Overall reasoning for final choice",
      "candidate_scores": [
        {{"type_id": 2, "score": 0.85, "reasoning": "Why type 2 scored 0.85"}},
        {{"type_id": 6, "score": 0.70, "reasoning": "Why type 6 scored 0.70"}}
      ]
    }}
  ]
}}

**IMPORTANT**:
- Each entity gets its own object in the resolutions array
- candidate_scores must include ALL candidate types, sorted by score descending
- The `name` must match exactly with the entity name from ENTITIES TO CLASSIFY
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def refine_types_from_summary(context: dict[str, Any]) -> list[Message]:
    """Phase 2: Refine entity types from Kernel to specific subtypes.

    Uses entity summary (NOT original episode content) to determine
    if an entity should be classified as a more specific subtype.

    Context:
    - kernel_type: The Kernel type (e.g., "Event")
    - subtypes: List of subtype definitions [{name, description}, ...]
    - entities: List of entities to refine [{name, summary, attributes}, ...]
    """
    sys_prompt = """You are a precise entity type classifier for a knowledge graph.

Given entity summaries that have already been classified into a general Kernel type,
determine if each entity should be refined to a more specific subtype.

**IMPORTANT**:
- Use ONLY the entity name, summary, and attributes provided.
- The original source text is NOT available - do not speculate about information not in the summary.
- If the summary does not clearly indicate a specific subtype, keep the entity at the Kernel type level."""

    subtypes_json = context.get('subtypes', [])
    if isinstance(subtypes_json, list) and subtypes_json:
        subtypes_desc = '\n'.join([
            f"- **{st.get('name', '')}**: {st.get('description', '')}"
            for st in subtypes_json
        ])
    else:
        subtypes_desc = '(No subtypes defined)'

    entities_json = context.get('entities', [])
    if isinstance(entities_json, list):
        entities_desc = '\n'.join([
            f"- **{e.get('name', '')}**\n  Summary: {e.get('summary', 'N/A')}\n  Attributes: {e.get('attributes', {})}"
            for e in entities_json
        ])
    else:
        entities_desc = str(entities_json)

    user_prompt = f"""
The following entities have been classified as "{context.get('kernel_type', 'Entity')}".
Determine if any should be refined to a more specific subtype.

<AVAILABLE SUBTYPES of {context.get('kernel_type', 'Entity')}>
{subtypes_desc}
</AVAILABLE SUBTYPES>

<ENTITIES TO REFINE>
{entities_desc}
</ENTITIES TO REFINE>

For each entity, respond with:
- entity_name: The exact name from input
- refined_type: The subtype name (e.g., "Ticket") or null if none matches well
- confidence: 0.0-1.0 score for your decision
- reasoning: Brief explanation (1-2 sentences)

**Decision Guidelines**:
- Score >= 0.7: Clearly matches a subtype based on summary content
- Score 0.5-0.7: Somewhat matches but not definitive
- Score < 0.5 or null: Does not clearly match any subtype, keep as "{context.get('kernel_type', 'Entity')}"

If the entity summary doesn't provide enough evidence for a specific subtype,
set refined_type to null (the entity stays as "{context.get('kernel_type', 'Entity')}").
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


versions: Versions = {
    'extract_message': extract_message,
    'extract_json': extract_json,
    'extract_text': extract_text,
    'extract_text_with_scores': extract_text_with_scores,
    'extract_message_with_scores': extract_message_with_scores,
    'reflexion': reflexion,
    'filter_entities': filter_entities,
    'validate_entity_types': validate_entity_types,
    'extract_summary': extract_summary,
    'extract_summaries_bulk': extract_summaries_bulk,  # EasyOps: batch summary extraction
    'classify_nodes': classify_nodes,
    'extract_attributes': extract_attributes,
    'resolve_ambiguous_types': resolve_ambiguous_types,
    'refine_types_from_summary': refine_types_from_summary,  # EasyOps: Phase 2 subtype refinement
}
