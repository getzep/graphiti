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
from datetime import datetime
from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion


class Edge(BaseModel):
    relation_type: str = Field(..., description='FACT_PREDICATE_IN_SCREAMING_SNAKE_CASE')
    source_entity_name: str = Field(
        ..., description='The name of the Entity that is the subject of the fact.'
    )
    target_entity_name: str = Field(
        ..., description='The name of the entity that is the Object of the fact.'
    )
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


class Versions(TypedDict):
    edge: PromptFunction
    reflexion: PromptFunction


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
        <PREVIOUS MESSAGES>
        {json.dumps([ep for ep in context['previous_episodes']], indent=2)}
        </PREVIOUS MESSAGES>
        <CURRENT MESSAGE>
        {context['episode_content']}
        </CURRENT MESSAGE>
        
        <CURRENT TIME>
        {context['reference_time']}
        </CURRENT TIME>
        
        <ENTITIES>
        {context['nodes']}
        </ENTITIES>
        
        {context['custom_prompt']}

        Given the above MESSAGES and ENTITIES, extract all facts pertaining to the listed ENTITIES from the CURRENT MESSAGE.
        For each fact, make sure to provide information on all relevant fields, including datetimes like valid_at and invalid_at.
        
        
        Fact Extraction Guidelines:
        1. Extract facts only between the provided entities.
        2. Each fact should represent a clear relationship between two DISTINCT nodes.
        3. The relation_type should be a SCREAMING_SNAKE_CASE predicate of the fact, excluding the object (e.g., LOVES, IS_FRIENDS_WITH, WORKS_FOR).
        4. Provide a more detailed fact containing all relevant information.
        5. Consider temporal aspects of relationships when relevant.

        Datetime Extraction Guidelines:
        1. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ) for datetimes.
        2. Use the CURRENT TIME when determining the valid_at and invalid_at dates.
        3. If the fact is written in the present tense, use the CURRENT TIME for the valid_at date
        4. If no temporal information is found that establishes or changes the relationship, leave the fields as null.
        5. Do not infer dates from related events. Only use dates that are directly stated to establish or change the relationship.
        6. For relative time mentions directly related to the relationship, calculate the actual datetime based on the CURRENT TIME.
        7. If only a date is mentioned without a specific time, use 00:00:00 (midnight) for that date.
        8. If only year is mentioned, use January 1st of that year at 00:00:00.
        9. Always include the time zone offset (use Z for UTC if no specific time zone is mentioned).
        10. A fact discussing that something is no longer true should have a valid_at according to when the negated fact became true.
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


versions: Versions = {'edge': edge, 'reflexion': reflexion}
