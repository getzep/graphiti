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
    relation_type: str = Field(..., description='RELATION_TYPE_IN_CAPS')
    source_entity_name: str = Field(..., description='name of the source entity')
    target_entity_name: str = Field(..., description='name of the target entity')
    fact: str = Field(..., description='extracted factual information')


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
            content='You are an expert fact extractor that extracts fact triples from text.',
        ),
        Message(
            role='user',
            content=f"""
        <PREVIOUS MESSAGES>
        {json.dumps([ep for ep in context['previous_episodes']], indent=2)}
        </PREVIOUS MESSAGES>
        <CURRENT MESSAGE>
        {context["episode_content"]}
        </CURRENT MESSAGE>
        
        <ENTITIES>
        {context["nodes"]}
        </ENTITIES>
        
        {context['custom_prompt']}

        Given the above MESSAGES and ENTITIES, extract all facts pertaining to the listed ENTITIES from the CURRENT MESSAGE. 
        
        Guidelines:
        1. Extract facts only between the provided entities.
        2. Each fact should represent a clear relationship between two DISTINCT nodes.
        3. The relation_type should be a concise, all-caps description of the fact (e.g., LOVES, IS_FRIENDS_WITH, WORKS_FOR).
        4. Provide a more detailed fact containing all relevant information.
        5. Consider temporal aspects of relationships when relevant.
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
{context["episode_content"]}
</CURRENT MESSAGE>

<EXTRACTED ENTITIES>
{context["nodes"]}
</EXTRACTED ENTITIES>

<EXTRACTED FACTS>
{context["extracted_facts"]}
</EXTRACTED FACTS>

Given the above MESSAGES, list of EXTRACTED ENTITIES entities, and list of EXTRACTED FACTS; 
determine if any facts haven't been extracted.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


versions: Versions = {'edge': edge, 'reflexion': reflexion}
