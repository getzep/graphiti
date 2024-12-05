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
from typing import Any, Optional, Protocol, TypedDict

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion


class EdgeDuplicate(BaseModel):
    is_duplicate: bool = Field(..., description='true or false')
    uuid: Optional[str] = Field(
        None,
        description="uuid of the existing edge like '5d643020624c42fa9de13f97b1b3fa39' or null",
    )


class UniqueFact(BaseModel):
    uuid: str = Field(..., description='unique identifier of the fact')
    fact: str = Field(..., description='fact of a unique edge')


class UniqueFacts(BaseModel):
    unique_facts: list[UniqueFact]


class Prompt(Protocol):
    edge: PromptVersion
    edge_list: PromptVersion


class Versions(TypedDict):
    edge: PromptFunction
    edge_list: PromptFunction


def edge(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that de-duplicates edges from edge lists.',
        ),
        Message(
            role='user',
            content=f"""
        Given the following context, determine whether the New Edge represents any of the edges in the list of Existing Edges.

        <EXISTING EDGES>
        {json.dumps(context['related_edges'], indent=2)}
        </EXISTING EDGES>

        <NEW EDGE>
        {json.dumps(context['extracted_edges'], indent=2)}
        </NEW EDGE>
        
        Task:
        1. If the New Edges represents the same factual information as any edge in Existing Edges, return 'is_duplicate: true' in the 
            response. Otherwise, return 'is_duplicate: false'
        2. If is_duplicate is true, also return the uuid of the existing edge in the response

        Guidelines:
        1. The facts do not need to be completely identical to be duplicates, they just need to express the same information.
        """,
        ),
    ]


def edge_list(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that de-duplicates edges from edge lists.',
        ),
        Message(
            role='user',
            content=f"""
        Given the following context, find all of the duplicates in a list of facts:

        Facts:
        {json.dumps(context['edges'], indent=2)}

        Task:
        If any facts in Facts is a duplicate of another fact, return a new fact with one of their uuid's.

        Guidelines:
        1. identical or near identical facts are duplicates
        2. Facts are also duplicates if they are represented by similar sentences
        3. Facts will often discuss the same or similar relation between identical entities
        4. The final list should have only unique facts. If 3 facts are all duplicates of each other, only one of their
            facts should be in the response
        """,
        ),
    ]


versions: Versions = {'edge': edge, 'edge_list': edge_list}
