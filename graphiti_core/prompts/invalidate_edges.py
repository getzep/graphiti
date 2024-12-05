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


class InvalidatedEdge(BaseModel):
    uuid: str = Field(..., description='The UUID of the edge to be invalidated')
    fact: str = Field(..., description='Updated fact of the edge')


class InvalidatedEdges(BaseModel):
    invalidated_edges: list[InvalidatedEdge] = Field(
        ..., description='List of edges that should be invalidated'
    )


class Prompt(Protocol):
    v1: PromptVersion
    v2: PromptVersion


class Versions(TypedDict):
    v1: PromptFunction
    v2: PromptFunction


def v1(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are an AI assistant that helps determine which relationships in a knowledge graph should be invalidated based solely on explicit contradictions in newer information.',
        ),
        Message(
            role='user',
            content=f"""
               Based on the provided existing edges and new edges with their timestamps, determine which relationships, if any, should be marked as expired due to contradictions or updates in the newer edges.
               Use the start and end dates of the edges to determine which edges are to be marked expired.
                Only mark a relationship as invalid if there is clear evidence from other edges that the relationship is no longer true.
                Do not invalidate relationships merely because they weren't mentioned in the episodes. You may use the current episode and previous episodes as well as the facts of each edge to understand the context of the relationships.

                Previous Episodes:
                {context['previous_episodes']}

                Current Episode:
                {context['current_episode']}

                Existing Edges (sorted by timestamp, newest first):
                {context['existing_edges']}

                New Edges:
                {context['new_edges']}

                Each edge is formatted as: "UUID | SOURCE_NODE - EDGE_NAME - TARGET_NODE (fact: EDGE_FACT), START_DATE (END_DATE, optional))"
            """,
        ),
    ]


def v2(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are an AI assistant that helps determine which relationships in a knowledge graph should be invalidated based solely on explicit contradictions in newer information.',
        ),
        Message(
            role='user',
            content=f"""
               Based on the provided Existing Edges and a New Edge, determine which existing edges, if any, should be marked as invalidated due to invalidations with the New Edge.

                Existing Edges:
                {context['existing_edges']}

                New Edge:
                {context['new_edge']}
            """,
        ),
    ]


versions: Versions = {'v1': v1, 'v2': v2}
