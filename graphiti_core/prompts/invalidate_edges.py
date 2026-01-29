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


class InvalidatedEdges(BaseModel):
    contradicted_facts: list[int] = Field(
        ...,
        description='List of ids of facts that should be invalidated. If no facts should be invalidated, the list should be empty.',
    )


class Prompt(Protocol):
    v2: PromptVersion


class Versions(TypedDict):
    v2: PromptFunction


def v2(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are an AI assistant that determines which facts contradict each other.',
        ),
        Message(
            role='user',
            content=f"""
               Based on the provided EXISTING FACTS and a NEW FACT, determine which existing facts the new fact contradicts.
               Return a list containing all ids of the facts that are contradicted by the NEW FACT.
               If there are no contradicted facts, return an empty list.

                <EXISTING FACTS>
                {context['existing_edges']}
                </EXISTING FACTS>

                <NEW FACT>
                {context['new_edge']}
                </NEW FACT>
            """,
        ),
    ]


versions: Versions = {'v2': v2}
