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


class Summary(BaseModel):
    summary: str = Field(
        ...,
        description='Summary containing the important information about the entity. Under 250 words',
    )


class SummaryDescription(BaseModel):
    description: str = Field(..., description='One sentence description of the provided summary')


class Prompt(Protocol):
    summarize_pair: PromptVersion
    summarize_context: PromptVersion
    summary_description: PromptVersion


class Versions(TypedDict):
    summarize_pair: PromptFunction
    summarize_context: PromptFunction
    summary_description: PromptFunction


def summarize_pair(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that combines summaries.',
        ),
        Message(
            role='user',
            content=f"""
        Synthesize the information from the following two summaries into a single succinct summary.
        
        Summaries must be under 250 words.

        Summaries:
        {json.dumps(context['node_summaries'], indent=2)}
        """,
        ),
    ]


def summarize_context(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts entity properties from the provided text.',
        ),
        Message(
            role='user',
            content=f"""
            
        <MESSAGES>
        {json.dumps(context['previous_episodes'], indent=2)}
        {json.dumps(context['episode_content'], indent=2)}
        </MESSAGES>
        
        Given the above MESSAGES and the following ENTITY name, create a summary for the ENTITY. Your summary must only use
        information from the provided MESSAGES. Your summary should also only contain information relevant to the
        provided ENTITY. Summaries must be under 250 words.
        
        In addition, extract any values for the provided entity properties based on their descriptions.
        If the value of the entity property cannot be found in the current context, set the value of the property to the Python value None.
        
        Guidelines:
        1. Do not hallucinate entity property values if they cannot be found in the current context.
        2. Only use the provided messages, entity, and entity context to set attribute values.
        
        <ENTITY>
        {context['node_name']}
        </ENTITY>
        
        <ENTITY CONTEXT>
        {context['node_summary']}
        </ENTITY CONTEXT>
        
        <ATTRIBUTES>
        {json.dumps(context['attributes'], indent=2)}
        </ATTRIBUTES>
        """,
        ),
    ]


def summary_description(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that describes provided contents in a single sentence.',
        ),
        Message(
            role='user',
            content=f"""
        Create a short one sentence description of the summary that explains what kind of information is summarized.
        Summaries must be under 250 words.

        Summary:
        {json.dumps(context['summary'], indent=2)}
        """,
        ),
    ]


versions: Versions = {
    'summarize_pair': summarize_pair,
    'summarize_context': summarize_context,
    'summary_description': summary_description,
}
