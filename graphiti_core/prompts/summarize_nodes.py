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

from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS

from .models import Message, PromptFunction, PromptVersion
from .prompt_helpers import to_prompt_json
from .snippets import summary_instructions


class Summary(BaseModel):
    summary: str = Field(
        ...,
        description=(
            f'Summary containing the important information about the entity. '
            f'Under {MAX_SUMMARY_CHARS} characters'
        ),
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
            content='You are a helpful assistant that combines summaries into a single dense factual summary.',
        ),
        Message(
            role='user',
            content=f"""
        Synthesize the information from the following two summaries into a single information-dense summary.

        IMPORTANT:
        - Preserve all materially relevant names, roles, places, dates, counts, and changes over time that are explicitly supported.
        - Prefer compact factual sentences over vague thematic phrasing.
        - When the durable fact is the content of what was said, state the content directly instead of narrating that it was said.
        - Use communication verbs only when the act of speaking, asking, sharing, presenting, or announcing is itself the important fact.
        - Avoid filler verbs like "mentioned", "described", "stated", "reported", "noted", "discussed", "referenced", and "indicated" unless the communication act itself matters.
        - SUMMARIES MUST BE LESS THAN {MAX_SUMMARY_CHARS} CHARACTERS.

        Summaries:
        {to_prompt_json(context['node_summaries'])}
        """,
        ),
    ]


def summarize_context(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that generates detailed, information-dense summaries and attributes from provided text.',
        ),
        Message(
            role='user',
            content=f"""
        Given the MESSAGES and the ENTITY name, create a summary for the ENTITY. Your summary must only use
        information from the provided MESSAGES. Your summary should also only contain information relevant to the
        provided ENTITY.

        In addition, extract any values for the provided entity properties based on their descriptions.
        If the value of the entity property cannot be found in the current context, set the value of the property to the Python value None.

        {summary_instructions}

        <MESSAGES>
        {to_prompt_json(context['previous_episodes'])}
        {to_prompt_json(context['episode_content'])}
        </MESSAGES>

        <ENTITY>
        {context['node_name']}
        </ENTITY>

        <ENTITY CONTEXT>
        {context['node_summary']}
        </ENTITY CONTEXT>

        <ATTRIBUTES>
        {to_prompt_json(context['attributes'])}
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
        Summaries must be under {MAX_SUMMARY_CHARS} characters.

        Summary:
        {to_prompt_json(context['summary'])}
        """,
        ),
    ]


versions: Versions = {
    'summarize_pair': summarize_pair,
    'summarize_context': summarize_context,
    'summary_description': summary_description,
}
