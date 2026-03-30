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


class SagaSummary(BaseModel):
    summary: str = Field(
        ...,
        description=(
            f'A concise summary of the conversation thread capturing the key topics, '
            f'decisions, and outcomes. Under {MAX_SUMMARY_CHARS} characters.'
        ),
    )


class Prompt(Protocol):
    summarize_saga: PromptVersion


class Versions(TypedDict):
    summarize_saga: PromptFunction


def summarize_saga(context: dict[str, Any]) -> list[Message]:
    saga_name = context.get('saga_name', 'Unknown')
    existing_summary = context.get('existing_summary', '')
    episodes = context.get('episodes', [])

    episodes_text = '\n---\n'.join(episodes) if episodes else '(no messages)'

    existing_summary_section = ''
    if existing_summary:
        existing_summary_section = f"""
Previous summary of this conversation thread:
{existing_summary}

The following messages may include new content since the previous summary. Update the summary to incorporate any new information.
"""

    return [
        Message(
            role='system',
            content=(
                'You are a helpful assistant that summarizes conversation threads. '
                'Produce a single dense factual summary of the conversation. '
                f'Keep the summary under {MAX_SUMMARY_CHARS} characters. '
                'State facts directly. Do not use filler verbs like "mentioned", "discussed", '
                '"noted", or "stated". Preserve names, dates, decisions, and outcomes. '
                'Begin with the main topic or outcome, not with "This conversation" or "The thread".'
            ),
        ),
        Message(
            role='user',
            content=f"""Summarize the following conversation thread "{saga_name}":
{existing_summary_section}
Messages:
{episodes_text}""",
        ),
    ]


versions: Versions = {
    'summarize_saga': summarize_saga,
}
