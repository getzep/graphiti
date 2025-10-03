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

from typing import Any, Protocol

from pydantic import BaseModel, Field

from .models import Message, PromptFunction


class DiffSummary(BaseModel):
    summary: str = Field(..., description='Semantic summary of document changes')


class Prompt(Protocol):
    summarize_diff: 'PromptFunction'


class Versions:
    summarize_diff: PromptFunction


def summarize_diff(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that summarizes document changes from unified diffs.
    Your task is to produce clear, semantic descriptions of what changed in a document."""

    user_prompt = f"""
<DOCUMENT URI>
{context['document_uri']}
</DOCUMENT URI>

<UNIFIED DIFF>
{context['diff_content']}
</UNIFIED DIFF>

Analyze the unified diff above and produce a semantic summary of what changed.

Guidelines:
1. Start the summary with the document name (e.g., "In api_design.md, ...")
2. Describe what was added, removed, or modified
3. Include document structure context (section names, headings) when helpful

Example output:
"In test_sync.md, added Feature 7 (zero-context diff isolation) to the feature list under the Version 1 section."
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


versions: Versions = {
    'summarize_diff': summarize_diff,
}
