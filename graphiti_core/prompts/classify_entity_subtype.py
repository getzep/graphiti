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
from typing_extensions import TypedDict

from .models import Message, PromptFunction, PromptVersion


class EntitySubtypeClassification(BaseModel):
    subtype_index: int = Field(
        ...,
        description='1-based index of the best subtype, or 0 when no subtype fits',
    )


class Prompt(Protocol):
    classify: PromptVersion


class Versions(TypedDict):
    classify: PromptFunction


def classify(context: dict[str, Any]) -> list[Message]:
    episode_text = '\n\n'.join(context['episode_content'])
    type_chain = '\n'.join(
        f'- {type_info["name"]}: {type_info["description"] or "No description"}'
        for type_info in context['chain']
    )
    candidates = '\n'.join(
        f'{candidate["index"]}. {candidate["name"]}: {candidate["description"] or "No description"}'
        for candidate in context['candidates']
    )
    custom_instructions = context.get('custom_extraction_instructions')
    instructions = (
        f'<CUSTOM INSTRUCTIONS>\n{custom_instructions}\n</CUSTOM INSTRUCTIONS>'
        if custom_instructions
        else ''
    )
    return [
        Message(
            role='system',
            content=(
                'Classify one entity into exactly one subtype of its current type, or 0 when '
                'none fits. Output only the integer in the subtype_index JSON field.'
            ),
        ),
        Message(
            role='user',
            content=f"""
<EPISODE TEXT>
{episode_text}
</EPISODE TEXT>
<ENTITY NAME>
{context['entity_name']}
</ENTITY NAME>
<TYPE CHAIN>
{type_chain}
</TYPE CHAIN>
<CANDIDATE SUBTYPES>
{candidates}
</CANDIDATE SUBTYPES>
{instructions}
""",
        ),
    ]


versions: Versions = {'classify': classify}
