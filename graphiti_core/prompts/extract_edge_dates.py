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


class EdgeDates(BaseModel):
    valid_at: str | None = Field(
        None,
        description='The date and time when the relationship described by the edge fact became true or was established. YYYY-MM-DDTHH:MM:SS.SSSSSSZ or null.',
    )
    invalid_at: str | None = Field(
        None,
        description='The date and time when the relationship described by the edge fact stopped being true or ended. YYYY-MM-DDTHH:MM:SS.SSSSSSZ or null.',
    )


class Prompt(Protocol):
    v1: PromptVersion


class Versions(TypedDict):
    v1: PromptFunction


def v1(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are an AI assistant that extracts datetime information for graph edges, focusing only on dates directly related to the establishment or change of the relationship described in the edge fact.',
        ),
        Message(
            role='user',
            content=f"""
            <PREVIOUS MESSAGES>
            {context['previous_episodes']}
            </PREVIOUS MESSAGES>
            <CURRENT MESSAGE>
            {context['current_episode']}
            </CURRENT MESSAGE>
            <REFERENCE TIMESTAMP>
            {context['reference_timestamp']}
            </REFERENCE TIMESTAMP>
            
            <FACT>
            {context['edge_fact']}
            </FACT>

            IMPORTANT: Only extract time information if it is part of the provided fact. Otherwise ignore the time mentioned. Make sure to do your best to determine the dates if only the relative time is mentioned. (eg 10 years ago, 2 mins ago) based on the provided reference timestamp
            If the relationship is not of spanning nature, but you are still able to determine the dates, set the valid_at only.
            Definitions:
            - valid_at: The date and time when the relationship described by the edge fact became true or was established.
            - invalid_at: The date and time when the relationship described by the edge fact stopped being true or ended.

            Task:
            Analyze the conversation and determine if there are dates that are part of the edge fact. Only set dates if they explicitly relate to the formation or alteration of the relationship itself.

            Guidelines:
            1. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ) for datetimes.
            2. Use the reference timestamp as the current time when determining the valid_at and invalid_at dates.
            3. If the fact is written in the present tense, use the Reference Timestamp for the valid_at date
            4. If no temporal information is found that establishes or changes the relationship, leave the fields as null.
            5. Do not infer dates from related events. Only use dates that are directly stated to establish or change the relationship.
			6. For relative time mentions directly related to the relationship, calculate the actual datetime based on the reference timestamp.
            7. If only a date is mentioned without a specific time, use 00:00:00 (midnight) for that date.
            8. If only year is mentioned, use January 1st of that year at 00:00:00.
            9. Always include the time zone offset (use Z for UTC if no specific time zone is mentioned).
            10. A fact discussing that something is no longer true should have a valid_at according to when the negated fact became true.
            """,
        ),
    ]


versions: Versions = {'v1': v1}
