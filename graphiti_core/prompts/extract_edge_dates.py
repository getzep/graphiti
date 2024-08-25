from typing import Any, Protocol, TypedDict

from .models import Message, PromptFunction, PromptVersion


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
            Edge:
            Edge Name: {context['edge_name']}
            Fact: {context['edge_fact']}

            Current Episode: {context['current_episode']}
            Previous Episodes: {context['previous_episodes']}
            Reference Timestamp: {context['reference_timestamp']}

            IMPORTANT: Only extract time information if it is part of the provided fact. Otherwise ignore the time mentioned. Make sure to do your best to determine the dates if only the relative time is mentioned. (eg 10 years ago, 2 mins ago) based on the provided reference timestamp
            If the relationship is not of spanning nature, but you are still able to determine the dates, set the valid_at only.
            Definitions:
            - valid_at: The date and time when the relationship described by the edge fact became true or was established.
            - invalid_at: The date and time when the relationship described by the edge fact stopped being true or ended.

            Task:
            Analyze the conversation and determine if there are dates that are part of the edge fact. Only set dates if they explicitly relate to the formation or alteration of the relationship itself.

            Guidelines:
            1. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ) for datetimes.
            2. Use the reference timestamp as the current time when determining the valid_at and invalid_at dates.
            3. If no temporal information is found that establishes or changes the relationship, leave the fields as null.
            4. Do not infer dates from related events. Only use dates that are directly stated to establish or change the relationship.
			5. For relative time mentions directly related to the relationship, calculate the actual datetime based on the reference timestamp.
            6. If only a date is mentioned without a specific time, use 00:00:00 (midnight) for that date.
            7. If only a year is mentioned, use January 1st of that year at 00:00:00.
            9. Always include the time zone offset (use Z for UTC if no specific time zone is mentioned).
            Respond with a JSON object:
            {{
                "valid_at": "YYYY-MM-DDTHH:MM:SSZ or null",
                "invalid_at": "YYYY-MM-DDTHH:MM:SSZ or null",
                "explanation": "Brief explanation of why these dates were chosen or why they were set to null"
            }}
            """,
        ),
    ]


versions: Versions = {'v1': v1}
