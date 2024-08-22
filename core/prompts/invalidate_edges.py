from typing import Protocol, TypedDict

from .models import Message, PromptFunction, PromptVersion


class Prompt(Protocol):
	v1: PromptVersion


class Versions(TypedDict):
	v1: PromptFunction


def v1(context: dict[str, any]) -> list[Message]:
	return [
		Message(
			role='system',
			content='You are an AI assistant that helps determine which relationships in a knowledge graph should be invalidated based solely on explicit contradictions in newer information.',
		),
		Message(
			role='user',
			content=f"""
               Based on the provided existing edges and new edges with their timestamps, determine which existing relationships, if any, should be invalidated due to contradictions or updates in the new edges.
                Only mark a relationship as invalid if there is clear evidence from new edges that the relationship is no longer true.
                Do not invalidate relationships merely because they weren't mentioned in new edges. You may use the current episode and previous episodes as well as the facts of each edge to understand the context of the relationships.

                Previous Episodes:
                {context['previous_episodes']}

                Current Episode:
                {context['current_episode']}

                Existing Edges (sorted by timestamp, newest first):
                {context['existing_edges']}

                New Edges:
                {context['new_edges']}

                Each edge is formatted as: "UUID | SOURCE_NODE - EDGE_NAME - TARGET_NODE (fact: EDGE_FACT), TIMESTAMP)"

                For each existing edge that should be invalidated, respond with a JSON object in the following format:
                {{
                    "invalidated_edges": [
                        {{
                            "edge_uuid": "The UUID of the edge to be invalidated (the part before the | character)",
                            "fact": "Updated fact of the edge"
                        }}
                    ]
                }}

                If no relationships need to be invalidated based on these strict criteria, return an empty list for "invalidated_edges".
            """,
		),
	]


versions: Versions = {'v1': v1}
