from typing import Protocol, TypedDict
from .models import Message, PromptVersion, PromptFunction


class Prompt(Protocol):
    v1: PromptVersion


class Versions(TypedDict):
    v1: PromptFunction


def v1(context: dict[str, any]) -> list[Message]:
    return [
        Message(
            role="system",
            content="You are an AI assistant that helps determine which relationships in a knowledge graph should be invalidated based solely on explicit contradictions in newer information.",
        ),
        Message(
            role="user",
            content=f"""
                Based on the provided existing edges and new edges with their timestamps, determine which existing relationships, if any, should be invalidated due to explicit contradictions in the new edges.
                
                Important guidelines:
                1. Only mark a relationship as invalid if there is an explicit, direct contradiction in the new edges.
                2. Do not make any assumptions or inferences about relationships.
                3. Do not invalidate edges based on implied changes or personal interpretations.
                4. A new edge does not automatically invalidate an existing edge unless it directly states the opposite.
                5. Different types of relationships can coexist and do not automatically invalidate each other.
                6. Do not invalidate relationships merely because they weren't mentioned in new edges.

                Existing Edges (sorted by timestamp, newest first):
                {context['existing_edges']}

                New Edges:
                {context['new_edges']}

                Each edge is formatted as: "UUID | SOURCE_NODE - EDGE_NAME - TARGET_NODE (TIMESTAMP)"

                For each existing edge that should be invalidated, respond with a JSON object in the following format:
                {{
                    "invalidated_edges": [
                        {{
                            "edge_uuid": "The UUID of the edge to be invalidated (the part before the | character)",
                            "reason": "Brief explanation citing the specific new edge that directly contradicts this edge"
                        }}
                    ]
                }}

                If no relationships need to be invalidated based on these strict criteria, return an empty list for "invalidated_edges".
            """,
        ),
    ]


versions: Versions = {"v1": v1}
