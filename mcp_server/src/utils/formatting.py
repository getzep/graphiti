"""Formatting utilities for Graphiti MCP Server."""

from typing import Any

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode


def format_node_result(node: EntityNode) -> dict[str, Any]:
    """Format an entity node into a readable result.

    Since EntityNode is a Pydantic BaseModel, we can use its built-in serialization capabilities.
    Excludes embedding vectors to reduce payload size and avoid exposing internal representations.

    Args:
        node: The EntityNode to format

    Returns:
        A dictionary representation of the node with serialized dates and excluded embeddings
    """
    result = node.model_dump(
        mode='json',
        exclude={
            'name_embedding',
        },
    )
    # Remove any embedding that might be in attributes
    result.get('attributes', {}).pop('name_embedding', None)
    return result


def format_fact_result(edge: EntityEdge) -> dict[str, Any]:
    """Format an entity edge into a readable result.

    Since EntityEdge is a Pydantic BaseModel, we can use its built-in serialization capabilities.

    Args:
        edge: The EntityEdge to format

    Returns:
        A dictionary representation of the edge with serialized dates and excluded embeddings
    """
    result = edge.model_dump(
        mode='json',
        exclude={
            'fact_embedding',
        },
    )
    result.get('attributes', {}).pop('fact_embedding', None)
    return result
