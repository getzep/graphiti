"""Formatting utilities for Graphiti MCP Server."""

from typing import Any

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode

from models.response_types import EdgeResult, NodeResult


def to_node_result(node: EntityNode) -> NodeResult:
    """Build a NodeResult TypedDict from an EntityNode, dropping embeddings."""
    attrs = node.attributes if node.attributes else {}
    attrs = {k: v for k, v in attrs.items() if 'embedding' not in k.lower()}
    return NodeResult(
        uuid=node.uuid,
        name=node.name,
        labels=node.labels if node.labels else [],
        created_at=node.created_at.isoformat() if node.created_at else None,
        summary=node.summary,
        group_id=node.group_id,
        attributes=attrs,
    )


def to_edge_result(edge: EntityEdge) -> EdgeResult:
    """Build an EdgeResult TypedDict from an EntityEdge."""
    return EdgeResult(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        source_node_uuid=edge.source_node_uuid,
        target_node_uuid=edge.target_node_uuid,
        group_id=edge.group_id,
        created_at=edge.created_at.isoformat() if edge.created_at else None,
        valid_at=edge.valid_at.isoformat() if edge.valid_at else None,
        invalid_at=edge.invalid_at.isoformat() if edge.invalid_at else None,
    )


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
