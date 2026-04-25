"""Formatting utilities for Graphiti MCP Server."""

from datetime import datetime
from typing import Any

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode

try:
    # neo4j.time.DateTime / Date / Time are returned untyped by the driver
    # for some load paths (notably the fact/edge search). Pydantic's
    # `model_dump(mode='json')` doesn't know how to serialize them, which
    # surfaces in MCP as: "Unable to serialize unknown type:
    # <class 'neo4j.time.DateTime'>".
    from neo4j.time import Date as _Neo4jDate
    from neo4j.time import DateTime as _Neo4jDateTime
except ImportError:  # neo4j isn't always available in non-bolt deployments
    _Neo4jDate = None  # type: ignore
    _Neo4jDateTime = None  # type: ignore


def _to_native_dt(v: Any) -> Any:
    """Convert a neo4j.time.{DateTime,Date} to a native datetime; pass
    through anything else."""
    if _Neo4jDateTime is not None and isinstance(v, _Neo4jDateTime):
        return v.to_native()
    if _Neo4jDate is not None and isinstance(v, _Neo4jDate):
        native = v.to_native()
        return datetime(native.year, native.month, native.day)
    return v


def _coerce_edge_datetimes(edge: EntityEdge) -> None:
    """In-place: convert any neo4j.time.{DateTime,Date} on an edge to
    native datetime / date so Pydantic can serialize them. Covers all
    typed temporal fields on EntityEdge (incl. parent Edge.created_at
    and reference_time) plus any datetimes nested in the free-form
    `attributes` dict."""
    if _Neo4jDateTime is None:
        return
    for field in ('created_at', 'valid_at', 'invalid_at',
                  'expired_at', 'reference_time'):
        v = getattr(edge, field, None)
        if v is None:
            continue
        coerced = _to_native_dt(v)
        if coerced is not v:
            setattr(edge, field, coerced)
    attrs = getattr(edge, 'attributes', None)
    if isinstance(attrs, dict):
        for k, v in list(attrs.items()):
            attrs[k] = _to_native_dt(v)


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
    _coerce_edge_datetimes(edge)
    result = edge.model_dump(
        mode='json',
        exclude={
            'fact_embedding',
        },
    )
    result.get('attributes', {}).pop('fact_embedding', None)
    return result
