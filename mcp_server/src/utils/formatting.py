"""Formatting utilities for Graphiti MCP Server."""

import json
from datetime import date, datetime, time, timedelta
from typing import Any

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode

from models.response_types import EdgeResult, NodeResult


def _json_safe(value: Any) -> Any:
    """Recursively convert driver-specific values into JSON-serializable ones.

    Graph drivers return temporal properties as their own types rather than
    Python's built-ins -- ``neo4j.time.DateTime``, for instance, is not a
    ``datetime`` subclass. Those values reach us inside ``attributes``, an
    untyped ``dict[str, Any]`` that Pydantic's JSON mode passes through
    unconverted, so serializing the response later raises.

    Detection is duck-typed rather than isinstance-based on purpose: this
    package depends on ``graphiti-core[falkordb]`` and does not require the
    neo4j driver, so this module must not import any driver package.
    """
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)

    # Driver temporal types (neo4j.time.Date/Time/DateTime and equivalents)
    # expose to_native(); Duration-like types do not and fall through to str().
    to_native = getattr(value, 'to_native', None)
    if callable(to_native):
        try:
            native = to_native()
        except Exception:
            return str(value)
        native_isoformat = getattr(native, 'isoformat', None)
        return native_isoformat() if callable(native_isoformat) else str(native)

    isoformat = getattr(value, 'isoformat', None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            return str(value)

    # Anything still unencodable (driver Duration types, for example) degrades
    # to its string form rather than breaking the whole response.
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


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
        attributes=_json_safe(attrs),
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

    Typed fields are serialized by Pydantic's JSON mode. ``attributes`` is
    excluded from that pass and sanitized separately, because it is an untyped
    bag that can hold driver temporal types Pydantic cannot serialize.

    Args:
        node: The EntityNode to format

    Returns:
        A dictionary representation of the node with serialized dates and excluded embeddings
    """
    result = node.model_dump(
        mode='json',
        exclude={
            'name_embedding',
            'attributes',
        },
    )
    attributes = _json_safe(node.attributes or {})
    attributes.pop('name_embedding', None)
    result['attributes'] = attributes
    return result


def format_fact_result(edge: EntityEdge) -> dict[str, Any]:
    """Format an entity edge into a readable result.

    Typed fields are serialized by Pydantic's JSON mode. ``attributes`` is
    excluded from that pass and sanitized separately, because it is an untyped
    bag that can hold driver temporal types Pydantic cannot serialize. This
    shows up in practice when the installed ``graphiti-core`` predates a field
    the driver returns: the value stays in ``attributes`` as a raw driver type
    instead of being parsed onto the model.

    Args:
        edge: The EntityEdge to format

    Returns:
        A dictionary representation of the edge with serialized dates and excluded embeddings
    """
    result = edge.model_dump(
        mode='json',
        exclude={
            'fact_embedding',
            'attributes',
        },
    )
    attributes = _json_safe(edge.attributes or {})
    attributes.pop('fact_embedding', None)
    result['attributes'] = attributes
    return result
