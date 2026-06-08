"""Helpers for translating MCP configuration into graphiti-core arguments.

These functions are intentionally free of any I/O or global state so they can be
unit-tested without a live database or LLM:

- ``parse_reference_time`` coerces an ISO-8601 string into a timezone-aware UTC
  ``datetime``.
- ``build_entity_types`` / ``build_edge_types`` turn the configured type list into
  the ``dict[str, type[BaseModel]]`` shape expected by ``Graphiti.add_episode``,
  preferring rich registered models and falling back to documentation-only models.
- ``build_edge_type_map`` turns the configured map entries into the
  ``dict[tuple[str, str], list[str]]`` shape expected by ``Graphiti.add_episode``.
"""

from datetime import datetime, timezone

from graphiti_core.search.search_filters import (
    ComparisonOperator,
    DateFilter,
    SearchFilters,
)
from pydantic import BaseModel, create_model

from config.schema import EdgeTypeConfig, EdgeTypeMapEntry, EntityTypeConfig
from models.edge_types import EDGE_TYPES
from models.entity_types import ENTITY_TYPES


def parse_reference_time(value: str | None) -> datetime | None:
    """Parse an ISO-8601 string into a timezone-aware UTC datetime.

    Accepts a trailing ``Z`` (Zulu/UTC) suffix. Timezone-naive values are
    assumed to be UTC. Returns ``None`` when ``value`` is ``None``.

    Raises ``ValueError`` on an unparseable string.
    """
    if value is None:
        return None

    normalized = value.strip()
    if normalized.endswith('Z') or normalized.endswith('z'):
        normalized = normalized[:-1] + '+00:00'

    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def coerce_group_ids(group_ids: str | list[str] | None) -> list[str] | None:
    """Normalize a ``group_ids`` argument that may be given as a scalar string.

    MCP tools accept either a single group_id string or a list of them, while
    graphiti-core expects ``list[str] | None``. A non-empty string becomes a
    one-element list; a blank string is treated as omitted (``None``) so it falls
    back to the configured default group rather than operating on group ``''``
    (which matters for the destructive ``clear_graph``). Lists and ``None`` pass
    through unchanged.
    """
    if isinstance(group_ids, str):
        return [group_ids] if group_ids else None
    return group_ids


def _doc_only_model(name: str, description: str) -> type[BaseModel]:
    """Build a documentation-only Pydantic model with no fields.

    The description becomes the model docstring, which graphiti-core surfaces to
    the LLM during extraction. Used as a fallback when no rich model is
    registered for ``name``.
    """
    model = create_model(name)
    model.__doc__ = description
    return model


def build_entity_types(
    entity_type_configs: list[EntityTypeConfig] | None,
) -> dict[str, type[BaseModel]] | None:
    """Build the ``entity_types`` mapping for ``Graphiti.add_episode``.

    For each configured entry, prefer the rich model registered in
    ``ENTITY_TYPES`` (matched by name). Fall back to a documentation-only model
    built from the configured description. Returns ``None`` when no entity types
    are configured, preserving graphiti-core's default extraction behavior.
    """
    if not entity_type_configs:
        return None

    result: dict[str, type[BaseModel]] = {}
    for cfg in entity_type_configs:
        registered = ENTITY_TYPES.get(cfg.name)
        result[cfg.name] = (
            registered if registered is not None else _doc_only_model(cfg.name, cfg.description)
        )
    return result


def build_edge_types(
    edge_type_configs: list[EdgeTypeConfig] | None,
) -> dict[str, type[BaseModel]] | None:
    """Build the ``edge_types`` mapping for ``Graphiti.add_episode``.

    Mirrors :func:`build_entity_types`, preferring models registered in
    ``EDGE_TYPES``. Returns ``None`` when no edge types are configured.
    """
    if not edge_type_configs:
        return None

    result: dict[str, type[BaseModel]] = {}
    for cfg in edge_type_configs:
        registered = EDGE_TYPES.get(cfg.name)
        result[cfg.name] = (
            registered if registered is not None else _doc_only_model(cfg.name, cfg.description)
        )
    return result


def build_edge_type_map(
    edge_type_map_entries: list[EdgeTypeMapEntry] | None,
) -> dict[tuple[str, str], list[str]] | None:
    """Build the ``edge_type_map`` for ``Graphiti.add_episode``.

    Translates the list-of-entries config shape into the
    ``{(source, target): [edge_type, ...]}`` mapping graphiti-core expects.
    Returns ``None`` when no entries are configured.
    """
    if not edge_type_map_entries:
        return None

    result: dict[tuple[str, str], list[str]] = {}
    for entry in edge_type_map_entries:
        result[(entry.source, entry.target)] = list(entry.edge_types)
    return result


def _date_range_or_group(
    after: datetime | None, before: datetime | None
) -> list[list[DateFilter]] | None:
    """Build a SearchFilters date filter for an optional ``[after, before]`` range.

    graphiti-core expresses date filters as ``list[list[DateFilter]]`` where the
    outer list is OR-ed and each inner list is AND-ed. A single range becomes one
    inner (AND) group with a ``>=`` and/or ``<=`` condition. Returns ``None`` when
    neither bound is supplied.
    """
    conditions: list[DateFilter] = []
    if after is not None:
        conditions.append(
            DateFilter(date=after, comparison_operator=ComparisonOperator.greater_than_equal)
        )
    if before is not None:
        conditions.append(
            DateFilter(date=before, comparison_operator=ComparisonOperator.less_than_equal)
        )
    if not conditions:
        return None
    return [conditions]


def build_fact_search_filters(
    edge_types: list[str] | None = None,
    valid_at_after: str | None = None,
    valid_at_before: str | None = None,
    invalid_at_after: str | None = None,
    invalid_at_before: str | None = None,
) -> SearchFilters | None:
    """Build a ``SearchFilters`` for fact (edge) search.

    Returns ``None`` when no filter criteria are provided so callers can pass
    ``search_filter=None``.

    The ``*_after`` / ``*_before`` arguments are ISO-8601 strings parsed via
    :func:`parse_reference_time` (UTC-coerced). Raises ``ValueError`` on a bad
    timestamp.
    """
    valid_after = parse_reference_time(valid_at_after)
    valid_before = parse_reference_time(valid_at_before)
    invalid_after = parse_reference_time(invalid_at_after)
    invalid_before = parse_reference_time(invalid_at_before)

    valid_at = _date_range_or_group(valid_after, valid_before)
    invalid_at = _date_range_or_group(invalid_after, invalid_before)

    if not edge_types and valid_at is None and invalid_at is None:
        return None

    return SearchFilters(
        edge_types=edge_types or None,
        valid_at=valid_at,
        invalid_at=invalid_at,
    )
