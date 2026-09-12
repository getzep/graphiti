"""Translate a Zep-style ontology into Graphiti's Pydantic type maps.

Zep Cloud stores the ontology server-side and applies it to every ingest.
Graphiti instead wants `entity_types` / `edge_types` / `edge_type_map` passed
into each `add_episode` call, as Pydantic model classes. This module builds
those classes from the JSON MiroFish sent to `PUT entity-types`.
"""

from __future__ import annotations

import keyword
import re
from typing import Any

from pydantic import BaseModel, Field, create_model

_TYPE_MAP: dict[str, type] = {
    'Text': str,
    'Int': int,
    'Float': float,
    'Boolean': bool,
}

# Graphiti raises EntityTypeValidationError when a custom attribute shadows a
# field on its own EntityNode. Union of Node + EntityNode annotated fields in
# graphiti_core/nodes.py. MiroFish's own reserved list misses `labels` and
# `attributes`, so we re-check here rather than trusting the caller.
_GRAPHITI_RESERVED = frozenset(
    {
        'uuid',
        'name',
        'group_id',
        'labels',
        'created_at',
        'name_embedding',
        'summary',
        'attributes',
    }
)

_IDENT_RE = re.compile(r'[^0-9a-zA-Z_]')


def safe_field_name(raw: str) -> str:
    """Coerce an arbitrary attribute name into a legal, non-colliding field."""
    name = _IDENT_RE.sub('_', (raw or '').strip()).strip('_')
    if not name:
        name = 'value'
    if name[0].isdigit():
        name = f'attr_{name}'
    if name in _GRAPHITI_RESERVED or keyword.iskeyword(name):
        name = f'attr_{name}'
    return name


def _safe_model_name(raw: str, fallback: str) -> str:
    name = _IDENT_RE.sub('', (raw or '').strip())
    if not name or name[0].isdigit():
        name = fallback
    return name


def _build_model(
    type_name: str, description: str, properties: list[dict[str, Any]] | None
) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    used: set[str] = set()
    for prop in properties or []:
        if not isinstance(prop, dict):
            continue
        field_name = safe_field_name(str(prop.get('name', '')))
        # Two distinct source names can normalize to the same field.
        candidate, suffix = field_name, 2
        while candidate in used:
            candidate = f'{field_name}_{suffix}'
            suffix += 1
        used.add(candidate)
        py_type = _TYPE_MAP.get(str(prop.get('type') or 'Text'), str)
        fields[candidate] = (
            py_type | None,
            Field(default=None, description=str(prop.get('description') or candidate)),
        )

    model = create_model(  # type: ignore[call-overload]
        _safe_model_name(type_name, 'OntologyType'),
        __base__=BaseModel,
        **fields,
    )
    # Graphiti feeds the class docstring to the extraction prompt as the type's
    # definition, so this is load-bearing, not cosmetic.
    model.__doc__ = description or f'A {type_name}.'
    return model


def build_entity_types(entity_types: list[dict[str, Any]]) -> dict[str, type[BaseModel]]:
    built: dict[str, type[BaseModel]] = {}
    for spec in entity_types or []:
        name = str(spec.get('name') or '').strip()
        if not name or name == 'Entity':
            # 'Entity' is Graphiti's built-in default label; redefining it is
            # rejected by excluded/registered-type handling upstream.
            continue
        built[name] = _build_model(
            name, str(spec.get('description') or ''), spec.get('properties')
        )
    return built


def build_edge_types(
    edge_types: list[dict[str, Any]],
) -> tuple[dict[str, type[BaseModel]], dict[tuple[str, str], list[str]]]:
    built: dict[str, type[BaseModel]] = {}
    type_map: dict[tuple[str, str], list[str]] = {}
    for spec in edge_types or []:
        name = str(spec.get('name') or '').strip()
        if not name:
            continue
        built[name] = _build_model(
            name, str(spec.get('description') or ''), spec.get('properties')
        )
        for pair in spec.get('source_targets') or []:
            if not isinstance(pair, dict):
                continue
            source = str(pair.get('source') or 'Entity').strip() or 'Entity'
            target = str(pair.get('target') or 'Entity').strip() or 'Entity'
            type_map.setdefault((source, target), []).append(name)
    return built, type_map
