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

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Default cap for free-form string attribute values. Calibrated to comfortably fit
# normal short-text fields (phones, industries, URLs, addresses, alias lists) while
# rejecting multi-paragraph LLM meta-reasoning. Customers with legitimately longer
# fields should set an explicit max_length on the Pydantic Field, or override the
# default globally via the GRAPHITI_ATTRIBUTE_MAX_LENGTH env var.
DEFAULT_ATTRIBUTE_MAX_LENGTH = 250

# When a list-typed string attribute is provided, the per-item cap and an aggregate
# multiplier together bound list-total length. The multiplier mirrors common usage
# (≤8 short entries) without being so loose that a 50× repetition slips through.
LIST_TOTAL_LENGTH_MULTIPLIER = 8

_ENV_VAR = 'GRAPHITI_ATTRIBUTE_MAX_LENGTH'

# Track invalid env values we've already warned about so a misconfigured deploy
# emits one warning per unique bad value rather than one per cap invocation.
_warned_invalid_env: set[str] = set()


# Merge semantics shared by node and edge call sites:
#
#   merge_mode='overlay'  (node attributes)
#       prior overlaid by kept fields. LLM-omitted fields retain prior values;
#       cap-dropped fields also retain prior values (because they are absent
#       from kept and overlay never overwrites with absence).
#
#   merge_mode='replace'  (edge attributes)
#       kept fully replaces prior, BUT cap-dropped fields are restored from
#       prior. LLM-omitted fields are cleared (replace semantics).
#
# The asymmetry exists because edge-attribute extraction historically used
# wholesale replacement, while node attributes have always been incrementally
# merged. The unified helper makes the choice explicit at each call site.


def _resolve_default_max_length(default_max_length: int) -> int:
    raw = os.environ.get(_ENV_VAR)
    if raw is None or raw.strip() == '':
        return default_max_length
    try:
        parsed = int(raw)
        if parsed <= 0:
            raise ValueError('non-positive')
    except ValueError:
        if raw not in _warned_invalid_env:
            _warned_invalid_env.add(raw)
            logger.warning(
                'Ignoring invalid %s=%r; expected a positive integer. Using default=%d.',
                _ENV_VAR,
                raw,
                default_max_length,
            )
        return default_max_length
    return parsed


def _field_max_length(model: type[BaseModel], field_name: str) -> int | None:
    field_info = model.model_fields.get(field_name)
    if field_info is None:
        return None
    for meta in getattr(field_info, 'metadata', []) or []:
        explicit = getattr(meta, 'max_length', None)
        if explicit is not None:
            return int(explicit)
    return None


def _field_is_required(model: type[BaseModel], field_name: str) -> bool:
    field_info = model.model_fields.get(field_name)
    if field_info is None:
        return False
    is_required = getattr(field_info, 'is_required', None)
    if callable(is_required):
        return bool(is_required())
    # Older pydantic shape; fall back conservatively.
    return False


def _check_value_against_cap(value: Any, max_len: int) -> tuple[bool, str, int, int]:
    """Decide whether ``value`` exceeds the cap and on which axis.

    Returns ``(exceeded, reason, observed_length, breached_cap)``:
      * ``exceeded``       — True if the field should be dropped.
      * ``reason``         — one of ``'per_item'``, ``'aggregate'``, ``'ok'``.
      * ``observed_length`` — the length to log: the offending element's length for
        per-item triggers, the aggregate string length for aggregate triggers.
      * ``breached_cap``    — the cap that was actually breached; ``max_len`` for
        per-item, ``max_len * LIST_TOTAL_LENGTH_MULTIPLIER`` for aggregate. Logging
        this alongside ``observed_length`` keeps the two directly comparable in
        DataDog instead of confusingly showing ``length=240 cap=250`` when 50
        just-under-cap items collectively breached the aggregate guard.

    Catching both axes prevents a single bleed slipping through inside one element
    AND prevents many "just-under-cap" items adding up to KB-scale list bleed.
    """
    if isinstance(value, str):
        if len(value) > max_len:
            return True, 'per_item', len(value), max_len
        return False, 'ok', 0, max_len
    if isinstance(value, list):
        max_item = max(
            (len(item) for item in value if isinstance(item, str)),
            default=0,
        )
        if max_item > max_len:
            return True, 'per_item', max_item, max_len
        total = sum(len(item) for item in value if isinstance(item, str))
        aggregate_cap = max_len * LIST_TOTAL_LENGTH_MULTIPLIER
        if total > aggregate_cap:
            return True, 'aggregate', total, aggregate_cap
        return False, 'ok', 0, max_len
    return False, 'ok', 0, max_len


def cap_string_attributes(
    response: dict[str, Any],
    model: type[BaseModel],
    *,
    default_max_length: int = DEFAULT_ATTRIBUTE_MAX_LENGTH,
    prompt_name: str = '',
    entity_uuid: str = '',
    group_id: str = '',
) -> tuple[dict[str, Any], set[str]]:
    """Drop string (or list-of-string) attributes whose value exceeds a length cap.

    Defends against meta-thinking / schema-description bleed where the LLM dumps
    multi-paragraph reasoning into a free-form attribute field.

    For string-typed fields the cap is the length of the value. For list-typed
    fields the cap is enforced both per-item and on the aggregate length of all
    string elements (max_len × ``LIST_TOTAL_LENGTH_MULTIPLIER``); see
    ``_check_value_against_cap``.

    Cap precedence: an explicit ``max_length`` on the Pydantic Field wins; otherwise
    the resolved default (``GRAPHITI_ATTRIBUTE_MAX_LENGTH`` env var if set, else
    ``default_max_length``). Non-string, non-string-list fields pass through untouched.

    Returns ``(kept, dropped)``:
      * ``kept`` — the response dict with over-cap fields removed (with one exception,
        below).
      * ``dropped`` — the set of field names that were dropped.

    Required-field exception: if a Pydantic field is REQUIRED (no default and no
    ``Optional``) and the LLM emitted an over-cap value, the value is retained
    (with a warning) rather than dropped. Dropping a required field would cause
    the subsequent ``model(**capped)`` validation in the node path to fail the
    entire response. Customers who want stricter behavior on required fields
    should set an explicit ``max_length`` on the Pydantic Field; Pydantic will
    enforce it at validation time.

    Logging deliberately uses ``entity_uuid`` (not name) per AGENTS.md "no PII in logs".
    """
    effective_default = _resolve_default_max_length(default_max_length)
    kept: dict[str, Any] = {}
    dropped: set[str] = set()
    for field_name, value in response.items():
        max_len = _field_max_length(model, field_name) or effective_default
        exceeded, reason, observed_length, breached_cap = _check_value_against_cap(value, max_len)
        if not exceeded:
            kept[field_name] = value
            continue
        # Required-field carve-out: don't drop, would crash Pydantic validation.
        if _field_is_required(model, field_name):
            logger.warning(
                'attribute_length_cap_skipped_required '
                'prompt=%s group_id=%s entity_uuid=%s field=%s '
                'reason=%s length=%d cap=%d',
                prompt_name,
                group_id or '<unknown>',
                entity_uuid or '<unknown>',
                field_name,
                reason,
                observed_length,
                breached_cap,
            )
            kept[field_name] = value
            continue
        logger.info(
            'attribute_length_cap_exceeded '
            'prompt=%s group_id=%s entity_uuid=%s field=%s '
            'reason=%s length=%d cap=%d',
            prompt_name,
            group_id or '<unknown>',
            entity_uuid or '<unknown>',
            field_name,
            reason,
            observed_length,
            breached_cap,
        )
        dropped.add(field_name)
    return kept, dropped


def apply_capped_attributes(
    response: dict[str, Any],
    model: type[BaseModel],
    prior_attributes: dict[str, Any],
    *,
    merge_mode: Literal['overlay', 'replace'],
    default_max_length: int = DEFAULT_ATTRIBUTE_MAX_LENGTH,
    prompt_name: str = '',
    entity_uuid: str = '',
    group_id: str = '',
) -> tuple[dict[str, Any], set[str]]:
    """Cap the LLM response and merge it with prior attributes per the merge mode.

    See the module-level docstring for the semantics of each ``merge_mode``. The
    return value is ``(merged, dropped)``; the dropped set is exposed for callers
    that want to log or react to it independently.
    """
    kept, dropped = cap_string_attributes(
        response,
        model,
        default_max_length=default_max_length,
        prompt_name=prompt_name,
        entity_uuid=entity_uuid,
        group_id=group_id,
    )
    if merge_mode == 'overlay':
        merged: dict[str, Any] = {**prior_attributes, **kept}
    elif merge_mode == 'replace':
        merged = dict(kept)
        for field in dropped:
            if field in prior_attributes:
                merged[field] = prior_attributes[field]
    else:  # pragma: no cover — Literal protects this at type-check time.
        raise ValueError(f'merge_mode must be "overlay" or "replace", got {merge_mode!r}')
    return merged, dropped
