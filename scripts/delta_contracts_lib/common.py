from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

COMMAND_PART_RE = re.compile(r'^[a-z0-9][a-z0-9-]*$')


def expect_dict(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f'{context} must be an object')
    return value


def expect_str(value: object, *, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f'{context} must be a string')
    return value


def expect_non_empty_str(value: object, *, context: str) -> str:
    text = expect_str(value, context=context).strip()
    if not text:
        raise ValueError(f'{context} must be a non-empty string')
    return text


def expect_bool(value: object, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f'{context} must be a boolean')
    return value


def expect_number(value: object, *, context: str, min_value: float | None = None) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f'{context} must be a number')

    number = float(value)
    if min_value is not None and number < min_value:
        raise ValueError(f'{context} must be >= {min_value}')
    return number


def expect_int(value: object, *, context: str, min_value: int | None = None) -> int:
    if not isinstance(value, int):
        raise ValueError(f'{context} must be an integer')

    if min_value is not None and value < min_value:
        raise ValueError(f'{context} must be >= {min_value}')
    return value


def expect_string_list(
    value: object,
    *,
    context: str,
    allow_empty: bool = True,
    unique: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f'{context} must be a list of strings')

    parsed: list[str] = []
    for index, item in enumerate(value):
        parsed.append(expect_non_empty_str(item, context=f'{context}[{index}]'))

    if not allow_empty and not parsed:
        raise ValueError(f'{context} must not be empty')

    if unique and len(set(parsed)) != len(parsed):
        raise ValueError(f'{context} must not contain duplicates')

    return parsed


def validate_glob_patterns(patterns: Iterable[str], *, context: str) -> list[str]:
    validated: list[str] = []
    for index, pattern in enumerate(patterns):
        stripped = expect_non_empty_str(pattern, context=f'{context}[{index}]')
        if stripped.startswith('/'):
            raise ValueError(f'{context}[{index}] must be relative (no absolute paths)')
        if '..' in stripped.split('/'):
            raise ValueError(f'{context}[{index}] must not contain path traversal (`..`)')
        validated.append(stripped)
    return validated


def validate_command_part(value: object, *, context: str) -> str:
    normalized = expect_non_empty_str(value, context=context)
    if not COMMAND_PART_RE.fullmatch(normalized):
        raise ValueError(
            f'{context} must match pattern `{COMMAND_PART_RE.pattern}`',
        )
    return normalized


def normalize_slug(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r'[^a-z0-9]+', '-', lowered)
    slug = re.sub(r'-+', '-', slug).strip('-')
    return slug
