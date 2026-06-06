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

import base64
import json
from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

DO_NOT_ESCAPE_UNICODE = '\nDo not escape unicode characters.\n'


def _prompt_json_default(value: Any) -> Any:
    """Serialize driver/model values that commonly appear in prompt contexts."""

    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes | bytearray | memoryview):
        return base64.b64encode(bytes(value)).decode('ascii')
    if isinstance(value, set | frozenset):
        return sorted(value, key=repr)

    # Neo4j temporal types expose iso_format(), not isoformat().
    iso_format = getattr(value, 'iso_format', None)
    if callable(iso_format):
        return iso_format()
    isoformat = getattr(value, 'isoformat', None)
    if callable(isoformat):
        return isoformat()

    model_dump = getattr(value, 'model_dump', None)
    if callable(model_dump):
        return model_dump(mode='json')

    if isinstance(value, Iterable) and not isinstance(value, str | bytes | bytearray | dict):
        return list(value)

    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def to_prompt_json(data: Any, ensure_ascii: bool = False, indent: int | None = None) -> str:
    """
    Serialize data to JSON for use in prompts.

    Args:
        data: The data to serialize
        ensure_ascii: If True, escape non-ASCII characters. If False (default), preserve them.
        indent: Number of spaces for indentation. Defaults to None (minified).

    Returns:
        JSON string representation of the data

    Notes:
        By default (ensure_ascii=False), non-ASCII characters (e.g., Korean, Japanese, Chinese)
        are preserved in their original form in the prompt, making them readable
        in LLM logs and improving model understanding.
    """
    return json.dumps(data, ensure_ascii=ensure_ascii, indent=indent, default=_prompt_json_default)
