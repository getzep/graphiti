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

import json
from datetime import date, datetime
from typing import Any

DO_NOT_ESCAPE_UNICODE = '\nDo not escape unicode characters.\n'


def _json_fallback(obj: Any) -> str:
    """`json.dumps(default=...)` handler for values the stock encoder rejects.

    Entity and edge attribute dicts commonly carry `datetime`/`date` values
    (e.g. a date extracted into an entity attribute). Without a fallback,
    serializing them for a prompt raises
    `TypeError: Object of type datetime is not JSON serializable`. Dates are
    rendered as ISO-8601; any other non-serializable value falls back to its
    `str()` so prompt construction never crashes.
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return str(obj)


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

        Values the standard JSON encoder cannot serialize (e.g. ``datetime``)
        are handled via ``_json_fallback``: dates become ISO-8601 strings and
        anything else falls back to ``str()``, so prompt construction never
        raises on real-world attribute data.
    """
    return json.dumps(data, ensure_ascii=ensure_ascii, indent=indent, default=_json_fallback)
