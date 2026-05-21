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
import re
from typing import Any

# Characters that follow a backslash in valid JSON escape sequences.
# JSON spec (RFC 8259 §7): \" \\ \/ \b \f \n \r \t \uXXXX
_VALID_JSON_ESCAPES = frozenset('"\\/bfnrtu')


def safe_json_loads(raw: str) -> Any:
    """Parse a JSON string, tolerating invalid backslash escapes.

    LLMs sometimes embed raw LaTeX, Windows paths, or other content
    containing unescaped backslashes (e.g. ``\\hat``, ``\\psi``,
    ``C:\\Users``) inside JSON string values.  Standard ``json.loads``
    rejects these as illegal escape sequences.

    This helper first tries the normal ``json.loads``.  If that fails
    with a ``JSONDecodeError``, it escapes every backslash that is
    **not** followed by one of the characters permitted by the JSON
    specification (``" \\ / b f n r t u``) and retries.

    Args:
        raw: The raw JSON string to parse.

    Returns:
        The parsed Python object (dict, list, str, etc.).

    Raises:
        json.JSONDecodeError: If the string still cannot be parsed
            after fixing illegal escapes.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Replace backslash + non-JSON-escape-char with double-backslash + char.
        # This turns e.g. \h into \\h, \p into \\p, etc.
        fixed = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', raw)
        return json.loads(fixed)
