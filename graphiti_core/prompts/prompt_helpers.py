import json
from typing import Any

DO_NOT_ESCAPE_UNICODE = '\nDo not escape unicode characters.\n'


def to_prompt_json(data: Any, ensure_ascii: bool = True, indent: int = 2) -> str:
    """
    Serialize data to JSON for use in prompts.

    Args:
        data: The data to serialize
        ensure_ascii: If True, escape non-ASCII characters. If False, preserve them.
        indent: Number of spaces for indentation

    Returns:
        JSON string representation of the data

    Notes:
        When ensure_ascii=False, non-ASCII characters (e.g., Korean, Japanese, Chinese)
        are preserved in their original form in the prompt, making them readable
        in LLM logs and improving model understanding.
    """
    return json.dumps(data, ensure_ascii=ensure_ascii, indent=indent)
