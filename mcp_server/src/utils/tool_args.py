"""Refuse unknown tool arguments instead of silently dropping them.

MCPServer validates a tool call against a pydantic model built from the function
signature, and that model ignores extra keys. A caller that passes `group_id` to a
tool that takes `group_ids` therefore gets a successful call in the wrong scope,
with no hint that the argument was discarded. `StrictMCPServer` checks the keys of
every call against the tool's declared parameters first and raises a `ToolError`
naming the offending argument and the closest valid one.
"""

import difflib
from collections.abc import Iterable, Sequence
from typing import Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError


def closest_argument(name: str, valid: Iterable[str]) -> str | None:
    """Return the valid argument name nearest to `name`, or None when nothing is close."""
    matches = difflib.get_close_matches(name, list(valid), n=1, cutoff=0.6)
    return matches[0] if matches else None


def unknown_argument_message(tool_name: str, unknown: Sequence[str], valid: Sequence[str]) -> str:
    """Build the refusal text: each unknown name, its nearest valid name, the valid list."""
    parts = []
    for name in unknown:
        suggestion = closest_argument(name, valid)
        part = f"Unknown argument '{name}' for {tool_name}."
        if suggestion is not None:
            part += f" Did you mean '{suggestion}'?"
        parts.append(part)
    return ' '.join(parts) + f' Valid arguments: {", ".join(valid)}.'


class StrictMCPServer(MCPServer):
    """MCPServer that refuses tool calls carrying arguments the tool does not declare."""

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: Context | None = None
    ) -> Any:
        tool = self._tool_manager.get_tool(name)
        if tool is not None:
            valid = list(tool.parameters.get('properties', {}))
            unknown = [key for key in arguments if key not in valid]
            if unknown:
                raise ToolError(unknown_argument_message(name, unknown, valid))
        return await super().call_tool(name, arguments, context)
