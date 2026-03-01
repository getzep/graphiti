"""Tool result allowlist — controls which tool types may be ingested into
the knowledge graph as structured episodes.

See :doc:`docs/scope-policy.md` for the full scope policy and change process.

To add a new entry:
1. Add the tool name to TOOL_RESULT_ALLOWLIST below.
2. Update the table in docs/scope-policy.md §1 and §2.
3. Add a test in tests/test_tool_result_scope.py.
4. Open a PR with justification.
"""

from __future__ import annotations

__all__ = [
    'TOOL_RESULT_ALLOWLIST',
    'ToolResultNotAllowedError',
    'is_tool_result_allowed',
    'enforce_tool_result_scope',
]

#: Frozenset of tool names permitted for structured episode ingestion.
#: Keep this list minimal — each addition expands the injection surface.
TOOL_RESULT_ALLOWLIST: frozenset[str] = frozenset({
    # Structured calendar facts (CalendarGuard-validated)
    'calendar_get_event',
    'calendar_list_events',

    # Contact/people data (confirmed safe origin)
    'contacts_lookup',

    # Restaurant/venue records (structured, human-curated)
    'restaurant_lookup',
})


class ToolResultNotAllowedError(ValueError):
    """Raised when a tool result is submitted for ingestion but its tool name
    is not on the TOOL_RESULT_ALLOWLIST.

    Inherits from ``ValueError`` so callers that catch ``ValueError`` for
    generic input validation continue to work.
    """


def is_tool_result_allowed(tool_name: str) -> bool:
    """Return True if *tool_name* is on the opt-in allowlist.

    Parameters
    ----------
    tool_name:
        The name of the tool that produced the result (e.g. 'calendar_get_event').

    Returns
    -------
    bool
        True = allowed for ingestion; False = excluded by default scope policy.
    """
    return tool_name in TOOL_RESULT_ALLOWLIST


def enforce_tool_result_scope(tool_name: str) -> None:
    """Fail closed if *tool_name* is not on the allowlist.

    Call this at the ingestion entry point before any DB writes to ensure
    that disallowed tool results are rejected unconditionally.

    Parameters
    ----------
    tool_name:
        The name of the tool that produced the result.

    Raises
    ------
    ToolResultNotAllowedError
        If *tool_name* is not in TOOL_RESULT_ALLOWLIST.
    """
    if not is_tool_result_allowed(tool_name):
        raise ToolResultNotAllowedError(
            f"Tool result from {tool_name!r} is not permitted for graph ingestion. "
            f"To add it, update TOOL_RESULT_ALLOWLIST in config/tool_result_allowlist.py "
            f"and open a PR with justification. "
            f"Allowed tools: {sorted(TOOL_RESULT_ALLOWLIST)!r}"
        )
