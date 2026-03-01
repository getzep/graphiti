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
    'is_tool_result_allowed',
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
