"""Phase C — Slice 5: Scope Policy tests.

Validates:
1. TOOL_RESULT_ALLOWLIST is a frozenset
2. Default entries are present (calendar, contacts, restaurant)
3. is_tool_result_allowed returns True for allowlisted tools
4. is_tool_result_allowed returns False for non-allowlisted tools
5. Common toolResult types that should be EXCLUDED by default
6. Scope policy doc exists and mentions key sections
7. enforce_tool_result_scope raises ToolResultNotAllowedError on disallowed tools
8. Integration: Graphiti.add_episode rejects disallowed tool_name before any DB write
9. Integration: Graphiti.add_episode passes allowed tool_name through scope gate
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. Allowlist module
# ---------------------------------------------------------------------------

class TestToolResultAllowlist:
    @pytest.fixture(autouse=True)
    def _import(self):
        from config.tool_result_allowlist import TOOL_RESULT_ALLOWLIST, is_tool_result_allowed
        self.allowlist = TOOL_RESULT_ALLOWLIST
        self.is_allowed = is_tool_result_allowed

    def test_allowlist_is_frozenset(self):
        assert isinstance(self.allowlist, frozenset)

    def test_calendar_get_event_allowed(self):
        assert self.is_allowed('calendar_get_event') is True

    def test_calendar_list_events_allowed(self):
        assert self.is_allowed('calendar_list_events') is True

    def test_contacts_lookup_allowed(self):
        assert self.is_allowed('contacts_lookup') is True

    def test_restaurant_lookup_allowed(self):
        assert self.is_allowed('restaurant_lookup') is True

    def test_web_search_not_allowed(self):
        """web_search results are high-noise; must not be in the allowlist."""
        assert self.is_allowed('web_search') is False

    def test_bash_not_allowed(self):
        """bash tool results are dangerous; never allowed."""
        assert self.is_allowed('bash') is False

    def test_generic_tool_result_not_allowed(self):
        assert self.is_allowed('some_random_tool') is False

    def test_empty_string_not_allowed(self):
        assert self.is_allowed('') is False

    def test_allowlist_is_immutable(self):
        """Callers must not be able to modify TOOL_RESULT_ALLOWLIST at runtime."""
        with pytest.raises((AttributeError, TypeError)):
            self.allowlist.add('injected_tool')  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 2. Scope policy doc
# ---------------------------------------------------------------------------

class TestScopePolicyDoc:
    DOC_PATH = Path(__file__).parents[1] / 'docs' / 'scope-policy.md'

    def test_doc_exists(self):
        assert self.DOC_PATH.exists(), 'docs/scope-policy.md must exist'

    def test_doc_mentions_message_only_default(self):
        content = self.DOC_PATH.read_text(encoding='utf-8')
        assert 'message' in content.lower()
        assert 'default' in content.lower()

    def test_doc_mentions_toolresult(self):
        content = self.DOC_PATH.read_text(encoding='utf-8')
        assert 'toolResult' in content or 'tool_result' in content or 'tool result' in content.lower()

    def test_doc_mentions_allowlist(self):
        content = self.DOC_PATH.read_text(encoding='utf-8')
        assert 'allowlist' in content.lower() or 'whitelist' in content.lower()

    def test_doc_mentions_contamination(self):
        content = self.DOC_PATH.read_text(encoding='utf-8')
        assert 'contamination' in content.lower()

    def test_doc_mentions_constrained_soft(self):
        content = self.DOC_PATH.read_text(encoding='utf-8')
        assert 'constrained_soft' in content

    def test_doc_has_change_process_section(self):
        content = self.DOC_PATH.read_text(encoding='utf-8')
        assert 'change' in content.lower() and ('process' in content.lower() or 'prd' in content.lower())

    def test_doc_mentions_group_id_requirement(self):
        content = self.DOC_PATH.read_text(encoding='utf-8')
        assert 'group_id' in content


# ---------------------------------------------------------------------------
# 3. EpisodeType sanity (message is the base type)
# ---------------------------------------------------------------------------

class TestEpisodeTypeDefaults:
    def test_message_episode_type_exists(self):
        from graphiti_core.nodes import EpisodeType
        assert hasattr(EpisodeType, 'message')

    def test_json_episode_type_exists(self):
        from graphiti_core.nodes import EpisodeType
        assert hasattr(EpisodeType, 'json')

    def test_message_value_is_message_string(self):
        from graphiti_core.nodes import EpisodeType
        assert EpisodeType.message.value == 'message'


# ---------------------------------------------------------------------------
# 4. enforce_tool_result_scope — fail-closed enforcer
# ---------------------------------------------------------------------------

class TestEnforceToolResultScope:
    @pytest.fixture(autouse=True)
    def _import(self):
        from config.tool_result_allowlist import (
            ToolResultNotAllowedError,
            enforce_tool_result_scope,
        )
        self.enforce = enforce_tool_result_scope
        self.Error = ToolResultNotAllowedError

    def test_raises_for_disallowed_tool(self):
        """bash is a high-risk tool; must be rejected."""
        with pytest.raises(self.Error):
            self.enforce('bash')

    def test_raises_for_web_search(self):
        with pytest.raises(self.Error):
            self.enforce('web_search')

    def test_raises_for_unknown_tool(self):
        with pytest.raises(self.Error):
            self.enforce('some_random_tool')

    def test_raises_for_empty_string(self):
        with pytest.raises(self.Error):
            self.enforce('')

    def test_passes_for_calendar_get_event(self):
        """calendar_get_event is allowlisted; must not raise."""
        self.enforce('calendar_get_event')  # no exception

    def test_passes_for_contacts_lookup(self):
        self.enforce('contacts_lookup')  # no exception

    def test_passes_for_restaurant_lookup(self):
        self.enforce('restaurant_lookup')  # no exception

    def test_error_is_valueerror_subclass(self):
        """ToolResultNotAllowedError must be catchable as ValueError."""
        with pytest.raises(ValueError):
            self.enforce('bash')

    def test_error_message_names_tool(self):
        """Error message must identify the disallowed tool."""
        with pytest.raises(self.Error, match='bash'):
            self.enforce('bash')


# ---------------------------------------------------------------------------
# 5. Integration: add_episode rejects disallowed tool_name before any DB write
# ---------------------------------------------------------------------------

class TestAddEpisodeScopePolicyGate:
    """Verify that Graphiti.add_episode enforces scope policy as a pre-DB gate.

    We run the actual add_episode code path against a mock driver that FAILS
    if any DB method is called, proving the scope check fires before writes.
    For allowed tools, the scope gate must not raise (later DB mock errors are
    expected and are caught separately).
    """

    @pytest.fixture(autouse=True)
    def _build_graphiti(self):
        """Construct a minimal Graphiti instance with mocked internals."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from datetime import datetime, timezone

        self.now = datetime(2026, 3, 1, tzinfo=timezone.utc)

        # Patch heavy async machinery so we can call add_episode synchronously
        with patch('graphiti_core.graphiti.Graphiti.__init__', return_value=None):
            from graphiti_core.graphiti import Graphiti
            self.g = object.__new__(Graphiti)

        # Minimal attribute shims needed before scope gate
        self.g.driver = MagicMock()
        self.g.driver.provider = MagicMock()
        self.g.tracer = MagicMock()
        self.g.clients = MagicMock()

        yield

    def _run(self, coro):
        import asyncio
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_disallowed_tool_name_raises_before_db(self):
        """bash is not allowlisted; add_episode must raise before any DB call."""
        from unittest.mock import MagicMock
        from graphiti_core.nodes import EpisodeType
        from config.tool_result_allowlist import ToolResultNotAllowedError

        # DB driver must NEVER be called — a call would mean the guard is missing
        self.g.driver.execute_query = MagicMock(
            side_effect=AssertionError('DB was called before scope gate fired')
        )

        with pytest.raises(ToolResultNotAllowedError):
            self._run(
                self.g.add_episode(
                    name='test',
                    episode_body='{"type": "bash_result"}',
                    source_description='bash result',
                    reference_time=self.now,
                    source=EpisodeType.json,
                    group_id='test-group',
                    tool_name='bash',
                )
            )
        # Confirm DB was never touched
        self.g.driver.execute_query.assert_not_called()

    def test_disallowed_web_search_raises(self):
        """web_search results are high-noise and must be rejected."""
        from graphiti_core.nodes import EpisodeType
        from config.tool_result_allowlist import ToolResultNotAllowedError

        with pytest.raises(ToolResultNotAllowedError, match='web_search'):
            self._run(
                self.g.add_episode(
                    name='web results',
                    episode_body='{"results": []}',
                    source_description='web search output',
                    reference_time=self.now,
                    source=EpisodeType.json,
                    group_id='test-group',
                    tool_name='web_search',
                )
            )

    def test_no_tool_name_bypasses_scope_gate(self):
        """Callers without tool_name (message/text episodes) must pass through.

        The scope gate must only activate when tool_name is explicitly provided.
        This preserves backwards compatibility for all existing callers.
        Subsequent operations may fail (mocked driver), but the scope gate
        itself must not raise.
        """
        from graphiti_core.nodes import EpisodeType
        from config.tool_result_allowlist import ToolResultNotAllowedError

        try:
            self._run(
                self.g.add_episode(
                    name='normal message',
                    episode_body='user: hello',
                    source_description='chat',
                    reference_time=self.now,
                    source=EpisodeType.message,
                    group_id='test-group',
                    tool_name=None,  # no tool name → scope gate bypassed
                )
            )
        except ToolResultNotAllowedError:
            pytest.fail(
                'add_episode raised ToolResultNotAllowedError for tool_name=None. '
                'The scope gate must only fire when tool_name is explicitly provided.'
            )
        except Exception:
            # Any other exception from mocked internals is fine —
            # the scope gate itself did not block this call.
            pass

    def test_allowed_tool_name_passes_scope_gate(self):
        """calendar_get_event is allowlisted; the scope gate must not raise.

        The call may fail later (mocked driver) but must pass the scope check.
        """
        from graphiti_core.nodes import EpisodeType
        from config.tool_result_allowlist import ToolResultNotAllowedError

        try:
            self._run(
                self.g.add_episode(
                    name='calendar event',
                    episode_body='{"title": "Team standup"}',
                    source_description='calendar',
                    reference_time=self.now,
                    source=EpisodeType.json,
                    group_id='test-group',
                    tool_name='calendar_get_event',
                )
            )
        except ToolResultNotAllowedError:
            pytest.fail(
                'add_episode raised ToolResultNotAllowedError for '
                "tool_name='calendar_get_event', which is on the allowlist."
            )
        except Exception:
            pass  # Expected: mocked driver will fail later — that's fine
