"""Phase C — Slice 5: Scope Policy tests.

Validates:
1. TOOL_RESULT_ALLOWLIST is a frozenset
2. Default entries are present (calendar, contacts, restaurant)
3. is_tool_result_allowed returns True for allowlisted tools
4. is_tool_result_allowed returns False for non-allowlisted tools
5. Common toolResult types that should be EXCLUDED by default
6. Scope policy doc exists and mentions key sections
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
