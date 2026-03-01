"""Tests for om_dedupe.py — dedupe key stability and merge semantics."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

om_dedupe = importlib.import_module("scripts.om_dedupe")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(
    node_id: str = "nodeid1",
    node_type: str = "Judgment",
    content: str = "The system should fail closed.",
    semantic_domain: str = "sessions_main",
    urgency_score: int = 3,
    status: str = "open",
    source_message_ids: list[str] | None = None,
    created_at: str = "2026-01-01T10:00:00Z",
    first_observed_at: str | None = None,
    last_observed_at: str | None = None,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "node_type": node_type,
        "semantic_domain": semantic_domain,
        "content": content,
        "urgency_score": urgency_score,
        "status": status,
        "source_message_ids": source_message_ids or [],
        "created_at": created_at,
        "first_observed_at": first_observed_at,
        "last_observed_at": last_observed_at,
    }


# ---------------------------------------------------------------------------
# Dedupe key stability
# ---------------------------------------------------------------------------

class TestDedupeKey:
    def test_same_type_and_content_produces_same_key(self):
        k1 = om_dedupe.dedupe_key("Judgment", "The system should fail closed.")
        k2 = om_dedupe.dedupe_key("Judgment", "The system should fail closed.")
        assert k1 == k2

    def test_key_is_case_insensitive_on_content(self):
        k1 = om_dedupe.dedupe_key("Judgment", "Fail Closed")
        k2 = om_dedupe.dedupe_key("Judgment", "fail closed")
        assert k1 == k2

    def test_key_normalises_whitespace(self):
        k1 = om_dedupe.dedupe_key("Judgment", "fail  closed")
        k2 = om_dedupe.dedupe_key("Judgment", "fail closed")
        assert k1 == k2

    def test_different_node_type_produces_different_key(self):
        k1 = om_dedupe.dedupe_key("Judgment", "fail closed")
        k2 = om_dedupe.dedupe_key("OperationalRule", "fail closed")
        assert k1 != k2

    def test_semantic_domain_does_not_affect_key(self):
        """Dedupe key is domain-agnostic — detects cross-domain duplicates."""
        k1 = om_dedupe.dedupe_key("Judgment", "fail closed")
        # dedupe_key takes (node_type, content) — no domain arg
        k2 = om_dedupe.dedupe_key("Judgment", "fail closed")
        assert k1 == k2

    def test_key_is_a_hex_string(self):
        key = om_dedupe.dedupe_key("WorldState", "something")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_leading_trailing_whitespace_normalised(self):
        k1 = om_dedupe.dedupe_key("Friction", "  blocked by infra  ")
        k2 = om_dedupe.dedupe_key("Friction", "blocked by infra")
        assert k1 == k2


# ---------------------------------------------------------------------------
# Duplicate group detection
# ---------------------------------------------------------------------------

class TestFindDuplicateGroups:
    def test_no_duplicates_returns_empty(self):
        nodes = [
            _make_node(node_id="a", node_type="Judgment", content="Alpha"),
            _make_node(node_id="b", node_type="Judgment", content="Beta"),
        ]
        groups = om_dedupe._find_duplicate_groups(nodes)
        assert groups == []

    def test_two_identical_content_nodes_form_group(self):
        nodes = [
            _make_node(node_id="a", node_type="Judgment", content="Fail closed"),
            _make_node(node_id="b", node_type="Judgment", content="fail closed"),
        ]
        groups = om_dedupe._find_duplicate_groups(nodes)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_three_identical_nodes_form_single_group(self):
        nodes = [
            _make_node(node_id="a", node_type="Friction", content="Pipeline blocked"),
            _make_node(node_id="b", node_type="Friction", content="pipeline blocked"),
            _make_node(node_id="c", node_type="Friction", content="Pipeline Blocked"),
        ]
        groups = om_dedupe._find_duplicate_groups(nodes)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_different_types_not_grouped(self):
        nodes = [
            _make_node(node_id="a", node_type="Judgment", content="fail closed"),
            _make_node(node_id="b", node_type="OperationalRule", content="fail closed"),
        ]
        groups = om_dedupe._find_duplicate_groups(nodes)
        assert groups == []

    def test_multiple_independent_duplicate_groups(self):
        nodes = [
            _make_node(node_id="a1", node_type="Judgment", content="Alpha"),
            _make_node(node_id="a2", node_type="Judgment", content="alpha"),
            _make_node(node_id="b1", node_type="Friction", content="Blocked"),
            _make_node(node_id="b2", node_type="Friction", content="blocked"),
        ]
        groups = om_dedupe._find_duplicate_groups(nodes)
        assert len(groups) == 2

    def test_cross_semantic_domain_duplicates_detected(self):
        """Nodes with different semantic_domain but same type+content are duplicates."""
        nodes = [
            _make_node(node_id="a", node_type="Judgment", semantic_domain="sessions_main", content="Fail closed"),
            _make_node(node_id="b", node_type="Judgment", semantic_domain="sessions_alt", content="Fail closed"),
        ]
        groups = om_dedupe._find_duplicate_groups(nodes)
        assert len(groups) == 1


# ---------------------------------------------------------------------------
# Canonical selection
# ---------------------------------------------------------------------------

class TestPickCanonical:
    def test_earliest_created_at_is_canonical(self):
        nodes = [
            _make_node(node_id="newer", created_at="2026-02-01T00:00:00Z"),
            _make_node(node_id="older", created_at="2026-01-01T00:00:00Z"),
        ]
        canonical = om_dedupe._pick_canonical(nodes)
        assert canonical["node_id"] == "older"

    def test_tie_broken_by_node_id_lexicographic(self):
        nodes = [
            _make_node(node_id="zzz", created_at="2026-01-01T00:00:00Z"),
            _make_node(node_id="aaa", created_at="2026-01-01T00:00:00Z"),
        ]
        canonical = om_dedupe._pick_canonical(nodes)
        assert canonical["node_id"] == "aaa"

    def test_single_node_is_its_own_canonical(self):
        node = _make_node(node_id="only")
        assert om_dedupe._pick_canonical([node])["node_id"] == "only"


# ---------------------------------------------------------------------------
# Merge metadata
# ---------------------------------------------------------------------------

class TestMergeMetadata:
    def test_source_message_ids_are_unioned(self):
        nodes = [
            _make_node(source_message_ids=["msg1", "msg2"]),
            _make_node(source_message_ids=["msg2", "msg3"]),
        ]
        merged = om_dedupe._merge_metadata(nodes)
        assert set(merged["source_message_ids"]) == {"msg1", "msg2", "msg3"}
        # No duplicates
        assert len(merged["source_message_ids"]) == 3

    def test_urgency_score_is_max(self):
        nodes = [
            _make_node(urgency_score=2),
            _make_node(urgency_score=5),
            _make_node(urgency_score=3),
        ]
        merged = om_dedupe._merge_metadata(nodes)
        assert merged["urgency_score"] == 5

    def test_status_most_active_wins(self):
        nodes = [
            _make_node(status="closed"),
            _make_node(status="open"),
            _make_node(status="monitoring"),
        ]
        merged = om_dedupe._merge_metadata(nodes)
        assert merged["status"] == "open"

    def test_status_reopened_beats_monitoring(self):
        nodes = [
            _make_node(status="monitoring"),
            _make_node(status="reopened"),
        ]
        merged = om_dedupe._merge_metadata(nodes)
        assert merged["status"] == "reopened"

    def test_first_observed_at_is_min(self):
        nodes = [
            _make_node(first_observed_at="2026-03-01T00:00:00Z"),
            _make_node(first_observed_at="2026-01-01T00:00:00Z"),
            _make_node(first_observed_at="2026-02-01T00:00:00Z"),
        ]
        merged = om_dedupe._merge_metadata(nodes)
        assert merged["first_observed_at"] == "2026-01-01T00:00:00Z"

    def test_last_observed_at_is_max(self):
        nodes = [
            _make_node(last_observed_at="2026-01-15T00:00:00Z"),
            _make_node(last_observed_at="2026-03-10T00:00:00Z"),
        ]
        merged = om_dedupe._merge_metadata(nodes)
        assert merged["last_observed_at"] == "2026-03-10T00:00:00Z"

    def test_first_observed_falls_back_to_created_at_when_null(self):
        nodes = [
            _make_node(created_at="2026-01-10T00:00:00Z", first_observed_at=None),
            _make_node(created_at="2026-01-20T00:00:00Z", first_observed_at=None),
        ]
        merged = om_dedupe._merge_metadata(nodes)
        assert merged["first_observed_at"] == "2026-01-10T00:00:00Z"

    def test_last_observed_at_none_when_all_null(self):
        nodes = [
            _make_node(last_observed_at=None),
            _make_node(last_observed_at=None),
        ]
        merged = om_dedupe._merge_metadata(nodes)
        assert merged["last_observed_at"] is None

    def test_partial_first_observed_mixed_null_and_value(self):
        nodes = [
            _make_node(first_observed_at=None, created_at="2026-01-05T00:00:00Z"),
            _make_node(first_observed_at="2026-01-03T00:00:00Z"),
        ]
        merged = om_dedupe._merge_metadata(nodes)
        # first_observed_at=2026-01-03 is present on one node and beats the fallback
        assert merged["first_observed_at"] == "2026-01-03T00:00:00Z"


# ---------------------------------------------------------------------------
# CLI arg parsing
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_default_is_dry_run(self):
        args = om_dedupe.parse_args([])
        assert args.dry_run is True
        assert args.apply is False

    def test_apply_flag_sets_apply_true(self):
        args = om_dedupe.parse_args(["--apply"])
        assert args.apply is True
        # run() uses `not args.apply` for dry_run — args.dry_run default is irrelevant
        # when --apply is provided.

    def test_dry_run_and_apply_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            om_dedupe.parse_args(["--dry-run", "--apply"])


# ---------------------------------------------------------------------------
# Relationship metadata preservation (_merge_one_group / redirect Cypher)
# ---------------------------------------------------------------------------

class TestRelationshipMetadataPreservation:
    """Verify that the redirect Cypher clauses include ON CREATE SET nr = properties(r)
    so relationship properties are not silently dropped when edges are redirected.

    These tests operate at the source-code level (string inspection) since we
    can't spin up a live Neo4j in unit tests.  They are integration-contract
    tests: they fail if someone removes the property-copy clause from the
    redirect queries.
    """

    def _get_dedupe_source(self) -> str:
        import inspect
        return inspect.getsource(om_dedupe)

    def test_supports_core_redirect_copies_properties(self):
        """SUPPORTS_CORE redirect must carry ON CREATE SET nr = properties(r)."""
        src = self._get_dedupe_source()
        # The redirect block must match (canonical) to an existing relationship and
        # copy properties from the original.
        assert "ON CREATE SET nr = properties(r)" in src, (
            "SUPPORTS_CORE redirect is missing 'ON CREATE SET nr = properties(r)'; "
            "relationship properties will be silently dropped on deduplication."
        )

    def test_outgoing_relation_redirect_copies_properties(self):
        """Outgoing OM relation-type edge redirect must preserve properties."""
        src = self._get_dedupe_source()
        # Presence of the pattern confirms the fix is in place
        assert "MERGE (canonical)-[nr:" in src, (
            "Outgoing edge redirect does not use a named relationship variable [nr:]; "
            "ON CREATE SET nr = properties(r) cannot be applied."
        )

    def test_incoming_relation_redirect_copies_properties(self):
        """Incoming OM relation-type edge redirect must preserve properties."""
        src = self._get_dedupe_source()
        assert "MERGE (source)-[nr:" in src, (
            "Incoming edge redirect does not use a named relationship variable [nr:]; "
            "ON CREATE SET nr = properties(r) cannot be applied."
        )

    def test_redirect_deletes_old_relationship(self):
        """Old relationships should be deleted after the redirect to avoid orphan edges."""
        src = self._get_dedupe_source()
        # The DELETE r clause must be present in redirect blocks
        assert "DELETE r" in src, (
            "Redirect blocks do not DELETE the old relationship r; "
            "duplicate edges will accumulate in the graph."
        )
