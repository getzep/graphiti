"""Tests for unified promotion policy v3 scaffolding.

Covers:
- Feature flag gating (GRAPHITI_POLICY_V3_ENABLED)
- active_policy_version() returns correct label
- policy_v3_enabled() respects env var
- New candidate rows get stamped with promotion-v3
- Rollback path: disabling flag stamps promotion-v2
- _decision_for_actor recognises both v2 and v3 actors
- Lane policy constants are complete and non-overlapping
- compute_policy_trace() stamps policy_version via active_policy_version()

All tests are stdlib-only (no Neo4j, no Graphiti MCP required).

Migration notes:
- Existing candidates rows with policy_version="promotion-v2" are valid; the DB
  schema stores policy_version as TEXT and has no CHECK constraint on it.
- No schema migration is required: v3 rows coexist with v2 rows.
- Downstream consumers must accept both "promotion-v2" and "promotion-v3" values.
"""
from __future__ import annotations

import importlib
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Import module under test
# ─────────────────────────────────────────────────────────────────────────────

# Ensure the truth package is importable from repo root
_TRUTH_PARENT = str(Path(__file__).resolve().parents[2])
if _TRUTH_PARENT not in sys.path:
    sys.path.insert(0, _TRUTH_PARENT)

candidates = importlib.import_module("truth.candidates")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _in_memory_db() -> sqlite3.Connection:
    """Open an in-memory candidates DB for isolation."""
    # DB_PATH_DEFAULT points to the on-disk file; for tests use :memory:
    conn = candidates.connect(":memory:")
    return conn


def _minimal_candidate(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "subject": "user:principal",
        "predicate": "pref.test_preference",
        "scope": "private",
        "assertion_type": "preference",
        "value": {"v": "test"},
        "evidence_refs": [{"source_key": "sessions:s1", "evidence_id": "e1"}],
        "speaker_id": "owner",
        "confidence": 0.92,
        "origin": "extracted",
    }
    defaults.update(overrides)
    return defaults


# ─────────────────────────────────────────────────────────────────────────────
# Feature flag: policy_v3_enabled
# ─────────────────────────────────────────────────────────────────────────────

class TestPolicyV3EnabledFlag:
    def test_default_is_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        assert candidates.policy_v3_enabled() is True

    def test_explicit_true_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for val in ("1", "true", "True", "TRUE", "yes", "on"):
            monkeypatch.setenv(candidates.POLICY_V3_ENABLED_ENV, val)
            assert candidates.policy_v3_enabled() is True, f"Expected True for {val!r}"

    def test_explicit_false_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for val in ("0", "false", "False", "FALSE", "no", "off"):
            monkeypatch.setenv(candidates.POLICY_V3_ENABLED_ENV, val)
            assert candidates.policy_v3_enabled() is False, f"Expected False for {val!r}"

    def test_unrecognised_value_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(candidates.POLICY_V3_ENABLED_ENV, "maybe")
        # Unrecognised → default (True)
        assert candidates.policy_v3_enabled() is True


# ─────────────────────────────────────────────────────────────────────────────
# active_policy_version
# ─────────────────────────────────────────────────────────────────────────────

class TestActivePolicyVersion:
    def test_returns_v3_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        assert candidates.active_policy_version() == candidates.POLICY_VERSION_V3
        assert candidates.active_policy_version() == "promotion-v3"

    def test_returns_v2_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(candidates.POLICY_V3_ENABLED_ENV, "0")
        assert candidates.active_policy_version() == candidates.POLICY_VERSION_V2
        assert candidates.active_policy_version() == "promotion-v2"

    def test_policy_version_default_constant_is_v3(self) -> None:
        """POLICY_VERSION_DEFAULT must equal POLICY_VERSION_V3 (US-001 contract)."""
        assert candidates.POLICY_VERSION_DEFAULT == candidates.POLICY_VERSION_V3


# ─────────────────────────────────────────────────────────────────────────────
# compute_policy_trace stamps correct version
# ─────────────────────────────────────────────────────────────────────────────

class TestComputePolicyTraceVersion:
    def test_stamps_v3_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        trace = candidates.compute_policy_trace(
            {
                "subject": "user:principal",
                "predicate": "pref.food",
                "scope": "private",
                "assertion_type": "preference",
                "value": "sushi",
                "speaker_id": "owner",
                "confidence": 0.85,
                "origin": "extracted",
            },
            {},
        )
        assert trace["policy_version"] == "promotion-v3"

    def test_stamps_v2_when_flag_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(candidates.POLICY_V3_ENABLED_ENV, "0")
        trace = candidates.compute_policy_trace(
            {
                "subject": "user:principal",
                "predicate": "pref.food",
                "scope": "private",
                "assertion_type": "preference",
                "value": "sushi",
                "speaker_id": "owner",
                "confidence": 0.85,
                "origin": "extracted",
            },
            {},
        )
        assert trace["policy_version"] == "promotion-v2"


# ─────────────────────────────────────────────────────────────────────────────
# upsert_candidate writes v3 policy_version to DB
# ─────────────────────────────────────────────────────────────────────────────

class TestUpsertCandidateV3Version:
    def test_new_row_gets_v3_policy_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        conn = _in_memory_db()
        try:
            c = _minimal_candidate()
            result = candidates.upsert_candidate(conn, **c)
            assert result.created is True

            row = conn.execute(
                "SELECT policy_version FROM candidates WHERE candidate_id = ?",
                (result.candidate_id,),
            ).fetchone()
            assert row is not None
            assert row["policy_version"] == "promotion-v3"
        finally:
            conn.close()

    def test_rollback_path_writes_v2_when_flag_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(candidates.POLICY_V3_ENABLED_ENV, "0")
        conn = _in_memory_db()
        try:
            c = _minimal_candidate(predicate="pref.rollback_check")
            result = candidates.upsert_candidate(conn, **c)
            assert result.created is True

            row = conn.execute(
                "SELECT policy_version FROM candidates WHERE candidate_id = ?",
                (result.candidate_id,),
            ).fetchone()
            assert row is not None
            assert row["policy_version"] == "promotion-v2"
        finally:
            conn.close()

    def test_merge_row_preserves_v3_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Merging evidence into an existing candidate should keep v3 version."""
        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        conn = _in_memory_db()
        try:
            c = _minimal_candidate()
            r1 = candidates.upsert_candidate(conn, **c)
            assert r1.created is True

            # Merge a new evidence ref
            c2 = dict(c)
            c2["evidence_refs"] = [{"source_key": "sessions:s2", "evidence_id": "e2"}]
            r2 = candidates.upsert_candidate(conn, **c2)
            assert r2.created is False
            assert r2.merged_evidence_added == 1

            row = conn.execute(
                "SELECT policy_version FROM candidates WHERE candidate_id = ?",
                (r1.candidate_id,),
            ).fetchone()
            assert row["policy_version"] == "promotion-v3"
        finally:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# _decision_for_actor accepts both policy actors (US-001)
# ─────────────────────────────────────────────────────────────────────────────

class TestDecisionForActor:
    def test_policy_v3_actor_returns_auto_promoted(self) -> None:
        assert candidates._decision_for_actor("policy:v3") == "auto_promoted"

    def test_policy_v2_actor_still_returns_auto_promoted(self) -> None:
        """Backward-compat: v2 actor must still produce the same decision value."""
        assert candidates._decision_for_actor("policy:v2") == "auto_promoted"

    def test_human_actor_returns_approved(self) -> None:
        assert candidates._decision_for_actor("ui:yuan") == "approved"
        assert candidates._decision_for_actor("crm:import") == "approved"


# ─────────────────────────────────────────────────────────────────────────────
# Lane policy constants (US-002 / locked matrix)
# ─────────────────────────────────────────────────────────────────────────────

class TestLanePolicyConstants:
    def test_retrieval_eligible_global_contents(self) -> None:
        assert "s1_sessions_main" in candidates.LANE_RETRIEVAL_ELIGIBLE_GLOBAL
        assert "s1_observational_memory" in candidates.LANE_RETRIEVAL_ELIGIBLE_GLOBAL

    def test_retrieval_eligible_vc_scoped_contents(self) -> None:
        assert "s1_chatgpt_history" in candidates.LANE_RETRIEVAL_ELIGIBLE_VC_SCOPED

    def test_corroboration_only_contents(self) -> None:
        assert "s1_curated_refs" in candidates.LANE_CORROBORATION_ONLY
        assert "s1_memory_day1" in candidates.LANE_CORROBORATION_ONLY

    def test_no_overlap_between_global_retrieval_and_corroboration_only(self) -> None:
        overlap = candidates.LANE_RETRIEVAL_ELIGIBLE_GLOBAL & candidates.LANE_CORROBORATION_ONLY
        assert overlap == frozenset(), f"Unexpected overlap: {overlap}"

    def test_no_overlap_between_vc_scoped_and_corroboration_only(self) -> None:
        overlap = candidates.LANE_RETRIEVAL_ELIGIBLE_VC_SCOPED & candidates.LANE_CORROBORATION_ONLY
        assert overlap == frozenset(), f"Unexpected overlap: {overlap}"

    def test_candidates_eligible_includes_sessions_main(self) -> None:
        assert "s1_sessions_main" in candidates.LANE_CANDIDATES_ELIGIBLE

    def test_candidates_eligible_includes_observational_memory(self) -> None:
        assert "s1_observational_memory" in candidates.LANE_CANDIDATES_ELIGIBLE

    def test_candidates_eligible_includes_chatgpt_history(self) -> None:
        assert "s1_chatgpt_history" in candidates.LANE_CANDIDATES_ELIGIBLE

    def test_candidates_eligible_includes_memory_day1(self) -> None:
        """s1_memory_day1 is corroboration-only but still candidate-generating."""
        assert "s1_memory_day1" in candidates.LANE_CANDIDATES_ELIGIBLE

    def test_curated_refs_not_in_candidates_eligible(self) -> None:
        """s1_curated_refs is corroboration-only and not a candidate source by default."""
        assert "s1_curated_refs" not in candidates.LANE_CANDIDATES_ELIGIBLE


# ─────────────────────────────────────────────────────────────────────────────
# Migration safety: v2 rows coexist with v3 rows (no schema conflict)
# ─────────────────────────────────────────────────────────────────────────────

class TestMigrationCompat:
    def test_v2_and_v3_rows_coexist_in_same_db(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Existing v2 rows must remain queryable alongside new v3 rows."""
        conn = _in_memory_db()
        try:
            # Insert a v3 row (flag enabled by default)
            monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
            c_v3 = _minimal_candidate(predicate="pref.v3_test")
            r_v3 = candidates.upsert_candidate(conn, **c_v3)

            # Manually backfill a v2-stamped row (simulates pre-migration row)
            conn.execute(
                "UPDATE candidates SET policy_version = 'promotion-v2' WHERE candidate_id = ?",
                (r_v3.candidate_id,),
            )
            conn.commit()

            # Insert another v3 row
            c_v3b = _minimal_candidate(predicate="pref.v3_test_b")
            candidates.upsert_candidate(conn, **c_v3b)

            rows = conn.execute(
                "SELECT policy_version FROM candidates ORDER BY created_at"
            ).fetchall()
            versions = [r["policy_version"] for r in rows]
            assert "promotion-v2" in versions
            assert "promotion-v3" in versions
        finally:
            conn.close()

    def test_refresh_policy_state_stamps_active_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """refresh_candidate_policy_state must stamp the active version on re-evaluation."""
        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        conn = _in_memory_db()
        try:
            c = _minimal_candidate(predicate="pref.refresh_test")
            r = candidates.upsert_candidate(conn, **c)

            # Backfill v2 to simulate a pre-existing row
            conn.execute(
                "UPDATE candidates SET policy_version = 'promotion-v2' WHERE candidate_id = ?",
                (r.candidate_id,),
            )
            conn.commit()

            candidates.refresh_candidate_policy_state(conn, r.candidate_id)
            updated_row = conn.execute(
                "SELECT policy_version FROM candidates WHERE candidate_id = ?",
                (r.candidate_id,),
            ).fetchone()
            assert updated_row["policy_version"] == "promotion-v3"
        finally:
            conn.close()
