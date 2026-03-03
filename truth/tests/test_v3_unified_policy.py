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
- policy_allows_promotion(): unified v3 decision contract (one path for OM + lane)
- promotion_policy_v3.promote_candidate(): v3 gate active when enabled; bypassed on rollback
- Contradiction gate: conflict candidates blocked at policy gate
- Supersede gate: seeded_supersede_ok candidates pass policy gate

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


# ─────────────────────────────────────────────────────────────────────────────
# policy_allows_promotion: unified v3 decision contract
# ─────────────────────────────────────────────────────────────────────────────

class TestPolicyAllowsPromotion:
    """policy_allows_promotion is the *one decision contract* for both OM and
    graph-lane candidates.  Both paths must produce the same allow/deny outcome
    for the same candidate data.
    """

    # -- helper: auto-promote-eligible candidate (low-risk pref, owner, 0.92 confidence)
    _AUTO_PROMOTE_CANDIDATE: dict = {
        "predicate": "pref.food",
        "confidence": 0.92,
        "speaker_id": "owner",
    }

    def test_auto_promote_candidate_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        conn = _in_memory_db()
        try:
            c = _minimal_candidate(**self._AUTO_PROMOTE_CANDIDATE)
            r = candidates.upsert_candidate(conn, **c)

            allowed, reason = candidates.policy_allows_promotion(r.candidate_id, conn=conn)

            assert allowed is True, f"Expected allowed, got reason={reason!r}"
            assert reason == "auto_promote"
        finally:
            conn.close()

    def test_not_found_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """policy_allows_promotion must fail closed when the candidate row is absent."""
        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        conn = _in_memory_db()
        try:
            allowed, reason = candidates.policy_allows_promotion(
                "DOESNOTEXIST", conn=conn
            )
            assert allowed is False
            assert reason == "candidate_not_found"
        finally:
            conn.close()

    def test_pending_low_confidence_is_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Candidates below auto-promote threshold must not be allowed."""
        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        conn = _in_memory_db()
        try:
            c = _minimal_candidate(
                predicate="pref.diet",
                confidence=0.50,  # below AUTO_PROMOTE_THRESHOLD
                speaker_id="owner",
            )
            r = candidates.upsert_candidate(conn, **c)
            allowed, reason = candidates.policy_allows_promotion(r.candidate_id, conn=conn)
            assert allowed is False
            # recommendation should be "pending" or "recommended_approve", not auto_promote
            assert "auto_promote" not in reason
        finally:
            conn.close()

    def test_contradiction_gate_blocks_conflicted_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A candidate with conflict_with_fact_id and seeded_supersede_ok=False must be
        blocked: compute_policy_trace → recommendation='requires_approval'."""
        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        conn = _in_memory_db()
        try:
            c = _minimal_candidate(
                predicate="pref.diet",
                confidence=0.92,
                speaker_id="owner",
                conflict_with_fact_id="fact:existing-001",
                seeded_supersede_ok=False,
            )
            r = candidates.upsert_candidate(conn, **c)
            allowed, reason = candidates.policy_allows_promotion(r.candidate_id, conn=conn)
            assert allowed is False, (
                f"Contradiction gate failed: conflicted candidate should be blocked. reason={reason!r}"
            )
            assert "auto_promote" not in reason
        finally:
            conn.close()

    def test_supersede_gate_allows_seeded_supersede(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A conflicted candidate with seeded_supersede_ok=True and sufficient
        confidence/owner speaker must pass the supersede gate:
        compute_policy_trace → recommendation='auto_supersede'."""
        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        conn = _in_memory_db()
        try:
            c = _minimal_candidate(
                predicate="pref.diet",  # medium risk → needs >=2 source families
                confidence=0.92,
                speaker_id="owner",
                conflict_with_fact_id="fact:old-001",
                seeded_supersede_ok=True,
                evidence_refs=[
                    {"source_key": "sessions:s1", "evidence_id": "e1"},
                    {"source_key": "chatgpt:s2", "evidence_id": "e2"},
                ],
            )
            r = candidates.upsert_candidate(conn, **c)
            allowed, reason = candidates.policy_allows_promotion(r.candidate_id, conn=conn)
            assert allowed is True, (
                f"Supersede gate failed: seeded_supersede_ok candidate should pass. reason={reason!r}"
            )
            assert reason == "auto_supersede"
        finally:
            conn.close()

    def test_uses_stored_trace_not_recomputed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the stored policy_trace_json has a recommendation, it must be used
        directly (no re-evaluation).  Verify by injecting a synthetic trace."""
        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        conn = _in_memory_db()
        try:
            c = _minimal_candidate(confidence=0.50)  # would normally block
            r = candidates.upsert_candidate(conn, **c)

            # Inject a synthetic trace that says auto_promote
            synthetic_trace = candidates.json_c14n(
                {"recommendation": "auto_promote", "policy_version": "promotion-v3"}
            )
            conn.execute(
                "UPDATE candidates SET policy_trace_json = ? WHERE candidate_id = ?",
                (synthetic_trace, r.candidate_id),
            )
            conn.commit()

            allowed, reason = candidates.policy_allows_promotion(r.candidate_id, conn=conn)
            assert allowed is True
            assert reason == "auto_promote"
        finally:
            conn.close()

    def test_recomputes_when_trace_version_is_stale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stored traces with stale policy_version must be recomputed fail-closed."""
        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        conn = _in_memory_db()
        try:
            c = _minimal_candidate(confidence=0.50)  # normally blocked
            r = candidates.upsert_candidate(conn, **c)

            # Inject stale trace (v2 under active v3) that would incorrectly allow.
            stale_trace = candidates.json_c14n(
                {"recommendation": "auto_promote", "policy_version": "promotion-v2"}
            )
            conn.execute(
                "UPDATE candidates SET policy_trace_json = ? WHERE candidate_id = ?",
                (stale_trace, r.candidate_id),
            )
            conn.commit()

            allowed, reason = candidates.policy_allows_promotion(r.candidate_id, conn=conn)
            assert allowed is False
            assert reason != "auto_promote"
            assert reason.startswith("recommendation=")
        finally:
            conn.close()

    def test_recompute_preserves_trace_only_content_cues(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stale-trace recompute must preserve cue-based risk escalation signals."""
        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        conn = _in_memory_db()
        try:
            c = _minimal_candidate(**self._AUTO_PROMOTE_CANDIDATE)
            r = candidates.upsert_candidate(conn, **c)

            stale_trace = candidates.json_c14n(
                {
                    "policy_version": "promotion-v2",
                    "origin": "extracted",
                    "content_cues": ["password reset token"],
                    "recommendation": "requires_approval",
                }
            )
            conn.execute(
                "UPDATE candidates SET policy_trace_json = ? WHERE candidate_id = ?",
                (stale_trace, r.candidate_id),
            )
            conn.commit()

            allowed, reason = candidates.policy_allows_promotion(r.candidate_id, conn=conn)
            assert allowed is False
            assert reason == "recommendation=requires_approval"
        finally:
            conn.close()

    @pytest.mark.parametrize("status", ["denied", "expired", "superseded"])
    def test_terminal_statuses_block_even_if_trace_recommends_auto_promote(
        self,
        monkeypatch: pytest.MonkeyPatch,
        status: str,
    ) -> None:
        """Terminal negative statuses must block promotion before trace pass-through."""
        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        conn = _in_memory_db()
        try:
            c = _minimal_candidate(**self._AUTO_PROMOTE_CANDIDATE)
            r = candidates.upsert_candidate(conn, **c)

            fresh_auto_promote = candidates.json_c14n(
                {
                    "policy_version": "promotion-v3",
                    "evaluated_at": "2026-01-01T00:00:00Z",
                    "recommendation": "auto_promote",
                }
            )
            conn.execute(
                "UPDATE candidates SET status = ?, policy_trace_json = ? WHERE candidate_id = ?",
                (status, fresh_auto_promote, r.candidate_id),
            )
            conn.commit()

            allowed, reason = candidates.policy_allows_promotion(r.candidate_id, conn=conn)
            assert allowed is False
            assert reason == f"status_terminal={status}"
        finally:
            conn.close()

    def test_fallback_recompute_when_trace_empty_fails_safe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty traces (migration fallback) must not silently loosen to auto-promote."""
        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        conn = _in_memory_db()
        try:
            c = _minimal_candidate(**self._AUTO_PROMOTE_CANDIDATE)
            r = candidates.upsert_candidate(conn, **c)

            # Wipe the stored trace to simulate a row with no recoverable trace inputs.
            conn.execute(
                "UPDATE candidates SET policy_trace_json = '{}' WHERE candidate_id = ?",
                (r.candidate_id,),
            )
            conn.commit()

            allowed, reason = candidates.policy_allows_promotion(r.candidate_id, conn=conn)
            assert allowed is False
            assert "auto_promote" not in reason
        finally:
            conn.close()

    def test_high_risk_candidate_requires_approval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """High-risk namespace candidates (identity., security., etc.) must never
        auto-promote — they require manual approval."""
        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        conn = _in_memory_db()
        try:
            c = _minimal_candidate(
                predicate="identity.full_name",  # high-risk namespace
                confidence=0.99,
                speaker_id="owner",
            )
            r = candidates.upsert_candidate(conn, **c)
            allowed, reason = candidates.policy_allows_promotion(r.candidate_id, conn=conn)
            assert allowed is False
            assert "auto_promote" not in reason
        finally:
            conn.close()

    def test_ineligible_assertion_type_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """IneligibleAssertionTypeError is raised before the row is inserted; the
        function must return (False, ...) without a row to read."""
        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        conn = _in_memory_db()
        try:
            import pytest as _pytest
            with _pytest.raises(candidates.IneligibleAssertionTypeError):
                candidates.upsert_candidate(
                    conn,
                    **_minimal_candidate(assertion_type="question"),  # ineligible
                )
            # No row inserted → policy_allows_promotion fails closed
            rows = conn.execute("SELECT candidate_id FROM candidates").fetchall()
            assert rows == []
        finally:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# promotion_policy_v3.promote_candidate: v3 gate integration
# ─────────────────────────────────────────────────────────────────────────────

class TestPromotionPolicyV3GateIntegration:
    """Tests that promote_candidate() in promotion_policy_v3 uses the unified
    v3 policy gate (policy_allows_promotion) when v3 is enabled, and bypasses
    it cleanly on rollback.
    """

    def _make_conn_with_auto_promote_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[Any, str]:
        """Return (conn, candidate_id) for an auto-promote-eligible candidate."""
        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        conn = _in_memory_db()
        c = _minimal_candidate(
            predicate="pref.food",
            confidence=0.92,
            speaker_id="owner",
        )
        r = candidates.upsert_candidate(conn, **c)
        return conn, r.candidate_id

    def _make_conn_with_conflicted_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[Any, str]:
        """Return (conn, candidate_id) for a contradicted (blocked) candidate."""
        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        conn = _in_memory_db()
        c = _minimal_candidate(
            predicate="pref.diet",
            confidence=0.92,
            speaker_id="owner",
            conflict_with_fact_id="fact:existing",
            seeded_supersede_ok=False,
        )
        r = candidates.upsert_candidate(conn, **c)
        return conn, r.candidate_id

    def _fake_promotion_policy_v3_module(self) -> Any:
        import importlib
        return importlib.import_module("truth.promotion_policy_v3")

    def _corroborated_verification(self, candidate_id: str) -> Any:
        ppv3 = self._fake_promotion_policy_v3_module()
        return ppv3.VerificationRecord(
            candidate_id=candidate_id,
            verification_status="corroborated",
            evidence_source_ids=["m1"],
            verifier_version="vtest",
            verified_at="2026-01-01T00:00:00Z",
        )

    def _no_block(self, _: str) -> bool:
        return False

    def test_v3_gate_active_allows_auto_promote_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When v3 is enabled and the candidate is auto-promote eligible, all gates
        pass and the Neo4j write proceeds (promoted=True from fake driver)."""
        ppv3 = self._fake_promotion_policy_v3_module()

        conn, candidate_id = self._make_conn_with_auto_promote_candidate(monkeypatch)
        try:
            from tests.test_promotion_policy_v3_runtime import _FakeDriver
            driver = _FakeDriver(nodes_created=1)
            result = ppv3.promote_candidate(
                candidate_id=candidate_id,
                verification=self._corroborated_verification(candidate_id),
                hard_block_check=self._no_block,
                neo4j_driver=driver,
                candidates_conn=conn,
            )
            assert result["promoted"] is True, (
                f"Expected promoted=True, got reason={result.get('reason')!r}"
            )
        finally:
            conn.close()

    def test_v3_gate_blocks_contradicted_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When v3 is enabled and the candidate has a contradiction (no supersede_ok),
        the policy gate blocks promotion even if verification is corroborated."""
        ppv3 = self._fake_promotion_policy_v3_module()

        conn, candidate_id = self._make_conn_with_conflicted_candidate(monkeypatch)
        try:
            result = ppv3.promote_candidate(
                candidate_id=candidate_id,
                verification=self._corroborated_verification(candidate_id),
                hard_block_check=self._no_block,
                neo4j_driver=None,
                candidates_conn=conn,
            )
            assert result["promoted"] is False
            assert "v3_policy_gate" in result["reason"], (
                f"Expected policy gate reason, got {result['reason']!r}"
            )
        finally:
            conn.close()

    def test_v3_rollback_skips_policy_gate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When GRAPHITI_POLICY_V3_ENABLED=0, the policy gate is skipped entirely.
        A contradicted candidate that would be blocked by the gate should proceed
        to the verification check (and pass if corroborated)."""
        ppv3 = self._fake_promotion_policy_v3_module()

        # Build the conflicted candidate first (need flag on for DB write)
        conn, candidate_id = self._make_conn_with_conflicted_candidate(monkeypatch)
        try:
            # Now disable v3 — policy gate should be bypassed
            monkeypatch.setenv(candidates.POLICY_V3_ENABLED_ENV, "0")

            from tests.test_promotion_policy_v3_runtime import _FakeDriver
            driver = _FakeDriver(nodes_created=1)
            result = ppv3.promote_candidate(
                candidate_id=candidate_id,
                verification=self._corroborated_verification(candidate_id),
                hard_block_check=self._no_block,
                neo4j_driver=driver,
                candidates_conn=conn,
            )
            # Gate was skipped; verification passed; hard_block passed → promoted
            assert result["promoted"] is True, (
                f"Expected rollback path to promote, got reason={result.get('reason')!r}"
            )
            assert "v3_policy_gate" not in result.get("reason", "")
        finally:
            conn.close()

    def test_v3_gate_fail_closed_when_no_conn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When v3 is enabled, missing candidates_conn must fail closed."""
        ppv3 = self._fake_promotion_policy_v3_module()

        conn, candidate_id = self._make_conn_with_conflicted_candidate(monkeypatch)
        try:
            from tests.test_promotion_policy_v3_runtime import _FakeDriver
            driver = _FakeDriver(nodes_created=1)
            result = ppv3.promote_candidate(
                candidate_id=candidate_id,
                verification=self._corroborated_verification(candidate_id),
                hard_block_check=self._no_block,
                neo4j_driver=driver,
                candidates_conn=None,
            )
            assert result["promoted"] is False
            assert result["reason"] == "v3_policy_gate:candidates_conn_missing"
        finally:
            conn.close()

    def test_hard_block_still_blocks_after_v3_gate_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hard-block gate must remain fail-closed: even when the v3 policy gate
        passes, a hard-blocked candidate must not be promoted."""
        ppv3 = self._fake_promotion_policy_v3_module()

        conn, candidate_id = self._make_conn_with_auto_promote_candidate(monkeypatch)
        try:
            result = ppv3.promote_candidate(
                candidate_id=candidate_id,
                verification=self._corroborated_verification(candidate_id),
                hard_block_check=lambda _: True,  # always blocks
                neo4j_driver=None,
                candidates_conn=conn,
            )
            assert result["promoted"] is False
            assert result["reason"] == "hard_blocked"
        finally:
            conn.close()

    def test_hard_block_fail_closed_on_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If hard_block_check raises, the gate must fail closed (blocked=True)."""
        ppv3 = self._fake_promotion_policy_v3_module()

        conn, candidate_id = self._make_conn_with_auto_promote_candidate(monkeypatch)
        try:
            def _raise(_: str) -> bool:
                raise RuntimeError("hard_block_check unavailable")

            result = ppv3.promote_candidate(
                candidate_id=candidate_id,
                verification=self._corroborated_verification(candidate_id),
                hard_block_check=_raise,
                neo4j_driver=None,
                candidates_conn=conn,
            )
            assert result["promoted"] is False
            assert result["reason"] == "hard_blocked"
        finally:
            conn.close()

    def test_verification_not_corroborated_blocks_after_v3_gate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verification gate must still block if status != corroborated, even
        when the v3 policy gate passes."""
        ppv3 = self._fake_promotion_policy_v3_module()

        conn, candidate_id = self._make_conn_with_auto_promote_candidate(monkeypatch)
        try:
            unverified = ppv3.VerificationRecord(
                candidate_id=candidate_id,
                verification_status="pending",  # not corroborated
                evidence_source_ids=["m1"],
                verifier_version="vtest",
                verified_at="2026-01-01T00:00:00Z",
            )
            result = ppv3.promote_candidate(
                candidate_id=candidate_id,
                verification=unverified,
                hard_block_check=self._no_block,
                neo4j_driver=None,
                candidates_conn=conn,
            )
            assert result["promoted"] is False
            assert "verification_status=pending" in result["reason"]
        finally:
            conn.close()

    def test_supersede_gate_allows_via_v3_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A conflicted candidate with seeded_supersede_ok=True (auto_supersede
        recommendation) must be allowed through the v3 policy gate."""
        ppv3 = self._fake_promotion_policy_v3_module()

        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        conn = _in_memory_db()
        try:
            c = _minimal_candidate(
                predicate="pref.diet",
                confidence=0.92,
                speaker_id="owner",
                conflict_with_fact_id="fact:old-001",
                seeded_supersede_ok=True,
                evidence_refs=[
                    {"source_key": "sessions:s1", "evidence_id": "e1"},
                    {"source_key": "chatgpt:s2", "evidence_id": "e2"},
                ],
            )
            r = candidates.upsert_candidate(conn, **c)

            from tests.test_promotion_policy_v3_runtime import _FakeDriver
            driver = _FakeDriver(nodes_created=1)
            result = ppv3.promote_candidate(
                candidate_id=r.candidate_id,
                verification=self._corroborated_verification(r.candidate_id),
                hard_block_check=self._no_block,
                neo4j_driver=driver,
                candidates_conn=conn,
            )
            assert result["promoted"] is True, (
                f"Supersede gate should allow. reason={result.get('reason')!r}"
            )
        finally:
            conn.close()

    def test_lane_om_candidate_uses_same_decision_path_as_lane_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The unified contract guarantee: policy_allows_promotion() applied to
        an s1_observational_memory lane candidate produces the same decision as
        what upsert_candidate() would compute for an equivalent lane candidate.
        Both use compute_policy_trace() as the shared engine.
        """
        monkeypatch.delenv(candidates.POLICY_V3_ENABLED_ENV, raising=False)
        conn = _in_memory_db()
        try:
            # Simulate an OM lane candidate
            om_candidate = _minimal_candidate(
                predicate="pref.travel_style",
                confidence=0.92,
                speaker_id="owner",
                evidence_refs=[
                    {"source_key": "s1_observational_memory:chunk-01", "evidence_id": "om1"},
                ],
            )
            r = candidates.upsert_candidate(conn, **om_candidate)
            # The status_suggested from upsert should be auto_promoted (lane path)
            assert r.status == "auto_promoted", (
                f"Lane path expected auto_promoted, got {r.status!r}"
            )
            # policy_allows_promotion (OM path) must agree with the lane path
            allowed, reason = candidates.policy_allows_promotion(r.candidate_id, conn=conn)
            assert allowed is True, (
                f"OM policy gate must agree with lane path. reason={reason!r}"
            )
        finally:
            conn.close()
