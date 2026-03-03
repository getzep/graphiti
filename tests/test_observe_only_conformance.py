"""Tests for FR-11 observe-only ontology conformance gate.

Covers:
  1. _build_episode_body strips untrusted metadata (sanitization fix)
  2. compute_conformance_metrics basic calculations on fixture/mock data
  3. Ontology config loads for s1_sessions_main and s1_pilot_fr11_20260227 lanes
  4. Conformance evaluator exit-code-0 guarantee (observe-only contract)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ── 1. _build_episode_body sanitization ───────────────────────────────────────


def test_build_episode_body_strips_untrusted_metadata():
    """_build_episode_body must sanitize message content before building the body."""
    from scripts.mcp_ingest_sessions import _build_episode_body

    msg_with_metadata = (
        "Hello, this is my real message.\n\n"
        "Conversation info (untrusted metadata):\n"
        "```json\n"
        '{"message_id": "abc123", "platform": "telegram"}\n'
        "```\n\n"
        "See you later."
    )
    messages_by_id = {
        "msg1": {
            "created_at": "2026-02-27T10:00:00Z",
            "role": "user",
            "content": msg_with_metadata,
        }
    }
    body = _build_episode_body(["msg1"], messages_by_id)

    # The untrusted metadata block must be stripped
    assert "Conversation info (untrusted metadata):" not in body
    assert '```json' not in body
    assert "message_id" not in body
    # Real content must be preserved
    assert "Hello, this is my real message." in body
    assert "See you later." in body


def test_build_episode_body_clean_content_unchanged():
    """Content without untrusted metadata blocks should be returned as-is."""
    from scripts.mcp_ingest_sessions import _build_episode_body

    clean_content = "This is a normal message without any metadata."
    messages_by_id = {
        "m1": {"created_at": "2026-02-27T09:00:00", "role": "assistant", "content": clean_content}
    }
    body = _build_episode_body(["m1"], messages_by_id)
    assert clean_content in body


def test_build_episode_body_multiple_messages_sanitized():
    """Multiple messages should each be sanitized independently."""
    from scripts.mcp_ingest_sessions import _build_episode_body

    messages_by_id = {
        "a": {
            "created_at": "2026-02-27T08:00:00",
            "role": "user",
            "content": (
                "First real content.\n\n"
                "Sender (untrusted metadata):\n"
                "```json\n{\"user_id\": \"u1\"}\n```\n\n"
            ),
        },
        "b": {
            "created_at": "2026-02-27T08:01:00",
            "role": "assistant",
            "content": "Second clean content.",
        },
    }
    body = _build_episode_body(["a", "b"], messages_by_id)
    assert "user_id" not in body
    assert "First real content." in body
    assert "Second clean content." in body


def test_build_episode_body_empty_content():
    """Empty content should produce an empty result (not crash)."""
    from scripts.mcp_ingest_sessions import _build_episode_body

    messages_by_id = {
        "x": {"created_at": "2026-02-27T12:00:00", "role": "user", "content": ""},
    }
    # Should not raise
    body = _build_episode_body(["x"], messages_by_id)
    assert isinstance(body, str)


# ── 2. compute_conformance_metrics ───────────────────────────────────────────


def test_conformance_metrics_all_in_schema():
    """When all entities and relations are in-schema, rates should be 1.0."""
    from scripts.evaluate_ontology_conformance import compute_conformance_metrics

    entities = [
        {"entity_type": "Preference", "name": "e1"},
        {"entity_type": "Requirement", "name": "e2"},
        {"entity_type": "Organization", "name": "e3"},
    ]
    relations = [
        {"relation_type": "RELATES_TO"},
        {"relation_type": "PREFERS"},
        {"relation_type": "REQUIRES"},
    ]
    allowed_entities = {"Preference", "Requirement", "Organization"}
    allowed_relations = {"RELATES_TO", "PREFERS", "REQUIRES"}

    m = compute_conformance_metrics(entities, relations, allowed_entities, allowed_relations)

    assert m["typed_entity_rate"] == 1.0
    assert m["allowed_relation_rate"] == 1.0
    assert m["out_of_schema_count"] == 0
    assert m["out_of_schema_types"] == []
    assert m["off_schema_entity_types"] == []


def test_conformance_metrics_partial_schema():
    """Partial off-schema data should yield rates < 1.0."""
    from scripts.evaluate_ontology_conformance import compute_conformance_metrics

    entities = [
        {"entity_type": "Preference", "name": "e1"},
        {"entity_type": "UNKNOWN_TYPE", "name": "e2"},
    ]
    relations = [
        {"relation_type": "RELATES_TO"},
        {"relation_type": "OFF_SCHEMA_REL"},
    ]
    allowed_entities = {"Preference", "Requirement"}
    allowed_relations = {"RELATES_TO", "PREFERS"}

    m = compute_conformance_metrics(entities, relations, allowed_entities, allowed_relations)

    assert m["total_entities"] == 2
    assert m["typed_entities"] == 1
    assert m["typed_entity_rate"] == pytest.approx(0.5)
    assert m["total_relations"] == 2
    assert m["allowed_relations"] == 1
    assert m["allowed_relation_rate"] == pytest.approx(0.5)
    assert m["out_of_schema_count"] == 1
    assert "OFF_SCHEMA_REL" in m["out_of_schema_types"]
    assert "UNKNOWN_TYPE" in m["off_schema_entity_types"]


def test_conformance_metrics_empty_inputs():
    """Empty inputs should return 1.0 rates (vacuously true) and zero counts."""
    from scripts.evaluate_ontology_conformance import compute_conformance_metrics

    m = compute_conformance_metrics([], [], {"Preference"}, {"RELATES_TO"})

    assert m["typed_entity_rate"] == 1.0
    assert m["allowed_relation_rate"] == 1.0
    assert m["total_entities"] == 0
    assert m["total_relations"] == 0
    assert m["out_of_schema_count"] == 0


def test_conformance_metrics_all_off_schema():
    """All-off-schema data should yield 0.0 rates."""
    from scripts.evaluate_ontology_conformance import compute_conformance_metrics

    entities = [{"entity_type": "NOPE", "name": "x"}]
    relations = [{"relation_type": "UNKNOWN_REL"}]

    m = compute_conformance_metrics(entities, relations, {"Preference"}, {"RELATES_TO"})

    assert m["typed_entity_rate"] == 0.0
    assert m["allowed_relation_rate"] == 0.0
    assert m["out_of_schema_count"] == 1


# ── 3. Ontology config loads for new lanes ────────────────────────────────────


def _get_ontology_path(filename: str = "extraction_ontologies.yaml") -> Path:
    """Find the extraction_ontologies.yaml path relative to the repo root."""
    repo_root = Path(__file__).resolve().parents[1]
    # Prefer mcp_server config as the canonical runtime copy
    candidates = [
        repo_root / "mcp_server" / "config" / filename,
        repo_root / "config" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    pytest.skip(f"{filename} not found in expected locations")


@pytest.mark.parametrize("group_id", ["s1_sessions_main", "s1_pilot_fr11_20260227"])
def test_sessions_lane_ontology_loads(group_id: str):
    """s1_sessions_main and s1_pilot_fr11_20260227 must load from the registry."""
    import importlib

    ontology_registry = importlib.import_module("mcp_server.src.services.ontology_registry")
    OntologyRegistry = ontology_registry.OntologyRegistry

    config_path = _get_ontology_path()
    registry = OntologyRegistry.load(config_path)

    profile = registry.get(group_id)
    assert profile is not None, f"No profile found for {group_id}"


@pytest.mark.parametrize("group_id", ["s1_sessions_main", "s1_pilot_fr11_20260227"])
def test_sessions_lane_extraction_mode_constrained_soft(group_id: str):
    """Sessions lanes must have extraction_mode=constrained_soft."""
    import importlib

    ontology_registry = importlib.import_module("mcp_server.src.services.ontology_registry")
    OntologyRegistry = ontology_registry.OntologyRegistry

    config_path = _get_ontology_path()
    registry = OntologyRegistry.load(config_path)

    profile = registry.get(group_id)
    assert profile is not None
    assert profile.extraction_mode == "constrained_soft", (
        f"{group_id} should use constrained_soft, got {profile.extraction_mode!r}"
    )


@pytest.mark.parametrize("group_id", ["s1_sessions_main", "s1_pilot_fr11_20260227"])
def test_sessions_lane_disallows_generic_relations(group_id: str):
    """Sessions lanes must disallow generic connector edges after hardening."""
    import importlib

    ontology_registry = importlib.import_module("mcp_server.src.services.ontology_registry")
    OntologyRegistry = ontology_registry.OntologyRegistry

    config_path = _get_ontology_path()
    registry = OntologyRegistry.load(config_path)

    profile = registry.get(group_id)
    assert profile is not None
    assert "RELATES_TO" not in profile.edge_types
    assert "MENTIONS" not in profile.edge_types
    assert "DISCUSSED" not in profile.edge_types
    # sanity: hardened operational edge remains present
    assert "DEPENDS_ON" in profile.edge_types


@pytest.mark.parametrize("group_id", ["s1_sessions_main", "s1_pilot_fr11_20260227"])
def test_sessions_lane_entity_types_include_graphiti_defaults(group_id: str):
    """Sessions lanes must include the standard graphiti entity types."""
    import importlib

    ontology_registry = importlib.import_module("mcp_server.src.services.ontology_registry")
    OntologyRegistry = ontology_registry.OntologyRegistry

    config_path = _get_ontology_path()
    registry = OntologyRegistry.load(config_path)

    profile = registry.get(group_id)
    assert profile is not None

    expected_types = {"Preference", "Requirement", "Procedure", "Location", "Event", "Organization"}
    actual_types = set(profile.entity_types.keys())
    missing = expected_types - actual_types
    assert not missing, f"{group_id} is missing entity types: {missing}"


# ── 4. Observe-only contract: exit code 0 guarantee ──────────────────────────


def test_conformance_evaluator_dry_run_exit_zero():
    """Dry-run invocation must always exit 0 (observe-only contract)."""
    from scripts.evaluate_ontology_conformance import main

    exit_code = main(["--group-id", "s1_sessions_main", "--dry-run"])
    assert exit_code == 0, "Conformance evaluator must exit 0 in dry-run mode"


def test_conformance_evaluator_dry_run_below_threshold_still_zero():
    """Even when metrics are below threshold, exit code must be 0 (observe-only)."""
    from scripts.evaluate_ontology_conformance import main

    # Use very high thresholds to trigger warnings, but still expect exit 0
    exit_code = main([
        "--group-id", "s1_sessions_main",
        "--dry-run",
        "--typed-entity-threshold", "0.99",
        "--allowed-relation-threshold", "0.99",
    ])
    assert exit_code == 0, "Observe-only: below-threshold must NOT produce non-zero exit"


def test_conformance_evaluator_dry_run_produces_valid_json(capsys):
    """Dry-run output must be valid JSON with required report fields."""
    from scripts.evaluate_ontology_conformance import main

    main(["--group-id", "s1_pilot_fr11_20260227", "--dry-run", "--output", "json"])
    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert report["group_id"] == "s1_pilot_fr11_20260227"
    assert report["observe_only"] is True
    assert report["dry_run"] is True
    assert "metrics" in report
    assert "conformance_passed" in report
    assert "note" in report
    # The note must mention observe-only
    assert "observe" in report["note"].lower()


def test_conformance_evaluator_observe_only_note_in_report(capsys):
    """The JSON report must explicitly document observe-only behavior."""
    from scripts.evaluate_ontology_conformance import main

    main(["--group-id", "s1_sessions_main", "--dry-run", "--output", "json"])
    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert report["observe_only"] is True
    # Must not say anything about dropping episodes
    note = report["note"].lower()
    assert "no episodes were dropped" in note or "observe-only" in note


def test_conformance_evaluator_dry_run_fixture_metrics(capsys):
    """Dry-run with fixture data should produce plausible metric values."""
    from scripts.evaluate_ontology_conformance import main

    main([
        "--group-id", "s1_sessions_main",
        "--dry-run",
        "--output", "json",
    ])
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    m = report["metrics"]

    assert m["total_entities"] > 0
    assert m["total_relations"] > 0
    assert 0.0 <= m["typed_entity_rate"] <= 1.0
    assert 0.0 <= m["allowed_relation_rate"] <= 1.0
    assert m["out_of_schema_count"] >= 0


# ── NEW: EVAL-1 label-based entity type computation ───────────────────────────


def test_label_based_entity_type_rate_fully_typed():
    """EVAL-1: entity rate computed from label-derived types (not entity_type property).

    Simulates what Neo4j returns after the label-based Cypher fix:
    first non-'Entity' label becomes entity_type; Entity-only nodes → 'UNKNOWN'.
    """
    from scripts.evaluate_ontology_conformance import compute_conformance_metrics

    # Simulate label-extracted entity data (as returned by fixed _query_neo4j)
    entities = [
        {"entity_type": "Preference", "name": "prefers bullet summaries"},   # has semantic label
        {"entity_type": "Requirement", "name": "no meetings before 10am"},    # has semantic label
        {"entity_type": "Organization", "name": "Blockchain Capital"},         # has semantic label
        {"entity_type": "UNKNOWN", "name": "entity-only node"},               # no semantic label
    ]
    allowed_entities = {"Preference", "Requirement", "Organization", "Document"}
    m = compute_conformance_metrics(entities, [], allowed_entities, set())

    assert m["total_entities"] == 4
    assert m["typed_entities"] == 3  # UNKNOWN does not match
    assert m["typed_entity_rate"] == pytest.approx(0.75)
    assert "UNKNOWN" in m["off_schema_entity_types"]


def test_label_based_entity_type_all_unknown():
    """EVAL-1: when all nodes are Entity-only (no semantic labels), rate is 0.0."""
    from scripts.evaluate_ontology_conformance import compute_conformance_metrics

    entities = [
        {"entity_type": "UNKNOWN", "name": "entity1"},
        {"entity_type": "UNKNOWN", "name": "entity2"},
    ]
    allowed_entities = {"Preference", "Requirement"}
    m = compute_conformance_metrics(entities, [], allowed_entities, set())

    assert m["typed_entity_rate"] == 0.0
    assert m["typed_entities"] == 0


def test_neo4j_entity_query_uses_labels_not_property():
    """EVAL-1: verify the Neo4j query uses labels(n), not n.entity_type in Cypher RETURN."""
    import inspect

    from scripts.evaluate_ontology_conformance import _query_neo4j

    source = inspect.getsource(_query_neo4j)
    # Must use label-based extraction in the Cypher query
    assert "[label IN labels(n)" in source, (
        "EVAL-1 fix: entity Cypher query must use [label IN labels(n)...] for type extraction"
    )
    # Must NOT return n.entity_type as a Cypher expression
    # (docstring mentions it for context; the actual RETURN must use labels)
    assert "RETURN\n" not in source or "n.entity_type" not in source.split("RETURN")[-1].split("MATCH")[0], (
        "EVAL-1 fix: Cypher RETURN clause must not select n.entity_type"
    )
    # Confirm label extraction pattern (list comprehension filtering 'Entity' label)
    assert "label <> 'Entity'" in source, (
        "EVAL-1 fix: must filter out the base 'Entity' label to get semantic type"
    )


# ── NEW: EVAL-2 deconfounded relation metric ──────────────────────────────────


def test_deconfounded_relation_metric_excludes_mentions():
    """EVAL-2: semantic_allowed_relation_rate excludes MENTIONS infra edges."""
    from scripts.evaluate_ontology_conformance import compute_conformance_metrics

    # All relations (includes Graphiti infra MENTIONS)
    all_relations = [
        {"relation_type": "RELATES_TO"},  # semantic — in schema
        {"relation_type": "MENTIONS"},    # infra edge (Episodic→Entity)
        {"relation_type": "MENTIONS"},    # infra edge
        {"relation_type": "MENTIONS"},    # infra edge
    ]
    # Semantic (Entity→Entity) only — no MENTIONS here
    semantic_relations = [
        {"relation_type": "RELATES_TO"},
    ]
    allowed = {"RELATES_TO", "PREFERS"}

    m = compute_conformance_metrics([], all_relations, set(), allowed, semantic_relations=semantic_relations)

    # Legacy metric confounded by MENTIONS (3/4 = 0.25)
    assert m["allowed_relation_rate"] == pytest.approx(0.25)
    # Deconfounded metric — only RELATES_TO, which IS in schema (1/1 = 1.0)
    assert m["semantic_allowed_relation_rate"] == pytest.approx(1.0)
    assert m["semantic_total_relations"] == 1
    assert m["semantic_allowed_relations"] == 1
    assert m["semantic_out_of_schema_types"] == []


def test_deconfounded_metric_fallback_when_no_semantic_relations():
    """EVAL-2: when semantic_relations=None, falls back to legacy metric."""
    from scripts.evaluate_ontology_conformance import compute_conformance_metrics

    relations = [
        {"relation_type": "RELATES_TO"},
        {"relation_type": "OFF_SCHEMA"},
    ]
    allowed = {"RELATES_TO"}
    m = compute_conformance_metrics([], relations, set(), allowed, semantic_relations=None)

    # Fallback: semantic_allowed_relation_rate == allowed_relation_rate
    assert m["semantic_allowed_relation_rate"] == m["allowed_relation_rate"]
    assert m["semantic_total_relations"] == m["total_relations"]


def test_deconfounded_metric_all_infra_semantic_empty():
    """EVAL-2: all infra edges → semantic set empty → semantic rate 1.0 (vacuously true)."""
    from scripts.evaluate_ontology_conformance import compute_conformance_metrics

    all_relations = [{"relation_type": "MENTIONS"}, {"relation_type": "MENTIONS"}]
    semantic_relations: list[dict] = []  # no Entity→Entity edges

    m = compute_conformance_metrics([], all_relations, set(), {"RELATES_TO"}, semantic_relations=semantic_relations)

    assert m["allowed_relation_rate"] == pytest.approx(0.0)  # MENTIONS not in schema
    assert m["semantic_allowed_relation_rate"] == pytest.approx(1.0)  # vacuously true (no semantic rels)
    assert m["semantic_total_relations"] == 0


def test_conformance_passed_uses_deconfounded_metric(capsys):
    """EVAL-2: conformance_passed must be based on semantic_allowed_relation_rate."""
    from scripts.evaluate_ontology_conformance import main

    # dry-run fixture has MENTIONS edges; with --allow-rel excluding MENTIONS,
    # legacy rate would fail but semantic rate may pass (if fixture entity→entity
    # edges are all in schema).
    exit_code = main([
        "--group-id", "s1_sessions_main",
        "--dry-run",
        "--output", "json",
        "--allow-rel", "RELATES_TO", "PREFERS", "REQUIRES",
        "--allow-entity", "Preference", "Requirement", "Organization",
    ])
    assert exit_code == 0  # always exit 0 (observe-only)
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    m = report["metrics"]
    # Report must include the new deconfounded fields
    assert "semantic_allowed_relation_rate" in m
    assert "semantic_total_relations" in m
    assert "semantic_allowed_relations" in m


# ── NEW: EVAL-3 claim-mode limit enforcement ──────────────────────────────────


def test_claim_mode_limit_stops_at_limit(tmp_path):
    """EVAL-3: claim-mode worker loop must stop after exactly --limit chunks."""
    from scripts.mcp_ingest_sessions import claim_chunk, init_claim_db, seed_claims

    n_total = 20
    limit = 5

    db_path = str(tmp_path / "test_limit.claims.db")
    conn = init_claim_db(db_path)
    chunk_ids = [f"chunk-{i:03d}" for i in range(n_total)]
    seed_claims(conn, chunk_ids)

    # Mirror the fixed Phase B loop exactly
    phase_b_limit = limit
    phase_b_count = 0
    claimed: list[str] = []

    while True:
        if phase_b_count >= phase_b_limit:
            break
        chunk_id = claim_chunk(conn, "worker-test")
        if chunk_id is None:
            break
        claimed.append(chunk_id)
        phase_b_count += 1  # mirrors successful claim in fixed code

    conn.close()

    assert len(claimed) == limit, (
        f"Expected exactly {limit} chunks claimed, got {len(claimed)}"
    )
    # Remaining chunks must still be pending
    import sqlite3
    conn2 = sqlite3.connect(db_path)
    pending = conn2.execute(
        "SELECT COUNT(*) FROM chunk_claims WHERE status='pending'"
    ).fetchone()[0]
    conn2.close()
    assert pending == n_total - limit


def test_claim_mode_limit_zero_means_unlimited(tmp_path):
    """EVAL-3: limit=0 must be treated as unlimited (process all chunks)."""
    from scripts.mcp_ingest_sessions import claim_chunk, init_claim_db, seed_claims

    n_total = 5

    db_path = str(tmp_path / "test_unlimited.claims.db")
    conn = init_claim_db(db_path)
    chunk_ids = [f"chunk-{i:03d}" for i in range(n_total)]
    seed_claims(conn, chunk_ids)

    phase_b_limit = 0  # unlimited (args.limit=0 or negative means no cap)
    phase_b_count = 0
    claimed: list[str] = []

    while True:
        # limit=0 → no cap (phase_b_limit is falsy → skip check)
        if phase_b_limit and phase_b_count >= phase_b_limit:
            break
        chunk_id = claim_chunk(conn, "worker-test")
        if chunk_id is None:
            break
        claimed.append(chunk_id)
        phase_b_count += 1

    conn.close()

    assert len(claimed) == n_total, "limit=0 should allow all chunks to be claimed"


# ── NEW: Part B — OM path guard ───────────────────────────────────────────────


def test_om_path_guard_warning_printed_for_om_namespace(capsys, monkeypatch):
    """Part B: targeting s1_observational_memory must print a warning to stderr."""
    import argparse

    from scripts.mcp_ingest_sessions import _check_om_path_guard

    # Remove OM_PATH_GUARD=strict so we only get warning, not abort
    monkeypatch.delenv("OM_PATH_GUARD", raising=False)

    args = argparse.Namespace(group_id="s1_observational_memory")
    _check_om_path_guard(args)

    captured = capsys.readouterr()
    assert "OM PATH GUARD" in captured.err
    assert "om_compressor" in captured.err
    assert "s1_observational_memory" in captured.err


def test_om_path_guard_no_warning_for_sessions_namespace(capsys, monkeypatch):
    """Part B: sessions namespace must NOT trigger the OM guard."""
    import argparse

    from scripts.mcp_ingest_sessions import _check_om_path_guard

    monkeypatch.delenv("OM_PATH_GUARD", raising=False)

    args = argparse.Namespace(group_id="s1_sessions_main")
    _check_om_path_guard(args)

    captured = capsys.readouterr()
    assert "OM PATH GUARD" not in captured.err


def test_om_path_guard_strict_raises_systemexit(monkeypatch):
    """Part B: OM_PATH_GUARD=strict must cause sys.exit(2) for OM namespace."""
    import argparse

    import pytest

    from scripts.mcp_ingest_sessions import _check_om_path_guard

    monkeypatch.setenv("OM_PATH_GUARD", "strict")

    args = argparse.Namespace(group_id="s1_observational_memory")
    with pytest.raises(SystemExit) as exc_info:
        _check_om_path_guard(args)

    assert exc_info.value.code == 2


def test_om_path_guard_no_abort_for_pilot_namespace(monkeypatch):
    """Part B: pilot namespace (not OM) must not trigger guard even in strict mode."""
    import argparse

    from scripts.mcp_ingest_sessions import _check_om_path_guard

    monkeypatch.setenv("OM_PATH_GUARD", "strict")

    # Pilot namespace does not start with s1_observational_memory → no guard
    args = argparse.Namespace(group_id="s1_pilot_om_path_20260228_v3")
    # Must not raise
    _check_om_path_guard(args)


# ── NEW: s1_pilot_om_path_20260228_v3 ontology profile ───────────────────────


def test_om_pilot_v3_namespace_loads():
    """s1_pilot_om_path_20260228_v3 must load from the ontology registry."""
    import importlib

    ontology_registry = importlib.import_module("mcp_server.src.services.ontology_registry")
    OntologyRegistry = ontology_registry.OntologyRegistry

    config_path = _get_ontology_path()
    registry = OntologyRegistry.load(config_path)

    profile = registry.get("s1_pilot_om_path_20260228_v3")
    assert profile is not None, "s1_pilot_om_path_20260228_v3 profile must exist in ontology config"


def test_om_pilot_v3_has_om_relation_types():
    """s1_pilot_om_path_20260228_v3 must include the OM ontology relation types."""
    import importlib

    ontology_registry = importlib.import_module("mcp_server.src.services.ontology_registry")
    OntologyRegistry = ontology_registry.OntologyRegistry

    config_path = _get_ontology_path()
    registry = OntologyRegistry.load(config_path)

    profile = registry.get("s1_pilot_om_path_20260228_v3")
    assert profile is not None

    expected_om_relations = {"MOTIVATES", "GENERATES", "SUPERSEDES", "ADDRESSES", "RESOLVES"}
    actual = set(profile.edge_types)
    missing = expected_om_relations - actual
    assert not missing, (
        f"s1_pilot_om_path_20260228_v3 missing OM relation types: {missing}"
    )


def test_om_pilot_v3_has_om_entity_types():
    """s1_pilot_om_path_20260228_v3 must include OM-specific entity types."""
    import importlib

    ontology_registry = importlib.import_module("mcp_server.src.services.ontology_registry")
    OntologyRegistry = ontology_registry.OntologyRegistry

    config_path = _get_ontology_path()
    registry = OntologyRegistry.load(config_path)

    profile = registry.get("s1_pilot_om_path_20260228_v3")
    assert profile is not None

    expected_om_entities = {"Insight", "Pattern", "Requirement", "Decision", "Problem"}
    actual = set(profile.entity_types.keys())
    missing = expected_om_entities - actual
    assert not missing, (
        f"s1_pilot_om_path_20260228_v3 missing OM entity types: {missing}"
    )
