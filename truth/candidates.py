"""Derived candidates queue + policy trace scaffolding.

Implements the SQLite candidates DB described in:
- tools/graphiti/prd/CANDIDATES-APPROVALS-SPEC.md
- tools/graphiti/prd/PROMOTION-TAXONOMY-v1.md
- tools/graphiti/prd/BOOTSTRAP-IMPORT-SPEC.md

The candidates DB is *derived* (rebuildable) and lives at:
  tools/graphiti/state/candidates.db

This module is intentionally stdlib-only.

Integration points that are intentionally stubbed (documented in policy_trace_json):
- Conflict detection against the canonical Fact Ledger (active fact lookup)
- Writing Fact Ledger events (SUPERSEDE/EXPIRE)
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH_DEFAULT = Path(__file__).resolve().parents[1] / "state" / "candidates.db"

# ─────────────────────────────────────────────────────────────────────────────
# Policy version constants (EXEC-UNIFIED-V3-AFTERNOON PRD)
# ─────────────────────────────────────────────────────────────────────────────

# Feature flag: set GRAPHITI_POLICY_V3_ENABLED=0 to keep v2 for existing ops.
# Defaults to True; flip off only if a rollback is needed.
POLICY_V3_ENABLED_ENV = "GRAPHITI_POLICY_V3_ENABLED"

# Compat constant: accepted on reads from existing rows; never written for new rows.
POLICY_VERSION_V2 = "promotion-v2"

# V3 label: written for all new candidates when GRAPHITI_POLICY_V3_ENABLED is truthy.
POLICY_VERSION_V3 = "promotion-v3"

# Public alias used throughout this module — resolves to v3 when enabled.
# Existing rows with "promotion-v2" remain valid (backward-compatible read path).
POLICY_VERSION_DEFAULT = POLICY_VERSION_V3  # v3 is now the default (EXEC-UNIFIED-V3-AFTERNOON)

# Thresholds (PROMOTION-TAXONOMY-v1.md)
RECOMMEND_THRESHOLD = 0.80
AUTO_PROMOTE_THRESHOLD = 0.90
EXPLICIT_UPDATE_MIN_CONFIDENCE = 0.80
CORROBORATION_BOOST_PER_SOURCE_FAMILY = 0.02
CORROBORATION_CAPS = {
    "low": 0.06,
    "medium": 0.04,
    "high": 0.00,
}

SENSITIVE_CONTENT_CUES = (
    "ssn",
    "social security",
    "password",
    "private key",
    "medical",
    "diagnosis",
)

CORROBORATION_KEYS = {
    "independent_source_families",
    "same_lineage_repeats",
}

OWNER_SPEAKER_IDS_ENV = "GRAPHITI_OWNER_SPEAKER_IDS"
AUTO_PROMOTE_ENABLED_ENV = "GRAPHITI_AUTO_PROMOTE_ENABLED"
OWNER_SPEAKER_IDS_DEFAULT = (
    "owner",
    "tg:1439681712",
)

CORE_TRUTH_PREFIXES = (
    "identity.",
    "relationship.",
    "legal.",
    "finance.",
    "security.",
    "health.",
)

LOW_RISK_PREFIXES = (
    "pref.",
    "style.",
)

ELIGIBLE_ASSERTION_TYPES = {
    "decision",
    "preference",
    "factual_assertion",
    "episode",
}

VERIFICATION_STATUSES = {
    "pending",
    "corroborated",
    "contradicted",
    "insufficient_evidence",
}

# ─────────────────────────────────────────────────────────────────────────────
# V3 lane policy contract (EXEC-UNIFIED-V3-AFTERNOON PRD, Locked Policy Matrix)
# Callers (graph_maintenance, import_graphiti_candidates, pack router) should
# reference these constants rather than hardcoding group-id sets.
# ─────────────────────────────────────────────────────────────────────────────

# Lanes eligible for direct retrieval across all runtime packs.
LANE_RETRIEVAL_ELIGIBLE_GLOBAL: frozenset[str] = frozenset({
    "s1_sessions_main",
    "s1_observational_memory",
})

# Lanes eligible for retrieval only in VC-scoped packs.
LANE_RETRIEVAL_ELIGIBLE_VC_SCOPED: frozenset[str] = frozenset({
    "s1_chatgpt_history",
})

# Lanes that are corroboration-only by default (not direct retrieval sources).
LANE_CORROBORATION_ONLY: frozenset[str] = frozenset({
    "s1_curated_refs",
    "s1_memory_day1",
})

# All lanes that may generate candidates (used by import_graphiti_candidates).
LANE_CANDIDATES_ELIGIBLE: frozenset[str] = (
    LANE_RETRIEVAL_ELIGIBLE_GLOBAL
    | LANE_RETRIEVAL_ELIGIBLE_VC_SCOPED
    | frozenset({"s1_memory_day1"})  # corroboration-only but still candidate-generating
)


class IneligibleAssertionTypeError(ValueError):
    """Raised when an assertion_type is not in ELIGIBLE_ASSERTION_TYPES.

    Hard gate: prevents ineligible types from entering candidates.db entirely.
    """

    def __init__(self, assertion_type: str) -> None:
        self.assertion_type = assertion_type
        super().__init__(
            f"Assertion type {assertion_type!r} is not eligible for candidates. "
            f"Allowed types: {sorted(ELIGIBLE_ASSERTION_TYPES)}"
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_c14n(obj: Any) -> str:
    """Deterministic JSON serialization (canonical-ish)."""

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# -----------------------------
# ULID (stdlib-only)
# -----------------------------

_CROCKFORD32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _encode_crockford_base32(data: bytes) -> str:
    n = int.from_bytes(data, "big")
    chars: list[str] = []
    for _ in range(26):
        n, rem = divmod(n, 32)
        chars.append(_CROCKFORD32[rem])
    return "".join(reversed(chars))


def new_ulid() -> str:
    """Generate a ULID-like identifier (26 Crockford base32 chars)."""

    ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    ts = ts_ms.to_bytes(6, "big", signed=False)  # 48-bit timestamp
    rand = os.urandom(10)  # 80-bit randomness
    return _encode_crockford_base32(ts + rand)


# -----------------------------
# Evidence refs / stats
# -----------------------------


def _parse_iso8601(s: str) -> datetime | None:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s[:-1]).replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def normalize_evidence_ref(ref: dict[str, Any]) -> dict[str, Any]:
    """Normalize evidence refs for stable storage and dedup."""

    stable_keys = {
        "source_key",
        "chunk_key",
        "evidence_id",
        "start_id",
        "end_id",
        "timestamp_range",
        "observed_at",
        "scope",
    }

    out: dict[str, Any] = {}
    extra: dict[str, Any] = {}

    for k, v in (ref or {}).items():
        if k in stable_keys:
            out[k] = v
        else:
            extra[k] = v

    if extra:
        out["extra"] = extra

    # Drop null-ish values for stability
    out = {k: v for k, v in out.items() if v not in (None, "", [], {})}

    return out


def evidence_ref_key(ref: dict[str, Any]) -> str:
    nref = normalize_evidence_ref(ref)
    if "evidence_id" in nref:
        return f"evidence_id:{nref['evidence_id']}"
    return sha256_hex(json_c14n(nref))


def _source_family_from_ref(ref: dict[str, Any]) -> str | None:
    family = ref.get("source_family")
    if isinstance(family, str):
        normalized = family.strip().lower()
        if normalized:
            return normalized

    source_key = ref.get("source_key")
    if not isinstance(source_key, str):
        return None
    source_key = source_key.strip()
    if not source_key:
        return None
    family = source_key.split(":", 1)[0].strip().lower()
    return family or None


def merge_evidence_refs(existing: Iterable[dict[str, Any]], incoming: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Merge + dedup evidence refs. Returns (merged, added_count)."""

    existing_list = [normalize_evidence_ref(r) for r in (existing or []) if r]
    incoming_list = [normalize_evidence_ref(r) for r in (incoming or []) if r]

    seen = {evidence_ref_key(r) for r in existing_list}
    merged = list(existing_list)

    added = 0
    for r in incoming_list:
        k = evidence_ref_key(r)
        if k in seen:
            continue
        seen.add(k)
        merged.append(r)
        added += 1

    merged.sort(key=lambda r: json_c14n(r))
    return merged, added


def compute_evidence_stats(evidence_refs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute evidence stats per PROMOTION-TAXONOMY-v1."""

    refs = [normalize_evidence_ref(r) for r in (evidence_refs or [])]

    source_keys = sorted({r.get("source_key") for r in refs if r.get("source_key")})
    source_families = sorted(
        {sf for r in refs if (sf := _source_family_from_ref(r)) is not None}
    )
    source_family_counts: dict[str, int] = {}
    for ref in refs:
        sf = _source_family_from_ref(ref)
        if sf is None:
            continue
        source_family_counts[sf] = source_family_counts.get(sf, 0) + 1

    def ref_ts(r: dict[str, Any]) -> datetime | None:
        ts = None
        if r.get("observed_at"):
            ts = _parse_iso8601(str(r.get("observed_at")))
        if ts is None and r.get("timestamp_range"):
            tr = r.get("timestamp_range")
            if isinstance(tr, dict):
                ts = _parse_iso8601(str(tr.get("start") or tr.get("from") or ""))
            elif isinstance(tr, (list, tuple)) and tr:
                ts = _parse_iso8601(str(tr[0]))
        return ts

    dts: list[datetime] = []
    for r in refs:
        ts = ref_ts(r)
        if ts is not None:
            dts.append(ts)

    clusters: dict[tuple[str, str], int] = {}
    for r in refs:
        sk = r.get("source_key")
        if not sk:
            continue
        ts = ref_ts(r)
        if ts is None:
            bucket = "unknown"
        else:
            iso_year, iso_week, _ = ts.isocalendar()
            bucket = f"{iso_year:04d}-W{iso_week:02d}"
        clusters[(sk, bucket)] = clusters.get((sk, bucket), 0) + 1

    repeat_clusters = [
        {"source_key": sk, "iso_week_bucket": wk, "count": n}
        for (sk, wk), n in sorted(clusters.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    ]

    if len(dts) >= 2:
        span = max(dts) - min(dts)
        time_span_days: int | None = int(span.total_seconds() // 86400)
    else:
        time_span_days = None

    return {
        "distinct_source_keys": len(source_keys),
        "independent_source_families": len(source_families),
        "source_families": source_families,
        "source_keys": source_keys,
        "source_family_counts": source_family_counts,
        "repeat_clusters": repeat_clusters,
        "time_span_days": time_span_days,
        "evidence_ref_count": len(refs),
    }


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def policy_v3_enabled() -> bool:
    """Return True when v3 promotion policy is active (the default).

    Flip GRAPHITI_POLICY_V3_ENABLED=0 for an emergency rollback.
    New candidate rows will be stamped with POLICY_VERSION_V2 ("promotion-v2")
    when disabled, preserving compat with downstream readers.
    """
    return _env_bool(POLICY_V3_ENABLED_ENV, default=True)


def active_policy_version() -> str:
    """Return the policy version string to stamp on new candidate rows."""
    return POLICY_VERSION_V3 if policy_v3_enabled() else POLICY_VERSION_V2


def owner_speaker_ids_from_env() -> set[str]:
    raw = (os.environ.get(OWNER_SPEAKER_IDS_ENV) or "").strip()
    if not raw:
        return set(OWNER_SPEAKER_IDS_DEFAULT)

    parsed = {part.strip() for part in raw.split(",") if part.strip()}
    if not parsed:
        return set(OWNER_SPEAKER_IDS_DEFAULT)
    return parsed


def auto_promote_enabled_from_env() -> bool:
    return _env_bool(AUTO_PROMOTE_ENABLED_ENV, default=True)


# -----------------------------
# Policy trace
# -----------------------------


def classify_risk_level(predicate: str) -> str:
    p = (predicate or "").strip()
    for pref in CORE_TRUTH_PREFIXES:
        if p.startswith(pref):
            return "high"
    for pref in LOW_RISK_PREFIXES:
        if p.startswith(pref):
            return "low"
    return "medium"


def _coerce_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_content_cues(candidate: dict[str, Any]) -> list[str]:
    raw = candidate.get("content_cues")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw.strip().lower()]
    if not isinstance(raw, (list, tuple, set)):
        return []
    return [
        str(cue).strip().lower()
        for cue in raw
        if isinstance(cue, str) and str(cue).strip()
    ]


def _sensitive_cues_hit(cues: list[str]) -> list[str]:
    hits: list[str] = []
    for cue in cues:
        if any(term in cue for term in SENSITIVE_CONTENT_CUES):
            hits.append(cue)
    return hits


def compute_policy_trace(candidate: dict[str, Any], evidence_stats: dict[str, Any]) -> dict[str, Any]:
    assertion_type = (candidate.get("assertion_type") or "").strip()
    predicate = (candidate.get("predicate") or "").strip()
    confidence = _coerce_float(candidate.get("confidence"), default=0.0)
    speaker_id = (candidate.get("speaker_id") or "").strip()

    origin = candidate.get("origin") or "extracted"
    reason = candidate.get("reason") or origin

    explicit_update = bool(candidate.get("explicit_update") or False)
    conflict = bool(candidate.get("conflict") or candidate.get("conflict_with_fact_id"))
    seeded_supersede_ok = bool(candidate.get("seeded_supersede_ok"))
    content_cues = _extract_content_cues(candidate)
    cue_hits = _sensitive_cues_hit(content_cues)
    owner_speaker_ids = owner_speaker_ids_from_env()
    auto_promote_enabled = auto_promote_enabled_from_env()

    risk_level = classify_risk_level(predicate)
    effective_risk_level = "high" if cue_hits else risk_level
    low_risk_namespace = any(predicate.startswith(pref) for pref in LOW_RISK_PREFIXES)
    owner_speaker = bool(speaker_id) and speaker_id in owner_speaker_ids

    thresholds = {
        "recommend_threshold": RECOMMEND_THRESHOLD,
        "auto_promote_threshold": AUTO_PROMOTE_THRESHOLD,
        "explicit_update_min_confidence": EXPLICIT_UPDATE_MIN_CONFIDENCE,
        "corroboration_cap_low": CORROBORATION_CAPS["low"],
        "corroboration_cap_medium": CORROBORATION_CAPS["medium"],
        "corroboration_cap_high": CORROBORATION_CAPS["high"],
    }

    # Bootstrap candidates must never be auto-promoted (BOOTSTRAP-IMPORT-SPEC.md)
    auto_promotion_allowed = origin != "bootstrap"

    eligible_type = assertion_type in ELIGIBLE_ASSERTION_TYPES
    no_conflict = not conflict

    ttl_seconds = None
    if assertion_type == "episode":
        ttl_seconds = candidate.get("ttl_seconds")
        if ttl_seconds is None and isinstance(candidate.get("value"), dict):
            ttl_seconds = candidate.get("value", {}).get("ttl_seconds")

    needs_ttl = assertion_type == "episode" and not ttl_seconds

    stats = evidence_stats or {}
    independent_source_families = _coerce_int(stats.get("independent_source_families"), default=0)
    if independent_source_families <= 0:
        source_families = stats.get("source_families") or stats.get("source_keys") or []
        independent_source_families = _coerce_int(len(source_families), default=0)
    same_lineage_repeats = 0
    for item in stats.get("repeat_clusters", []) or []:
        if isinstance(item, dict):
            same_lineage_repeats += max(0, _coerce_int(item.get("count"), default=0) - 1)

    corroboration_cap = CORROBORATION_CAPS.get(effective_risk_level, CORROBORATION_CAPS["medium"])
    corroboration_boost = min(max(0, independent_source_families - 1) * CORROBORATION_BOOST_PER_SOURCE_FAMILY, corroboration_cap)
    if effective_risk_level == "high":
        corroboration_boost = 0.0
    effective_confidence = min(1.0, max(0.0, confidence + corroboration_boost))

    meets_auto_confidence = effective_confidence >= AUTO_PROMOTE_THRESHOLD
    meets_recommend_confidence = effective_confidence >= RECOMMEND_THRESHOLD
    meets_explicit_update_confidence = effective_confidence >= EXPLICIT_UPDATE_MIN_CONFIDENCE
    medium_requires_additional_evidence = independent_source_families >= 2

    auto_supersede_eligible = (
        conflict
        and seeded_supersede_ok
        and eligible_type
        and not needs_ttl
        and owner_speaker
        and risk_level in {"low", "medium"}
        and meets_auto_confidence
        and (risk_level != "medium" or medium_requires_additional_evidence)
        and meets_explicit_update_confidence
        and auto_promote_enabled
        and auto_promotion_allowed
    )

    checks = [
        {"check": "eligible_assertion_type", "pass": eligible_type, "assertion_type": assertion_type},
        {"check": "content_cue_escalation", "pass": not cue_hits, "matched_cues": cue_hits},
        {"check": "episode_requires_ttl", "pass": not needs_ttl, "ttl_seconds": ttl_seconds},
        {"check": "risk_level", "pass": True, "risk_level": effective_risk_level},
        {"check": "risk_level_escalation", "pass": not bool(cue_hits), "base_risk_level": risk_level},
        {"check": "low_risk_namespace", "pass": low_risk_namespace, "predicate": predicate},
        {"check": "confidence_at_or_above_auto_promote_threshold", "pass": meets_auto_confidence, "confidence": effective_confidence},
        {"check": "confidence_at_or_above_recommend_threshold", "pass": meets_recommend_confidence, "confidence": effective_confidence},
        {"check": "effective_corroboration_boost", "pass": corroboration_boost > 0.0, "boost": corroboration_boost},
        {"check": "corroboration_cap", "pass": corroboration_boost <= corroboration_cap, "cap": corroboration_cap},
        {
            "check": "owner_speaker_allowlist",
            "pass": owner_speaker,
            "speaker_id": speaker_id or None,
            "owner_speaker_ids": sorted(owner_speaker_ids),
        },
        {"check": "no_conflict", "pass": no_conflict, "conflict": conflict},
        {"check": "auto_promote_enabled", "pass": auto_promote_enabled, "env_var": AUTO_PROMOTE_ENABLED_ENV},
        {"check": "auto_promotion_allowed", "pass": auto_promotion_allowed, "origin": origin},
        {"check": "seeded_supersede_ok", "pass": seeded_supersede_ok, "seeded_supersede_ok": seeded_supersede_ok},
        {"check": "explicit_update", "pass": explicit_update, "explicit_update": explicit_update},
        {
            "check": "medium_requires_additional_evidence",
            "pass": risk_level != "medium" or medium_requires_additional_evidence,
            "independent_source_families": independent_source_families,
        },
        {"check": "independent_source_families", "pass": independent_source_families >= 1, "independent_source_families": independent_source_families},
    ]

    recommendation = "pending"
    status_suggested = "pending"
    auto_promote_eligible = (
        risk_level == "low"
        and low_risk_namespace
        and meets_auto_confidence
        and eligible_type
        and owner_speaker
        and no_conflict
        and auto_promote_enabled
        and auto_promotion_allowed
    ) or (
        risk_level == "medium"
        and meets_auto_confidence
        and medium_requires_additional_evidence
        and eligible_type
        and owner_speaker
        and no_conflict
        and auto_promote_enabled
        and auto_promotion_allowed
    )

    if not eligible_type:
        recommendation = "blocked_ineligible"
        status_suggested = "denied"  # safe sink for ineligible items
    elif needs_ttl:
        recommendation = "needs_ttl"
        status_suggested = "requires_approval"
    elif effective_risk_level == "high" or not owner_speaker or not auto_promote_enabled:
        recommendation = "requires_approval"
        status_suggested = "requires_approval"
    elif conflict:
        if auto_supersede_eligible:
            recommendation = "auto_supersede"
            status_suggested = "auto_supersede"
        else:
            recommendation = "requires_approval"
            status_suggested = "requires_approval"
    elif risk_level == "medium":
        if auto_promote_eligible:
            recommendation = "auto_promote"
            status_suggested = "auto_promoted"
        elif explicit_update and meets_recommend_confidence:
            recommendation = "recommended_approve"
            status_suggested = "pending"
        else:
            recommendation = "requires_approval"
            status_suggested = "requires_approval"
    elif auto_promote_eligible:
        recommendation = "auto_promote"
        status_suggested = "auto_promoted"
    elif meets_recommend_confidence and explicit_update or meets_recommend_confidence:
        recommendation = "recommended_approve"
        status_suggested = "pending"
    else:
        recommendation = "pending"
        status_suggested = "pending"

    return {
        "policy_version": active_policy_version(),
        "evaluated_at": _now_iso(),
        "origin": origin,
        "reason": reason,
        "risk_level": effective_risk_level,
        "base_risk_level": risk_level,
        "risk_escalation": {
            "applied": bool(cue_hits),
            "matched_cues": cue_hits,
        },
        "assertion_type": assertion_type,
        "confidence": confidence,
        "evidence_confidence": confidence,
        "effective_confidence": effective_confidence,
        "explicit_update": explicit_update,
        "conflict": conflict,
        "seeded_supersede_ok": seeded_supersede_ok,
        "thresholds": thresholds,
        "evidence_stats": stats,
        "content_cues": content_cues,
        "corroboration": {
            "independent_source_families": independent_source_families,
            "same_lineage_repeats": same_lineage_repeats,
            "max_boost": corroboration_cap,
            "boost": corroboration_boost,
            "source_family_counts": stats.get("source_family_counts") or {},
        },
        "checks": checks,
        "recommendation": recommendation,
        "status_suggested": status_suggested,
        "notes": {
            "auto_promote_eligible": auto_promote_eligible,
            "medium_requires_additional_evidence": medium_requires_additional_evidence,
            "seeded_supersede_eligible": auto_supersede_eligible,
        },
    }


# -----------------------------
# Candidate fingerprint
# -----------------------------


def candidate_fingerprint(subject: str, predicate: str, scope: str, value: Any) -> str:
    value_c14n = json_c14n(value)
    data = f"{subject}\n{predicate}\n{scope}\n{value_c14n}"
    return sha256_hex(data)


# -----------------------------
# SQLite
# -----------------------------

_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "candidates.schema.sql"
SCHEMA_SQL = _SCHEMA_PATH.read_text(encoding="utf-8")


def connect(db_path: Path | str = DB_PATH_DEFAULT) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


@dataclass
class UpsertResult:
    candidate_id: str
    created: bool
    merged_evidence_added: int
    candidate_fingerprint: str
    status: str
    risk_level: str


def _coerce_json(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except Exception:
            return default
    return default


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _sanitize_decision_reason(decision_reason: str | None) -> str | None:
    """Normalize error decision tags so raw exception messages are never stored."""

    if decision_reason is None:
        return None

    reason = str(decision_reason).strip()
    if not reason:
        return None

    error_tag, sep, remainder = reason.partition(":")
    tag = error_tag.strip()
    if not sep or not tag.endswith("_error"):
        return reason

    exception_token, _, _ = remainder.partition(":")
    exception_type = exception_token.strip().split(".")[-1]
    normalized = exception_type.replace("_", "")
    if normalized and normalized.isalnum():
        return f"{tag}:{exception_type}"
    return f"{tag}:Error"


def refresh_candidate_policy_state(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    conflict_with_fact_id: str | None = None,
    status_override: str | None = None,
    decision_reason: str | None = None,
    preserve_terminal_status: bool = True,
) -> dict[str, Any]:
    """Recompute and persist policy trace/status for an existing candidate row."""

    row = conn.execute(
        "SELECT * FROM candidates WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Candidate not found: {candidate_id}")

    row_dict = dict(row)
    trace_prev = _coerce_json(row_dict.get("policy_trace_json"), {})
    if not isinstance(trace_prev, dict):
        trace_prev = {}

    evidence_refs = _coerce_json(row_dict.get("evidence_refs_json"), [])
    if not isinstance(evidence_refs, list):
        evidence_refs = []
    evidence_stats = _coerce_json(row_dict.get("evidence_stats_json"), {})
    if not isinstance(evidence_stats, dict) or not evidence_stats:
        evidence_stats = compute_evidence_stats(evidence_refs)

    value_obj: Any
    try:
        value_obj = json.loads(row_dict.get("value_json") or "null")
    except Exception:
        value_obj = row_dict.get("value_json")

    confidence_raw = row_dict.get("confidence")
    try:
        confidence = float(confidence_raw) if confidence_raw is not None else 0.0
    except (TypeError, ValueError):
        confidence = 0.0

    resolved_conflict_with_fact_id = conflict_with_fact_id
    if resolved_conflict_with_fact_id is None:
        resolved_conflict_with_fact_id = row_dict.get("conflict_with_fact_id")
    if resolved_conflict_with_fact_id == "":
        resolved_conflict_with_fact_id = None

    origin = str(trace_prev.get("origin") or trace_prev.get("reason") or "extracted")
    reason = str(trace_prev.get("reason") or origin)
    explicit_update = _coerce_bool(trace_prev.get("explicit_update"))
    content_cues = trace_prev.get("content_cues")
    content_cues = list(content_cues) if isinstance(content_cues, (list, tuple, set)) else None
    seeded_supersede_ok = bool(trace_prev.get("seeded_supersede_ok"))

    trace = compute_policy_trace(
        {
            "subject": row_dict.get("subject"),
            "predicate": row_dict.get("predicate"),
            "scope": row_dict.get("scope"),
            "assertion_type": row_dict.get("assertion_type"),
            "value": value_obj,
            "speaker_id": row_dict.get("speaker_id"),
            "confidence": confidence,
            "conflict_with_fact_id": resolved_conflict_with_fact_id,
            "origin": origin,
            "reason": reason,
            "explicit_update": explicit_update,
            "content_cues": content_cues,
            "seeded_supersede_ok": seeded_supersede_ok,
        },
        evidence_stats,
    )

    current_status = str(row_dict.get("status") or "pending")
    terminal_statuses = {"approved", "denied", "expired", "superseded"}
    if status_override is not None:
        next_status = status_override
    elif preserve_terminal_status and current_status in terminal_statuses:
        next_status = current_status
    else:
        next_status = str(trace.get("status_suggested") or "pending")

    safe_decision_reason = _sanitize_decision_reason(decision_reason)

    conn.execute(
        """
        UPDATE candidates
           SET risk_level = ?,
               status = ?,
               policy_version = ?,
               policy_trace_json = ?,
               evidence_stats_json = ?,
               conflict_with_fact_id = ?,
               decision_reason = COALESCE(?, decision_reason)
         WHERE candidate_id = ?
        """,
        (
            str(trace.get("risk_level") or row_dict.get("risk_level") or "medium"),
            next_status,
            active_policy_version(),
            json_c14n(trace),
            json_c14n(evidence_stats),
            resolved_conflict_with_fact_id,
            safe_decision_reason,
            candidate_id,
        ),
    )
    conn.commit()

    return {
        "candidate_id": candidate_id,
        "status": next_status,
        "risk_level": trace.get("risk_level"),
        "conflict_with_fact_id": resolved_conflict_with_fact_id,
        "policy_trace": trace,
        "evidence_stats": evidence_stats,
    }


def upsert_candidate(
    conn: sqlite3.Connection,
    *,
    subject: str,
    predicate: str,
    scope: str = "private",
    assertion_type: str,
    value: Any,
    evidence_refs: list[dict[str, Any]],
    evidence_quote: str | None = None,
    speaker_id: str | None = None,
    confidence: float | None = None,
    source_trust: str | None = None,
    conflict_with_fact_id: str | None = None,
    # policy inputs
    origin: str = "extracted",
    reason: str | None = None,
    explicit_update: bool = False,
    content_cues: list[str] | None = None,
    seeded_supersede_ok: bool = False,
    # preserve terminal states
    preserve_decision_status: bool = True,
    # optional ledger wiring
    ledger: Any | None = None,
) -> UpsertResult:
    """Insert or merge a candidate by fingerprint.

    Upsert semantics (CANDIDATES-APPROVALS-SPEC.md):
    - if fingerprint exists: merge evidence refs; update evidence stats; store latest policy trace
    - else insert new candidate row

    This function does not write to the Fact Ledger unless ledger kwarg is provided
    and the candidate is auto-promoted.

    Raises:
        IneligibleAssertionTypeError: if assertion_type is not in ELIGIBLE_ASSERTION_TYPES.
            The row is NOT inserted; callers should catch and handle gracefully.
    """

    # ── Merge gate: hard-block ineligible assertion types ──
    if assertion_type not in ELIGIBLE_ASSERTION_TYPES:
        raise IneligibleAssertionTypeError(assertion_type)

    # Ensure canonical-ish JSON value
    value_json = json.loads(json_c14n(value))
    fp = candidate_fingerprint(subject, predicate, scope, value_json)

    row = conn.execute(
        "SELECT * FROM candidates WHERE candidate_fingerprint = ?",
        (fp,),
    ).fetchone()

    incoming_refs = [normalize_evidence_ref(r) for r in (evidence_refs or [])]

    if row is None:
        candidate_id = new_ulid()
        created_at = _now_iso()

        merged_refs, added = merge_evidence_refs([], incoming_refs)
        stats = compute_evidence_stats(merged_refs)
        trace = compute_policy_trace(
            {
                "subject": subject,
                "predicate": predicate,
                "scope": scope,
                "assertion_type": assertion_type,
                "value": value_json,
                "speaker_id": speaker_id,
                "confidence": confidence,
                "conflict_with_fact_id": conflict_with_fact_id,
                "origin": origin,
                "reason": reason or origin,
                "explicit_update": explicit_update,
                "content_cues": content_cues,
                "seeded_supersede_ok": seeded_supersede_ok,
            },
            stats,
        )

        risk_level = trace["risk_level"]
        status_suggested = trace["status_suggested"]

        conn.execute(
            """
            INSERT INTO candidates (
              candidate_id, created_at, candidate_fingerprint,
              subject, predicate, scope, assertion_type, value_json,
              evidence_refs_json, evidence_quote,
              speaker_id, confidence, source_trust,
              risk_level, status, policy_version,
              policy_trace_json, evidence_stats_json,
              conflict_with_fact_id
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                candidate_id,
                created_at,
                fp,
                subject,
                predicate,
                scope,
                assertion_type,
                json_c14n(value_json),
                json_c14n(merged_refs),
                (evidence_quote[:200] if evidence_quote else None),
                speaker_id,
                float(confidence) if confidence is not None else None,
                source_trust,
                risk_level,
                status_suggested,
                active_policy_version(),
                json_c14n(trace),
                json_c14n(stats),
                conflict_with_fact_id,
            ),
        )
        conn.commit()

        promoted = False
        if status_suggested == "auto_promoted" and ledger is not None:
            promoted = auto_promote_if_eligible(conn, candidate_id, ledger=ledger)

        return UpsertResult(
            candidate_id=candidate_id,
            created=True,
            merged_evidence_added=added,
            candidate_fingerprint=fp,
            status=("approved" if promoted else status_suggested),
            risk_level=risk_level,
        )

    # merge existing
    candidate_id = row["candidate_id"]
    existing_refs = json.loads(row["evidence_refs_json"]) if row["evidence_refs_json"] else []

    merged_refs, added = merge_evidence_refs(existing_refs, incoming_refs)
    stats = compute_evidence_stats(merged_refs)

    current_status = row["status"]
    terminal = {"approved", "denied", "expired", "superseded"}

    trace = compute_policy_trace(
        {
            "subject": subject,
            "predicate": predicate,
            "scope": scope,
            "assertion_type": assertion_type,
            "value": value_json,
            "speaker_id": speaker_id or row["speaker_id"],
            "confidence": float(confidence) if confidence is not None else float(row["confidence"] or 0.0),
            "conflict_with_fact_id": conflict_with_fact_id or row["conflict_with_fact_id"],
            "origin": origin,
            "reason": reason or origin,
            "explicit_update": explicit_update,
            "content_cues": content_cues,
            "seeded_supersede_ok": seeded_supersede_ok,
        },
        stats,
    )

    risk_level = trace["risk_level"]
    status_suggested = trace["status_suggested"]

    new_status = current_status
    if not (preserve_decision_status and current_status in terminal):
        new_status = status_suggested

    conn.execute(
        """
        UPDATE candidates
           SET evidence_refs_json = ?,
               evidence_stats_json = ?,
               policy_trace_json = ?,
               risk_level = ?,
               status = ?,
               policy_version = ?,
               evidence_quote = COALESCE(?, evidence_quote),
               speaker_id = COALESCE(?, speaker_id),
               confidence = COALESCE(?, confidence),
               source_trust = COALESCE(?, source_trust),
               conflict_with_fact_id = COALESCE(?, conflict_with_fact_id)
         WHERE candidate_fingerprint = ?
        """,
        (
            json_c14n(merged_refs),
            json_c14n(stats),
            json_c14n(trace),
            risk_level,
            new_status,
            active_policy_version(),
            (evidence_quote[:200] if evidence_quote else None),
            speaker_id,
            float(confidence) if confidence is not None else None,
            source_trust,
            conflict_with_fact_id,
            fp,
        ),
    )
    conn.commit()

    promoted = False
    if status_suggested == "auto_promoted" and ledger is not None:
        promoted = auto_promote_if_eligible(conn, candidate_id, ledger=ledger)

    return UpsertResult(
        candidate_id=candidate_id,
        created=False,
        merged_evidence_added=added,
        candidate_fingerprint=fp,
        status=("approved" if promoted else new_status),
        risk_level=risk_level,
    )


def _candidate_fact_payload(row: sqlite3.Row) -> dict[str, Any]:
    value = json.loads(row["value_json"]) if row["value_json"] else None
    evidence_refs = json.loads(row["evidence_refs_json"]) if row["evidence_refs_json"] else []
    return {
        "subject": row["subject"],
        "predicate": row["predicate"],
        "scope": row["scope"] or "private",
        "assertion_type": row["assertion_type"],
        "speaker_id": row["speaker_id"],
        "confidence": float(row["confidence"]) if row["confidence"] is not None else None,
        "value": value,
        "evidence_refs": evidence_refs,
    }


def _decision_for_actor(actor_id: str) -> str:
    # Both v2 and v3 policy actors produce the same DB decision value.
    if actor_id in ("policy:v2", "policy:v3"):
        return "auto_promoted"
    return "approved"


def _normalize_reason(reason: str | None, *, fallback: str) -> str:
    safe = _sanitize_decision_reason(reason)
    return safe if safe else fallback


def promote_candidate(
    conn: sqlite3.Connection,
    candidate_id: str,
    actor_id: str,
    reason: str,
    ledger: Any | None = None,
) -> tuple[int, str | None]:
    """Promote a candidate to the personal fact ledger."""

    row = conn.execute(
        "SELECT * FROM candidates WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone()
    if row is None:
        return (0, None)

    # Guard: don't re-promote or write duplicate ledger events
    if row["status"] in ("approved", "auto_promoted") and row["ledger_event_id"]:
        return (0, row["ledger_event_id"])

    decision = _decision_for_actor(actor_id)
    decided_at = _now_iso()
    safe_reason = _normalize_reason(reason, fallback=decision)

    ledger_event_id = row["ledger_event_id"]
    if ledger is not None and not ledger_event_id:
        fact_payload = _candidate_fact_payload(row)
        ledger_row = ledger.append_event(
            "PROMOTE",
            actor_id=actor_id,
            reason=safe_reason,
            policy_version=row["policy_version"] or POLICY_VERSION_DEFAULT,
            candidate_id=candidate_id,
            fact=fact_payload,
            recorded_at=decided_at,
        )
        ledger_event_id = ledger_row.event_id

    res = conn.execute(
        """
        UPDATE candidates
           SET status = 'approved',
               decided_at = ?,
               actor_id = ?,
               decision = ?,
               decision_reason = ?,
               ledger_event_id = COALESCE(?, ledger_event_id)
         WHERE candidate_id = ?
        """,
        (
            decided_at,
            actor_id,
            decision,
            safe_reason,
            ledger_event_id,
            candidate_id,
        ),
    )
    conn.commit()
    return (res.rowcount, ledger_event_id)


def deny_candidate(
    conn: sqlite3.Connection,
    candidate_id: str,
    actor_id: str,
    reason: str,
    ledger: Any | None = None,
) -> tuple[int, str | None]:
    """Deny a candidate."""

    row = conn.execute(
        "SELECT * FROM candidates WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone()
    if row is None:
        return (0, None)

    # Guard: don't re-deny or write duplicate ledger events
    if row["status"] == "denied" and row["ledger_event_id"]:
        return (0, row["ledger_event_id"])

    decided_at = _now_iso()
    safe_reason = _normalize_reason(reason, fallback="denied")

    ledger_event_id = row["ledger_event_id"]
    if ledger is not None and not ledger_event_id:
        ledger_row = ledger.append_event(
            "DENY",
            actor_id=actor_id,
            reason=safe_reason,
            policy_version=row["policy_version"] or POLICY_VERSION_DEFAULT,
            candidate_id=candidate_id,
            fact=_candidate_fact_payload(row),
            recorded_at=decided_at,
        )
        ledger_event_id = ledger_row.event_id

    res = conn.execute(
        """
        UPDATE candidates
           SET status = 'denied',
               decided_at = ?,
               actor_id = ?,
               decision = 'denied',
               decision_reason = ?,
               ledger_event_id = COALESCE(?, ledger_event_id)
         WHERE candidate_id = ?
        """,
        (
            decided_at,
            actor_id,
            safe_reason,
            ledger_event_id,
            candidate_id,
        ),
    )
    conn.commit()
    return (res.rowcount, ledger_event_id)


def auto_promote_if_eligible(
    conn: sqlite3.Connection,
    candidate_id: str,
    ledger: Any | None = None,
) -> bool:
    """Promote candidates that have been marked auto_promoted by policy."""

    if ledger is None:
        return False

    row = conn.execute(
        "SELECT status FROM candidates WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone()
    if row is None:
        return False
    if row["status"] != "auto_promoted":
        return False

    actor_id = "policy:v3" if policy_v3_enabled() else "policy:v2"
    updated, _ = promote_candidate(
        conn,
        candidate_id,
        actor_id=actor_id,
        reason="auto_promote",
        ledger=ledger,
    )
    return updated > 0


def _normalize_verification_status(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized not in VERIFICATION_STATUSES:
        raise ValueError(
            f"Invalid verification_status {status!r}. "
            f"Allowed: {sorted(VERIFICATION_STATUSES)}"
        )
    return normalized


def upsert_candidate_verification(
    conn: sqlite3.Connection,
    *,
    candidate_id: str,
    verification_status: str,
    evidence_source_ids: list[str],
    verifier_version: str,
    verified_at: str | None = None,
) -> dict[str, Any]:
    """Insert a verification record for a candidate.

    Verification records are append-only by `(candidate_id, verifier_version, verified_at)`.
    """

    cid = str(candidate_id or "").strip()
    vv = str(verifier_version or "").strip()
    if not cid:
        raise ValueError("candidate_id is required")
    if not vv:
        raise ValueError("verifier_version is required")

    status = _normalize_verification_status(verification_status)
    ids = sorted({str(mid).strip() for mid in (evidence_source_ids or []) if str(mid).strip()})
    ts = str(verified_at or _now_iso()).strip()
    if not ts:
        ts = _now_iso()

    conn.execute(
        """
        INSERT OR REPLACE INTO candidate_verifications(
            candidate_id, verification_status, evidence_source_ids, verifier_version, verified_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (cid, status, json_c14n(ids), vv, ts),
    )
    conn.commit()
    return {
        "candidate_id": cid,
        "verification_status": status,
        "evidence_source_ids": ids,
        "verifier_version": vv,
        "verified_at": ts,
    }


def get_latest_candidate_verification(
    conn: sqlite3.Connection,
    candidate_id: str,
) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT candidate_id, verification_status, evidence_source_ids, verifier_version, verified_at
          FROM candidate_verifications
         WHERE candidate_id = ?
         ORDER BY verified_at DESC
         LIMIT 1
        """,
        (candidate_id,),
    ).fetchone()
    if row is None:
        return None
    ids = _coerce_json(row["evidence_source_ids"], [])
    if not isinstance(ids, list):
        ids = []
    return {
        "candidate_id": row["candidate_id"],
        "verification_status": row["verification_status"],
        "evidence_source_ids": [str(v) for v in ids if str(v).strip()],
        "verifier_version": row["verifier_version"],
        "verified_at": row["verified_at"],
    }


def upsert_om_dead_letter(
    conn: sqlite3.Connection,
    *,
    message_id: str,
    source_session_id: str,
    attempts: int,
    last_error: str,
    first_failed_at: str,
    last_failed_at: str,
    last_chunk_id: str | None,
) -> dict[str, Any]:
    """UPSERT one row in om_dead_letter_queue keyed by message_id."""

    message = str(message_id or "").strip()
    if not message:
        raise ValueError("message_id is required")

    source_session = str(source_session_id or "unknown").strip() or "unknown"
    err = str(last_error or "unknown_error").strip() or "unknown_error"
    first_failed = str(first_failed_at or _now_iso()).strip() or _now_iso()
    last_failed = str(last_failed_at or _now_iso()).strip() or _now_iso()
    chunk_id = str(last_chunk_id).strip() if last_chunk_id else None

    conn.execute(
        """
        INSERT INTO om_dead_letter_queue(
            message_id, source_session_id, attempts, last_error,
            first_failed_at, last_failed_at, last_chunk_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(message_id) DO UPDATE SET
            source_session_id = excluded.source_session_id,
            attempts = excluded.attempts,
            last_error = excluded.last_error,
            first_failed_at = excluded.first_failed_at,
            last_failed_at = excluded.last_failed_at,
            last_chunk_id = excluded.last_chunk_id
        """,
        (message, source_session, int(attempts), err, first_failed, last_failed, chunk_id),
    )
    conn.commit()
    return {
        "message_id": message,
        "source_session_id": source_session,
        "attempts": int(attempts),
        "last_error": err,
        "first_failed_at": first_failed,
        "last_failed_at": last_failed,
        "last_chunk_id": chunk_id,
    }


def remove_om_dead_letter(conn: sqlite3.Connection, message_id: str) -> int:
    res = conn.execute(
        "DELETE FROM om_dead_letter_queue WHERE message_id = ?",
        (message_id,),
    )
    conn.commit()
    return int(res.rowcount or 0)


def list_om_dead_letters(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT message_id, source_session_id, attempts, last_error,
               first_failed_at, last_failed_at, last_chunk_id
          FROM om_dead_letter_queue
         ORDER BY last_failed_at DESC, message_id ASC
        """
    ).fetchall()
    return [dict(r) for r in rows]
