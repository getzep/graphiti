#!/usr/bin/env python3
"""Promotion Policy v3 for OM-derived candidates.

Contracts implemented from EXEC-OBSERVATIONAL-MEMORY-SYNTHESIS PRD:
- promote only when verification_status == corroborated
- fail closed on hard-block interface unavailability/errors
- stable core_memory_id derivation: SHA256("core|{candidate_id}")
- idempotent MERGE writes for CoreMemory and SUPPORTS_CORE
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from truth import candidates as candidates_store


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _event(name: str, **payload: Any) -> dict[str, Any]:
    body = {"event": name, "timestamp": _now_iso(), **payload}
    print(json.dumps(body, ensure_ascii=True))
    return body


@dataclass(frozen=True)
class VerificationRecord:
    candidate_id: str
    verification_status: str
    evidence_source_ids: list[str]
    verifier_version: str
    verified_at: str


HardBlockCheck = Callable[[str], bool]
HardBlockLoader = Callable[[], HardBlockCheck]


def _load_policy_scanner_hard_block_check() -> HardBlockCheck:
    from truth import policy_scanner  # type: ignore

    fn = getattr(policy_scanner, "hard_block_check", None)
    if not callable(fn):
        raise RuntimeError("truth.policy_scanner.hard_block_check unavailable")
    return fn


def _load_security_policy_hard_block_check() -> HardBlockCheck:
    from truth import security_policy  # type: ignore

    fn = getattr(security_policy, "hard_block_check", None)
    if not callable(fn):
        raise RuntimeError("truth.security_policy.hard_block_check unavailable")
    return fn


def _load_promotion_policy_hard_block_check() -> HardBlockCheck:
    from truth import promotion_policy  # type: ignore

    fn = getattr(promotion_policy, "hard_block_check", None)
    if not callable(fn):
        raise RuntimeError("truth.promotion_policy.hard_block_check unavailable")
    return fn


ALLOWED_HARD_BLOCK_LOADERS: dict[str, HardBlockLoader] = {
    "policy_scanner": _load_policy_scanner_hard_block_check,
    "security_policy": _load_security_policy_hard_block_check,
    "promotion_policy": _load_promotion_policy_hard_block_check,
}
ALLOWED_HARD_BLOCK_IMPORT_ALIASES = {
    "truth.policy_scanner:hard_block_check": "policy_scanner",
    "truth.security_policy:hard_block_check": "security_policy",
    "truth.promotion_policy:hard_block_check": "promotion_policy",
}
DEFAULT_HARD_BLOCK_ORDER = ("policy_scanner", "security_policy", "promotion_policy")

_NEO4J_DRIVER_SINGLETON: Any | None = None


def _resolve_hard_block_check() -> HardBlockCheck:
    """Resolve hard_block_check(candidate_id) from approved policy module allowlist.

    Contract is fail-closed when the function is unavailable.
    """

    configured_raw = os.environ.get("OM_HARD_BLOCK_IMPORT", "").strip()
    configured_key = ALLOWED_HARD_BLOCK_IMPORT_ALIASES.get(configured_raw, configured_raw)

    candidate_keys: list[str] = []
    if configured_raw:
        if configured_key in ALLOWED_HARD_BLOCK_LOADERS:
            candidate_keys.append(configured_key)
        else:
            _event(
                "OM_PROMOTION_HARD_BLOCK_IMPORT_REJECTED",
                configured=configured_raw,
                reason="not_in_allowlist",
            )

    for key in DEFAULT_HARD_BLOCK_ORDER:
        if key not in candidate_keys:
            candidate_keys.append(key)

    for key in candidate_keys:
        loader = ALLOWED_HARD_BLOCK_LOADERS[key]
        try:
            return loader()
        except Exception:
            continue

    def _fail_closed(_: str) -> bool:
        raise RuntimeError("hard_block_check interface unavailable")

    return _fail_closed


def _read_verification_record(
    conn: Any,
    candidate_id: str,
) -> VerificationRecord | None:
    row = candidates_store.get_latest_candidate_verification(conn, candidate_id)
    if row is None:
        return None
    return VerificationRecord(
        candidate_id=row["candidate_id"],
        verification_status=row["verification_status"],
        evidence_source_ids=list(row.get("evidence_source_ids") or []),
        verifier_version=row["verifier_version"],
        verified_at=row["verified_at"],
    )


def _neo4j_driver_from_env() -> Any:
    try:
        from neo4j import GraphDatabase  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency/runtime specific
        raise RuntimeError("neo4j driver is required for promotion_policy_v3") from exc

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD is required")

    return GraphDatabase.driver(uri, auth=(user, password))


def _shared_neo4j_driver() -> Any:
    global _NEO4J_DRIVER_SINGLETON
    if _NEO4J_DRIVER_SINGLETON is None:
        _NEO4J_DRIVER_SINGLETON = _neo4j_driver_from_env()
    return _NEO4J_DRIVER_SINGLETON


def _close_shared_neo4j_driver() -> None:
    global _NEO4J_DRIVER_SINGLETON
    if _NEO4J_DRIVER_SINGLETON is None:
        return
    try:
        _NEO4J_DRIVER_SINGLETON.close()
    finally:
        _NEO4J_DRIVER_SINGLETON = None


atexit.register(_close_shared_neo4j_driver)


def promote_candidate(
    *,
    candidate_id: str,
    verification: VerificationRecord,
    hard_block_check: Callable[[str], bool],
    neo4j_driver: Any | None = None,
    candidates_conn: Any | None = None,
) -> dict[str, Any]:
    """Promote an OM candidate to a CoreMemory node in Neo4j.

    Gate order (all must pass):
    1. Unified v3 policy gate — when v3 is enabled, ``candidates_conn`` is
       required and ``candidates_store.policy_allows_promotion()`` must return
       (True, reason).  Missing ``candidates_conn`` fails closed.  This is the
       **one decision contract** shared with the graph-lane candidate path.
       Rollback: set GRAPHITI_POLICY_V3_ENABLED=0 and the gate is bypassed.
    2. OM verification gate — ``verification.verification_status`` must be
       "corroborated".  Always applied regardless of v3 flag.
    3. Fail-closed hard-block gate — ``hard_block_check(candidate_id)`` must
       return False.  If the interface is unavailable the gate fails closed
       (blocks promotion).  Always applied regardless of v3 flag.
    4. Neo4j MERGE write — idempotent CoreMemory + SUPPORTS_CORE upsert.
    """
    promoted_at = _now_iso()
    core_memory_id = sha256_hex(f"core|{candidate_id}")

    # ── Gate 1: Unified v3 policy decision contract ──────────────────────────
    # Consult the same decision engine used for lane candidates so OM and
    # graph-lane paths share one promotion contract.  Skip only when v3 is
    # disabled (rollback).  Under v3, missing candidates_conn is fail-closed.
    if candidates_store.policy_v3_enabled():
        if candidates_conn is None:
            gate_reason = "candidates_conn_missing"
            _event(
                "OM_PROMOTION_V3_POLICY_GATE_BLOCKED",
                candidate_id=candidate_id,
                gate_reason=gate_reason,
            )
            return {
                "candidate_id": candidate_id,
                "core_memory_id": core_memory_id,
                "promoted": False,
                "reason": f"v3_policy_gate:{gate_reason}",
            }

        allowed, gate_reason = candidates_store.policy_allows_promotion(
            candidate_id,
            conn=candidates_conn,
        )
        if not allowed:
            _event(
                "OM_PROMOTION_V3_POLICY_GATE_BLOCKED",
                candidate_id=candidate_id,
                gate_reason=gate_reason,
            )
            return {
                "candidate_id": candidate_id,
                "core_memory_id": core_memory_id,
                "promoted": False,
                "reason": f"v3_policy_gate:{gate_reason}",
            }

    # ── Gate 2: OM-specific verification gate ────────────────────────────────
    if verification.verification_status != "corroborated":
        return {
            "candidate_id": candidate_id,
            "core_memory_id": core_memory_id,
            "promoted": False,
            "reason": f"verification_status={verification.verification_status}",
        }

    # ── Gate 3: Fail-closed hard-block gate ──────────────────────────────────
    try:
        blocked = bool(hard_block_check(candidate_id))
    except Exception as exc:
        _event(
            "OM_PROMOTION_HARD_BLOCK_UNAVAILABLE",
            candidate_id=candidate_id,
            error=str(exc),
        )
        blocked = True

    if blocked:
        return {
            "candidate_id": candidate_id,
            "core_memory_id": core_memory_id,
            "promoted": False,
            "reason": "hard_blocked",
        }

    driver = neo4j_driver or _shared_neo4j_driver()
    created = False
    linked_edges = 0

    core_query = """
    MATCH (n:OMNode {node_id:$candidate_id})
    MERGE (c:CoreMemory {core_memory_id:$core_memory_id})
    ON CREATE SET
      c.candidate_id = $candidate_id,
      c.content = n.content,
      c.promoted_at = $promoted_at,
      c.retention_status = 'active',
      c.content_embedding = n.content_embedding,
      c.embedding_model = n.embedding_model,
      c.embedding_dim = n.embedding_dim
    RETURN c.core_memory_id AS core_memory_id,
           c.promoted_at AS promoted_at,
           c.candidate_id AS candidate_id
    """

    support_query = """
    MATCH (m:Message {message_id:$message_id})
    MATCH (c:CoreMemory {core_memory_id:$core_memory_id})
    MERGE (m)-[r:SUPPORTS_CORE {source_candidate_id:$candidate_id}]->(c)
    ON CREATE SET r.linked_at = $linked_at
    RETURN count(r) AS rel_count
    """

    with driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
        core_result = session.run(
            core_query,
            {
                "candidate_id": candidate_id,
                "core_memory_id": core_memory_id,
                "promoted_at": promoted_at,
            },
        )
        row = core_result.single()
        summary = core_result.consume()
        if row is None:
            # OMNode absent from Neo4j — emit event and skip gracefully.
            # Covers verification-only promotion paths (e.g. re-runs after
            # node eviction) where the verification record exists in
            # candidates.db but the OMNode was never written to Neo4j or
            # has since been removed.  Fail-closed: no CoreMemory is created.
            _event(
                "OM_PROMOTION_OMNODE_NOT_FOUND",
                candidate_id=candidate_id,
                core_memory_id=core_memory_id,
            )
            return {
                "candidate_id": candidate_id,
                "core_memory_id": core_memory_id,
                "promoted": False,
                "reason": "omnode_not_found",
            }
        created = bool(summary.counters.nodes_created > 0)

        for message_id in verification.evidence_source_ids:
            sid = str(message_id).strip()
            if not sid:
                continue
            session.run(
                support_query,
                {
                    "message_id": sid,
                    "core_memory_id": core_memory_id,
                    "candidate_id": candidate_id,
                    "linked_at": promoted_at,
                },
            ).consume()
            linked_edges += 1

    return {
        "candidate_id": candidate_id,
        "core_memory_id": core_memory_id,
        "promoted": True,
        "created": created,
        "supports_core_edges_attempted": linked_edges,
        "promoted_at": promoted_at,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Promotion policy v3")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--candidates-db",
        default=str(candidates_store.DB_PATH_DEFAULT),
        help="Path to candidates.db",
    )
    args = parser.parse_args()

    db_path = Path(args.candidates_db)
    conn = candidates_store.connect(db_path)
    try:
        verification = _read_verification_record(conn, args.candidate_id)
        if verification is None:
            _event(
                "OM_PROMOTION_SKIPPED_NO_VERIFICATION",
                candidate_id=args.candidate_id,
            )
            return 0

        hard_block_check = _resolve_hard_block_check()
        result = promote_candidate(
            candidate_id=args.candidate_id,
            verification=verification,
            hard_block_check=hard_block_check,
            candidates_conn=conn,  # wire unified v3 policy gate
        )
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0
    except Exception as exc:
        _event("OM_PROMOTION_FAILED", candidate_id=args.candidate_id, error=str(exc))
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
