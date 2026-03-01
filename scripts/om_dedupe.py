#!/usr/bin/env python3
"""OM node exact-duplicate detector and merger.

Exact duplicates are OMNodes that share the same stable content key:

    node_type + normalize_text(content).lower()

This matches the hash used in the compressor's node-ID derivation (minus
``semantic_domain``), so nodes that ended up with different ``node_id`` values
due to semantic_domain drift, extractor-version changes, or manual writes are
caught here.

Modes
-----
--dry-run  (DEFAULT)
    Report duplicate groups to stdout as JSONL events. No writes.
--apply
    Merge each duplicate group into a single canonical node, redirect edges,
    and delete the non-canonical nodes. All changes are emitted as JSONL
    audit events.

Merge semantics (safe, conservative)
-------------------------------------
- Canonical node: the node with the earliest ``created_at`` (deterministic
  tie-break: smallest ``node_id`` lexicographically).
- ``source_message_ids``: union of all duplicate node source lists, deduped.
- ``first_observed_at``: min of all non-null values; fallback to canonical's
  ``created_at``.
- ``last_observed_at``: max of all non-null values; may remain NULL if none set.
- ``urgency_score``: max across duplicates (retain highest urgency).
- ``status``: most-active wins (open > reopened > monitoring > closed >
  abandoned).  Never demote a node to a less-active status.

Idempotency
-----------
Running ``--apply`` multiple times is safe.  The dedupe key scan only
returns nodes that are CURRENTLY duplicates.  Already-merged nodes produce
no events on subsequent runs.

Audit log
---------
All events are written to stdout as JSONL (one JSON object per line).
Pipe to a file for a persistent audit trail:

    python3 scripts/om_dedupe.py --dry-run 2>&1 | tee om-dedupe-dry.jsonl
    python3 scripts/om_dedupe.py --apply 2>&1 | tee om-dedupe-apply.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Status ordering (higher = more active)
# ---------------------------------------------------------------------------
_STATUS_RANK: dict[str, int] = {
    "abandoned": 0,
    "closed": 1,
    "monitoring": 2,
    "reopened": 3,
    "open": 4,
}
_STATUS_RANK_DEFAULT = 2  # unknown strings treated as monitoring-level

# Relation types allowed on OMNodes (mirrors om_compressor.RELATION_TYPES)
RELATION_TYPES = frozenset({"MOTIVATES", "GENERATES", "SUPERSEDES", "ADDRESSES", "RESOLVES"})

NEO4J_ENV_FALLBACK_FILE = Path.home() / ".clawdbot" / "credentials" / "neo4j.env"
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
_NON_DEV_ENV_MARKERS = {"prod", "production", "staging", "stage", "preprod", "preview"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    text = text.strip()
    return re.sub(r"\s+", " ", text)


def emit(name: str, **payload: Any) -> None:
    event = {"event": name, "timestamp": now_iso(), **payload}
    print(json.dumps(event, ensure_ascii=True), flush=True)


def dedupe_key(node_type: str, content: str) -> str:
    """Stable dedupe key for OM nodes.

    Mirrors the node-ID scheme in om_compressor but excludes ``semantic_domain``
    so nodes with drift in that field are still detected as duplicates.
    """
    return sha256_hex(f"dedupekey|{node_type}|{normalize_text(content).lower()}")


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    txt = str(value).strip()
    if not txt:
        return None
    try:
        if txt.endswith("Z"):
            return datetime.fromisoformat(txt[:-1]).replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(txt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _status_rank(status: str | None) -> int:
    return _STATUS_RANK.get((status or "").lower(), _STATUS_RANK_DEFAULT)


def _pick_canonical(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the canonical node from a duplicate group.

    Prefer the node with the earliest ``created_at``.
    Tie-break: smallest ``node_id`` lexicographically (deterministic).
    """
    def sort_key(n: dict[str, Any]) -> tuple[str, str]:
        ts = n.get("created_at") or "9999"
        return (str(ts), str(n.get("node_id") or ""))

    return sorted(nodes, key=sort_key)[0]


def _merge_metadata(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute merged metadata for a canonical node from all duplicates."""
    # union source_message_ids
    all_source_ids: list[str] = []
    seen_ids: set[str] = set()
    for n in nodes:
        for mid in (n.get("source_message_ids") or []):
            if mid and mid not in seen_ids:
                all_source_ids.append(str(mid))
                seen_ids.add(str(mid))

    # max urgency (most conservative = retain highest urgency)
    urgency = max(
        max(1, min(5, int(n.get("urgency_score") or 3)))
        for n in nodes
    )

    # most-active status
    best_status = max(
        ((n.get("status") or "open") for n in nodes),
        key=_status_rank,
    )

    # first_observed_at: min of all non-null values
    foa_candidates = [
        _parse_iso(str(n.get("first_observed_at") or ""))
        for n in nodes
        if n.get("first_observed_at")
    ]
    # fallback to created_at if no first_observed_at is set
    if not foa_candidates:
        foa_candidates = [
            _parse_iso(str(n.get("created_at") or ""))
            for n in nodes
            if n.get("created_at")
        ]
    first_observed_at = (
        min(foa_candidates).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        if foa_candidates
        else None
    )

    # last_observed_at: max of all non-null values
    loa_candidates = [
        _parse_iso(str(n.get("last_observed_at") or ""))
        for n in nodes
        if n.get("last_observed_at")
    ]
    last_observed_at = (
        max(loa_candidates).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        if loa_candidates
        else None
    )

    return {
        "source_message_ids": all_source_ids,
        "urgency_score": urgency,
        "status": best_status,
        "first_observed_at": first_observed_at,
        "last_observed_at": last_observed_at,
    }


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

def _is_non_dev_mode() -> bool:
    for key in ("OM_ENV", "APP_ENV", "ENVIRONMENT", "NODE_ENV"):
        value = os.environ.get(key)
        if value and value.strip().lower() in _NON_DEV_ENV_MARKERS:
            return True
    return os.environ.get("CI", "").strip().lower() in _TRUTHY_ENV_VALUES


def _load_neo4j_env_fallback() -> None:
    if os.environ.get("NEO4J_PASSWORD"):
        return
    if _is_non_dev_mode():
        opt_in = os.environ.get("OM_NEO4J_ENV_FALLBACK_NON_DEV", "").strip().lower()
        if opt_in not in _TRUTHY_ENV_VALUES:
            return
    if not NEO4J_ENV_FALLBACK_FILE.exists():
        return
    for raw_line in NEO4J_ENV_FALLBACK_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in {"NEO4J_PASSWORD", "NEO4J_USER", "NEO4J_URI"} and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _neo4j_driver() -> Any:
    try:
        from neo4j import GraphDatabase  # type: ignore
    except Exception as exc:
        raise RuntimeError("neo4j driver is required") from exc

    _load_neo4j_env_fallback()

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD is required")

    return GraphDatabase.driver(uri, auth=(user, password))


# ---------------------------------------------------------------------------
# Core scan logic
# ---------------------------------------------------------------------------

def _fetch_all_om_nodes(session: Any) -> list[dict[str, Any]]:
    """Fetch all OMNode records needed for dedupe analysis."""
    rows = session.run(
        """
        MATCH (n:OMNode)
        RETURN n.node_id            AS node_id,
               n.node_type          AS node_type,
               n.semantic_domain    AS semantic_domain,
               n.content            AS content,
               n.urgency_score      AS urgency_score,
               n.status             AS status,
               coalesce(n.source_message_ids, []) AS source_message_ids,
               n.created_at         AS created_at,
               n.first_observed_at  AS first_observed_at,
               n.last_observed_at   AS last_observed_at
        ORDER BY n.created_at ASC, n.node_id ASC
        """
    ).data()

    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({
            "node_id": str(r.get("node_id") or ""),
            "node_type": str(r.get("node_type") or ""),
            "semantic_domain": str(r.get("semantic_domain") or ""),
            "content": str(r.get("content") or ""),
            "urgency_score": int(r.get("urgency_score") or 3),
            "status": str(r.get("status") or "open"),
            "source_message_ids": [str(mid) for mid in (r.get("source_message_ids") or [])],
            "created_at": str(r.get("created_at") or ""),
            "first_observed_at": str(r.get("first_observed_at") or "") or None,
            "last_observed_at": str(r.get("last_observed_at") or "") or None,
        })
    return out


def _find_duplicate_groups(nodes: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group OMNodes by their dedupe key; return only groups with >1 node."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        key = dedupe_key(node["node_type"], node["content"])
        groups.setdefault(key, []).append(node)

    return [group for group in groups.values() if len(group) > 1]


# ---------------------------------------------------------------------------
# Apply: merge duplicate group into canonical
# ---------------------------------------------------------------------------

def _apply_merge_group(
    session: Any,
    group: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> None:
    canonical = _pick_canonical(group)
    merged = _merge_metadata(group)
    duplicates = [n for n in group if n["node_id"] != canonical["node_id"]]
    dup_ids = [n["node_id"] for n in duplicates]

    emit(
        "OM_DEDUPE_GROUP_FOUND",
        canonical_node_id=canonical["node_id"],
        duplicate_node_ids=dup_ids,
        node_type=canonical["node_type"],
        merged_source_message_count=len(merged["source_message_ids"]),
        merged_urgency_score=merged["urgency_score"],
        merged_status=merged["status"],
        dry_run=dry_run,
    )

    if dry_run:
        return

    now = now_iso()

    # 1. Update canonical node with merged metadata
    session.run(
        """
        MATCH (n:OMNode {node_id: $node_id})
        SET n.source_message_ids = $source_message_ids,
            n.urgency_score      = $urgency_score,
            n.status             = $status,
            n.first_observed_at  = $first_observed_at,
            n.last_observed_at   = $last_observed_at,
            n.dedupe_merged_at   = $now
        """,
        {
            "node_id": canonical["node_id"],
            "source_message_ids": merged["source_message_ids"],
            "urgency_score": merged["urgency_score"],
            "status": merged["status"],
            "first_observed_at": merged["first_observed_at"],
            "last_observed_at": merged["last_observed_at"],
            "now": now,
        },
    ).consume()

    # 2. Redirect edges from duplicates to canonical, then delete duplicates
    for dup in duplicates:
        dup_id = dup["node_id"]

        # Redirect EVIDENCE_FOR edges (Message → OMNode)
        session.run(
            """
            MATCH (m:Message)-[r:EVIDENCE_FOR]->(dup:OMNode {node_id: $dup_id})
            MATCH (canonical:OMNode {node_id: $canonical_id})
            MERGE (m)-[:EVIDENCE_FOR {
                chunk_id: coalesce(r.chunk_id, 'dedupe_merge'),
                extractor_version: coalesce(r.extractor_version, 'dedupe_merge'),
                linked_at: coalesce(r.linked_at, $now)
            }]->(canonical)
            """,
            {"dup_id": dup_id, "canonical_id": canonical["node_id"], "now": now},
        ).consume()

        # Redirect SUPPORTS_CORE edges (OMNode → CoreMemory) — copy properties
        session.run(
            """
            MATCH (dup:OMNode {node_id: $dup_id})-[r:SUPPORTS_CORE]->(c:CoreMemory)
            MATCH (canonical:OMNode {node_id: $canonical_id})
            MERGE (canonical)-[nr:SUPPORTS_CORE]->(c)
            ON CREATE SET nr = properties(r)
            DELETE r
            """,
            {"dup_id": dup_id, "canonical_id": canonical["node_id"]},
        ).consume()

        # Redirect OM relation-type edges (outgoing from dup) — copy properties
        for rel in RELATION_TYPES:
            session.run(
                f"""
                MATCH (dup:OMNode {{node_id: $dup_id}})-[r:{rel}]->(target:OMNode)
                WHERE target.node_id <> $canonical_id
                MATCH (canonical:OMNode {{node_id: $canonical_id}})
                MERGE (canonical)-[nr:{rel}]->(target)
                ON CREATE SET nr = properties(r)
                DELETE r
                """,
                {"dup_id": dup_id, "canonical_id": canonical["node_id"]},
            ).consume()

        # Redirect OM relation-type edges (incoming to dup) — copy properties
        for rel in RELATION_TYPES:
            session.run(
                f"""
                MATCH (source:OMNode)-[r:{rel}]->(dup:OMNode {{node_id: $dup_id}})
                WHERE source.node_id <> $canonical_id
                MATCH (canonical:OMNode {{node_id: $canonical_id}})
                MERGE (source)-[nr:{rel}]->(canonical)
                ON CREATE SET nr = properties(r)
                DELETE r
                """,
                {"dup_id": dup_id, "canonical_id": canonical["node_id"]},
            ).consume()

        # Redirect OMExtractionEvent → OMNode pointer
        session.run(
            """
            MATCH (e:OMExtractionEvent)-[r:EMITTED]->(dup:OMNode {node_id: $dup_id})
            MATCH (canonical:OMNode {node_id: $canonical_id})
            MERGE (e)-[:EMITTED]->(canonical)
            DELETE r
            """,
            {"dup_id": dup_id, "canonical_id": canonical["node_id"]},
        ).consume()

        # Delete the duplicate node (detach removes remaining edges)
        session.run(
            """
            MATCH (dup:OMNode {node_id: $dup_id})
            DETACH DELETE dup
            """,
            {"dup_id": dup_id},
        ).consume()

        emit(
            "OM_DEDUPE_MERGED",
            canonical_node_id=canonical["node_id"],
            deleted_node_id=dup_id,
            merged_at=now,
        )


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    dry_run: bool = not getattr(args, "apply", False)

    emit("OM_DEDUPE_START", dry_run=dry_run, mode="dry-run" if dry_run else "apply")

    driver = _neo4j_driver()
    with driver, driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
        nodes = _fetch_all_om_nodes(session)
        emit("OM_DEDUPE_SCANNED", total_nodes=len(nodes))

        groups = _find_duplicate_groups(nodes)
        emit("OM_DEDUPE_GROUPS_FOUND", duplicate_groups=len(groups))

        if not groups:
            emit("OM_DEDUPE_DONE", duplicate_groups=0, nodes_merged=0, dry_run=dry_run)
            return 0

        merged = 0
        for group in groups:
            _apply_merge_group(session, group, dry_run=dry_run)
            if not dry_run:
                merged += len(group) - 1  # canonical kept, rest merged/deleted

    emit(
        "OM_DEDUPE_DONE",
        duplicate_groups=len(groups),
        nodes_merged=merged,
        dry_run=dry_run,
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect and optionally merge exact-duplicate OMNodes. "
            "Default: --dry-run (safe, no writes)."
        )
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Report duplicates without making changes (default)",
    )
    mode_group.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Merge and delete duplicate nodes (writes to Neo4j)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        emit("OM_DEDUPE_ERROR", error=str(exc))
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
