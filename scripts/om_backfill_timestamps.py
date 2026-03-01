#!/usr/bin/env python3
"""One-time backfill: populate first_observed_at / last_observed_at on OMNodes.

Background
----------
Phase B (timeline semantics fix) wired ``first_observed_at`` and
``last_observed_at`` into the extraction path so new OMNodes carry accurate
event-time timestamps derived from their source messages.  Existing nodes
written before this fix have these fields set to NULL.

This script backfills those fields by querying each un-timestamped OMNode's
EVIDENCE_FOR-linked Messages, reading their ``created_at`` values, and
setting:

  first_observed_at = min(message.created_at)   over linked messages
  last_observed_at  = max(message.created_at)   over linked messages

Fallback (no linked messages): both fields are set to the node's own
``created_at`` (wall-clock OMNode creation time) with a warning event.
Nodes with no ``created_at`` at all are skipped with an error event.

Modes
-----
--dry-run  (DEFAULT)
    Report what would change without writing.
--apply
    Write the backfilled timestamps to Neo4j.

Idempotency
-----------
Only nodes where BOTH ``first_observed_at`` AND ``last_observed_at`` are NULL
are processed.  Re-running ``--apply`` on a fully-backfilled graph is a no-op.

Audit log
---------
All events written to stdout as JSONL.

    python3 scripts/om_backfill_timestamps.py --dry-run 2>&1 | tee backfill-dry.jsonl
    python3 scripts/om_backfill_timestamps.py --apply  2>&1 | tee backfill-apply.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NEO4J_ENV_FALLBACK_FILE = Path.home() / ".clawdbot" / "credentials" / "neo4j.env"
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
_NON_DEV_ENV_MARKERS = {"prod", "production", "staging", "stage", "preprod", "preview"}

BATCH_SIZE = 500  # nodes processed per query batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def emit(name: str, **payload: Any) -> None:
    event = {"event": name, "timestamp": now_iso(), **payload}
    print(json.dumps(event, ensure_ascii=True), flush=True)


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


def _fmt_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Neo4j helpers (mirrors om_compressor pattern)
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
# Backfill logic
# ---------------------------------------------------------------------------

def _count_unbackfilled(session: Any) -> int:
    row = session.run(
        """
        MATCH (n:OMNode)
        WHERE n.first_observed_at IS NULL
          AND n.last_observed_at IS NULL
        RETURN count(n) AS total
        """
    ).single()
    return int(row["total"] or 0) if row else 0


def _fetch_batch(session: Any, skip: int) -> list[dict[str, Any]]:
    """Fetch one batch of un-backfilled OMNodes with their linked message timestamps."""
    rows = session.run(
        """
        MATCH (n:OMNode)
        WHERE n.first_observed_at IS NULL
          AND n.last_observed_at IS NULL
        WITH n
        ORDER BY n.node_id ASC
        SKIP $skip
        LIMIT $limit
        OPTIONAL MATCH (m:Message)-[:EVIDENCE_FOR]->(n)
        RETURN n.node_id AS node_id,
               n.created_at AS node_created_at,
               collect(m.created_at) AS message_timestamps
        """,
        {"skip": skip, "limit": BATCH_SIZE},
    ).data()
    return rows


def _compute_timestamps(row: dict[str, Any]) -> tuple[str | None, str | None, str]:
    """Derive first/last_observed_at for one node row.

    Returns (first_observed_at, last_observed_at, source) where source is
    'messages' or 'fallback_created_at'.
    """
    msg_timestamps = [
        _parse_iso(str(ts))
        for ts in (row.get("message_timestamps") or [])
        if ts
    ]
    msg_timestamps = [dt for dt in msg_timestamps if dt is not None]

    if msg_timestamps:
        return _fmt_iso(min(msg_timestamps)), _fmt_iso(max(msg_timestamps)), "messages"

    # Fallback: use node's own created_at
    node_created_at = _parse_iso(str(row.get("node_created_at") or ""))
    if node_created_at is not None:
        ts = _fmt_iso(node_created_at)
        return ts, ts, "fallback_created_at"

    return None, None, "no_timestamp"


def _apply_batch(session: Any, updates: list[dict[str, Any]]) -> None:
    """Write timestamp updates for a batch of nodes in a single transaction."""
    session.run(
        """
        UNWIND $updates AS u
        MATCH (n:OMNode {node_id: u.node_id})
        SET n.first_observed_at = u.first_observed_at,
            n.last_observed_at  = u.last_observed_at,
            n.timestamps_backfilled_at = u.backfilled_at
        """,
        {
            "updates": [
                {
                    "node_id": u["node_id"],
                    "first_observed_at": u["first_observed_at"],
                    "last_observed_at": u["last_observed_at"],
                    "backfilled_at": now_iso(),
                }
                for u in updates
                if u.get("first_observed_at") is not None
            ]
        },
    ).consume()


def run(args: argparse.Namespace) -> int:
    dry_run: bool = not getattr(args, "apply", False)

    emit("OM_BACKFILL_TIMESTAMPS_START", dry_run=dry_run)

    driver = _neo4j_driver()
    with driver, driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
        total = _count_unbackfilled(session)
        emit("OM_BACKFILL_TIMESTAMPS_TOTAL", nodes_to_backfill=total, dry_run=dry_run)

        if total == 0:
            emit("OM_BACKFILL_TIMESTAMPS_DONE", processed=0, updated=0, skipped=0, dry_run=dry_run)
            return 0

        processed = 0
        updated = 0
        skipped = 0

        skip = 0
        while skip < total:
            rows = _fetch_batch(session, skip)
            if not rows:
                break

            batch_updates: list[dict[str, Any]] = []
            for row in rows:
                node_id = str(row.get("node_id") or "")
                first_ts, last_ts, source = _compute_timestamps(row)
                processed += 1

                if first_ts is None:
                    emit(
                        "OM_BACKFILL_TIMESTAMPS_SKIP",
                        node_id=node_id,
                        reason="no_timestamp_available",
                    )
                    skipped += 1
                    continue

                if source == "fallback_created_at":
                    emit(
                        "OM_BACKFILL_TIMESTAMPS_FALLBACK",
                        node_id=node_id,
                        source=source,
                        first_observed_at=first_ts,
                    )

                batch_updates.append({
                    "node_id": node_id,
                    "first_observed_at": first_ts,
                    "last_observed_at": last_ts,
                })
                updated += 1

            if not dry_run and batch_updates:
                _apply_batch(session, batch_updates)

            emit(
                "OM_BACKFILL_TIMESTAMPS_BATCH",
                batch_size=len(rows),
                updated_in_batch=len(batch_updates),
                dry_run=dry_run,
            )

            # Re-fetch from skip=0 on each non-dry-run iteration since nodes
            # no longer match the WHERE clause once updated.  For dry-run,
            # advance the window manually.
            if dry_run:
                skip += len(rows)
            else:
                # After applying, the updated nodes are gone from the query set.
                # Stay at skip=0 and read the next batch of still-unbackfilled nodes.
                pass

    emit(
        "OM_BACKFILL_TIMESTAMPS_DONE",
        processed=processed,
        updated=updated,
        skipped=skipped,
        dry_run=dry_run,
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill first_observed_at / last_observed_at on existing OMNodes. "
            "Default: --dry-run (safe, no writes)."
        )
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Report what would be backfilled without writing (default)",
    )
    mode_group.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Write backfilled timestamps to Neo4j",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        emit("OM_BACKFILL_TIMESTAMPS_ERROR", error=str(exc))
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
