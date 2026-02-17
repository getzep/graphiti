#!/usr/bin/env python3
"""Enqueue an incremental ingest job when `/done` is used.

This script is intended to be invoked by an assistant-side `/done` hook. It writes a durable
queue entry into the ingest registry DB (`state/ingest_registry.db`). A separate worker
(`scripts/run_incremental_ingest.py`) is responsible for executing queued jobs.

Dry-run behavior:
- We still write a *deduped* queue entry tagged `dry_run=true` in the payload. This lets the
  validation script exercise the queue lifecycle without sending anything to Graphiti.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Shared queue schema + helpers (single source of truth).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.queue import connect, ensure_schema, validate_identifier  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "state" / "ingest_registry.db"

DEFAULT_JOB_TYPE = "sessions_incremental"
DEFAULT_LANE = "primary"
DEFAULT_OVERLAP_CHUNKS = 10
DEFAULT_MAX_ATTEMPTS = 6


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso_ts(ts: str) -> Optional[float]:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Enqueue incremental ingest job on /done")
    ap.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--source", required=True, help="Trigger source (e.g. done)")
    ap.add_argument("--session-key", required=True, help="Opaque session key (used as group_id by default)")
    ap.add_argument("--ts", required=True, help="Event timestamp (ISO 8601, e.g. 2026-02-07T00:00:00Z)")
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP_CHUNKS, help="Overlap chunks for incremental ingest")
    ap.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    ap.add_argument("--dry-run", action="store_true", help="Queue a dry-run job (safe for validation)")
    args = ap.parse_args()

    db_path = Path(args.db_path)
    now_iso = _utc_now_iso()
    now_ts = time.time()
    requested_ts_epoch = _parse_iso_ts(args.ts)

    # /done jobs are sessions incremental ingest by default.
    job_type = DEFAULT_JOB_TYPE
    lane = DEFAULT_LANE

    try:
        group_id = validate_identifier(args.session_key, "session_key")
        validate_identifier(args.source, "source")
    except ValueError as exc:
        print(f"validation_error: {exc}", file=sys.stderr)
        return 1

    payload: dict[str, Any] = {
        "job_type": job_type,
        "group_id": group_id,
        "lane": lane,
        "session_key": args.session_key,
        "requested_ts": args.ts,
        "requested_ts_epoch": requested_ts_epoch,
        "incremental": True,
        "overlap": int(args.overlap),
        "dry_run": bool(args.dry_run),
    }

    # Dedupe at the event level.
    dedupe_key = f"{args.source}:{job_type}:{group_id}:{args.ts}"
    job_id = uuid.uuid4().hex

    try:
        conn = connect(db_path)
        try:
            ensure_schema(conn)

            cur = conn.execute(
                """
                INSERT OR IGNORE INTO ingest_jobs (
                  job_id, dedupe_key, created_at, updated_at,
                  source, job_type, group_id, lane,
                  session_key, requested_ts,
                  status, run_after,
                  attempts, max_attempts,
                  payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    dedupe_key,
                    now_iso,
                    now_iso,
                    str(args.source),
                    job_type,
                    group_id,
                    lane,
                    str(args.session_key),
                    str(args.ts),
                    "queued",
                    now_iso,  # runnable immediately
                    0,
                    int(args.max_attempts),
                    json.dumps(payload, sort_keys=True),
                ),
            )
            conn.commit()

            if cur.rowcount == 1:
                row = conn.execute(
                    "SELECT job_id, status, job_type, group_id, lane, attempts, max_attempts, run_after "
                    "FROM ingest_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                print(
                    "ENQUEUED "
                    f"job_id={row['job_id']} status={row['status']} "
                    f"job_type={row['job_type']} group_id={row['group_id']} lane={row['lane']} "
                    f"run_after={row['run_after']} dry_run={bool(args.dry_run)}"
                )
            else:
                # Insert was ignored due to active dedupe (queued/running).
                row = conn.execute(
                    "SELECT job_id, status, job_type, group_id, lane, attempts, max_attempts, run_after "
                    "FROM ingest_jobs WHERE dedupe_key = ? AND status IN ('queued', 'running')",
                    (dedupe_key,),
                ).fetchone()
                print(
                    "ALREADY_ENQUEUED "
                    f"job_id={row['job_id']} status={row['status']} "
                    f"job_type={row['job_type']} group_id={row['group_id']} lane={row['lane']} "
                    f"run_after={row['run_after']} dry_run={bool(args.dry_run)}"
                )

            return 0
        finally:
            conn.close()
    except sqlite3.Error as e:
        # Avoid leaking raw exception messages into logs/state.
        print(f"error_type:{type(e).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
