#!/usr/bin/env python3
"""
Display ingest registry status.

Shows:
- Summary statistics (sources, chunks, exclusions)
- Per-source watermarks and chunk counts
- Constantine exclusion counts

Usage:
  cd tools/graphiti
  python scripts/registry_status.py
  python scripts/registry_status.py --verbose
  python scripts/registry_status.py --sources chatgpt
  python scripts/registry_status.py --exclusions
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add ingest module to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
from registry import (
    EXTRACTION_STATUS_FAILED,
    EXTRACTION_STATUS_QUEUED,
    get_registry,
)


QUEUE_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS ingest_jobs (
  job_id TEXT PRIMARY KEY,
  dedupe_key TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  source TEXT NOT NULL,
  job_type TEXT NOT NULL,
  group_id TEXT NOT NULL,
  lane TEXT NOT NULL,
  session_key TEXT,
  requested_ts TEXT,
  status TEXT NOT NULL,
  run_after TEXT NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 6,
  payload_json TEXT NOT NULL DEFAULT '{}',
  last_error TEXT,
  last_error_at TEXT,
  last_started_at TEXT,
  last_finished_at TEXT,
  last_exit_code INTEGER,
  last_duration_s REAL
);

CREATE INDEX IF NOT EXISTS idx_ingest_jobs_created_at
  ON ingest_jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_ingest_jobs_job_type
  ON ingest_jobs(job_type);
CREATE INDEX IF NOT EXISTS idx_ingest_jobs_status_run_after
  ON ingest_jobs(status, run_after);
CREATE INDEX IF NOT EXISTS idx_ingest_jobs_dedupe_key
  ON ingest_jobs(dedupe_key);

CREATE UNIQUE INDEX IF NOT EXISTS ux_ingest_jobs_active_dedupe
  ON ingest_jobs(dedupe_key)
  WHERE status IN ('queued', 'running');
"""


def format_timestamp(ts_str: str | None) -> str:
    """Format ISO timestamp for display."""
    if not ts_str:
        return "never"
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return ts_str


def get_queue_overview(db_path: Path) -> dict:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(QUEUE_SCHEMA_DDL)

        by_status = {
            row["status"]: int(row["cnt"] or 0)
            for row in conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM ingest_jobs GROUP BY status ORDER BY status"
            ).fetchall()
        }

        next_jobs = [
            dict(row)
            for row in conn.execute(
                """
                SELECT job_id, source, job_type, group_id, lane, status, run_after, attempts, max_attempts
                FROM ingest_jobs
                WHERE status IN ('queued', 'failed')
                ORDER BY run_after ASC, created_at ASC
                LIMIT 10
                """
            ).fetchall()
        ]

        running = [
            dict(row)
            for row in conn.execute(
                """
                SELECT job_id, source, job_type, group_id, lane, status, last_started_at, attempts, max_attempts
                FROM ingest_jobs
                WHERE status = 'running'
                ORDER BY last_started_at ASC
                LIMIT 10
                """
            ).fetchall()
        ]

        last_runs = [
            dict(row)
            for row in conn.execute(
                """
                SELECT
                  source,
                  job_type,
                  group_id,
                  lane,
                  MAX(CASE WHEN status = 'succeeded' THEN last_finished_at END) AS last_success_at,
                  MAX(CASE WHEN status IN ('failed', 'dead') THEN last_error_at END) AS last_error_at,
                  MAX(CASE WHEN status IN ('failed', 'dead') THEN last_error END) AS last_error
                FROM ingest_jobs
                GROUP BY source, job_type, group_id, lane
                ORDER BY job_type, source, group_id
                """
            ).fetchall()
        ]

        return {
            "by_status": by_status,
            "next_jobs": next_jobs,
            "running": running,
            "last_runs": last_runs,
        }
    finally:
        conn.close()


def _print_extraction_summary(extraction_stats: dict) -> None:
    total = int(extraction_stats.get("total") or 0)
    queued = int(extraction_stats.get("queued") or 0)
    failed = int(extraction_stats.get("failed") or 0)
    succeeded = int(extraction_stats.get("succeeded") or 0)
    unresolved = int(extraction_stats.get("unresolved") or 0)

    print("\n🧪 Extraction Lifecycle")
    print("-" * 60)
    print(f"Tracked episodes: {total}")
    print(f"Queued:           {queued}")
    print(f"Failed:           {failed}")
    print(f"Succeeded:        {succeeded}")
    print(f"Unresolved debt:  {unresolved}")

    oldest = extraction_stats.get("oldest_unresolved")
    newest = extraction_stats.get("newest_update")
    if oldest:
        print(f"Oldest unresolved: {format_timestamp(oldest)}")
    if newest:
        print(f"Latest update:    {format_timestamp(newest)}")


def _print_extraction_debt_rows(rows: list, limit: int) -> None:
    if not rows:
        print("No unresolved extraction debt 🎉")
        return

    print(f"Unresolved extraction rows: {len(rows)}")
    print("(status in queued/failed)")

    for row in rows[:limit]:
        reason = (row.last_failure_reason or "").strip()
        print(
            f"- [{row.status}] group={row.group_id} episode={row.episode_uuid} "
            f"chunk={row.chunk_key or '(missing)'} source={row.source_key or '(missing)'} "
            f"queued_at={format_timestamp(row.last_queued_at)}"
        )
        if reason:
            print(f"    reason: {reason}")

    if len(rows) > limit:
        print(f"... and {len(rows) - limit} more (use --limit to expand)")


def _print_replay_groups(groups: list[dict]) -> None:
    if not groups:
        print("No replay candidates found.")
        return

    print(f"Replay groups: {len(groups)}")
    for g in groups:
        print(f"\nGroup: {g['group_id']}")
        print(f"  Episodes:   {len(g.get('episodes') or [])}")
        print(f"  Source keys ({len(g.get('source_keys') or [])}):")
        for sk in g.get("source_keys") or []:
            print(f"    - {sk}")
        print(f"  Chunk keys ({len(g.get('chunk_keys') or [])}):")
        for ck in g.get("chunk_keys") or []:
            print(f"    - {ck}")


def main():
    ap = argparse.ArgumentParser(description="Display ingest registry status")
    ap.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Registry DB path override (default: state/ingest_registry.db)",
    )
    ap.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed per-source information",
    )
    ap.add_argument(
        "--sources",
        type=str,
        help="Filter to source type (chatgpt or session)",
    )
    ap.add_argument(
        "--exclusions",
        action="store_true",
        help="List all Constantine exclusions",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    ap.add_argument(
        "--extraction-debt",
        action="store_true",
        help="Show unresolved extraction debt (queued + failed)",
    )
    ap.add_argument(
        "--replay-list",
        action="store_true",
        help="Show grouped replay helper output for failed extractions",
    )
    ap.add_argument(
        "--group-id",
        type=str,
        help="Optional extraction group_id filter (for debt/replay views)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Limit rows in extraction debt/replay views (default: 200)",
    )
    args = ap.parse_args()

    registry = get_registry(db_path=args.db_path)
    stats = registry.get_stats()
    queue = get_queue_overview(registry.db_path)
    extraction_stats = registry.get_extraction_stats(group_id=args.group_id)

    if args.replay_list:
        replay_groups = registry.get_replay_groups(
            statuses=(EXTRACTION_STATUS_FAILED,),
            group_id=args.group_id,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps({"replay_groups": replay_groups}, indent=2))
        else:
            print("\n🎯 Targeted Replay Helper")
            print("=" * 60)
            if args.group_id:
                print(f"group_id filter: {args.group_id}")
            _print_replay_groups(replay_groups)
        return

    if args.extraction_debt:
        debt_rows = registry.list_extractions(
            statuses=(EXTRACTION_STATUS_QUEUED, EXTRACTION_STATUS_FAILED),
            group_id=args.group_id,
            limit=args.limit,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "extraction_stats": extraction_stats,
                        "rows": [row.__dict__ for row in debt_rows],
                    },
                    indent=2,
                )
            )
        else:
            print("\n🚨 Extraction Debt")
            print("=" * 60)
            if args.group_id:
                print(f"group_id filter: {args.group_id}")
            _print_extraction_summary(extraction_stats)
            print()
            _print_extraction_debt_rows(debt_rows, limit=args.limit)
        return

    if args.json:
        if args.exclusions:
            exclusions = registry.get_exclusions()
            print(json.dumps(exclusions, indent=2))
        else:
            output = {
                "stats": stats,
                "queue": queue,
                "extraction_stats": extraction_stats,
                "sources": [
                    {
                        "source_key": s.source_key,
                        "source_type": s.source_type,
                        "watermark": s.watermark,
                        "watermark_str": s.watermark_str,
                        "overlap_window": s.overlap_window,
                        "last_ingested_at": s.last_ingested_at,
                        "chunk_count": s.chunk_count,
                    }
                    for s in registry.list_sources(args.sources)
                ],
            }
            print(json.dumps(output, indent=2))
        return

    if args.exclusions:
        exclusions = registry.get_exclusions()
        print(f"\n🚫 Constantine Exclusions ({len(exclusions)} total)")
        print("=" * 60)
        
        if not exclusions:
            print("No exclusions recorded.")
            return
        
        by_reason: dict[str, list] = {}
        for e in exclusions:
            reason = e.get("reason", "unknown")
            by_reason.setdefault(reason, []).append(e)
        
        for reason, items in sorted(by_reason.items()):
            print(f"\n{reason} ({len(items)} conversations):")
            if args.verbose:
                for e in items[:20]:  # Limit to 20 per category
                    title = e.get("title", "")[:40]
                    print(f"  - {e['conversation_id'][:12]}... {title}")
                if len(items) > 20:
                    print(f"  ... and {len(items) - 20} more")
        return

    # Summary
    print("\n📊 Ingest Registry Status")
    print("=" * 60)
    print(f"Database: {registry.db_path}")
    print()
    print(f"Sources tracked:       {stats['source_count']}")
    print(f"Chunks ingested:       {stats['chunk_count']}")
    print(f"Exclusions (Constantine): {stats['exclusion_count']}")
    print(f"Latest ingest:         {format_timestamp(stats['latest_ingest'])}")

    if stats['sources_by_type']:
        print("\nSources by type:")
        for stype, count in sorted(stats['sources_by_type'].items()):
            chunks = stats['chunks_by_type'].get(stype, 0)
            print(f"  {stype}: {count} sources, {chunks} chunks")

    _print_extraction_summary(extraction_stats)

    # Queue overview
    by_status = queue.get("by_status") or {}
    queued = int(by_status.get("queued") or 0)
    failed = int(by_status.get("failed") or 0)
    running = int(by_status.get("running") or 0)
    succeeded = int(by_status.get("succeeded") or 0)
    dead = int(by_status.get("dead") or 0)

    print("\n⏳ Ingest Queue")
    print("-" * 60)
    print(f"Queued:    {queued}")
    print(f"Failed:    {failed}")
    print(f"Running:   {running}")
    print(f"Succeeded: {succeeded}")
    print(f"Dead:      {dead}")

    next_jobs = queue.get("next_jobs") or []
    if next_jobs:
        print("\nNext runnable:")
        for j in next_jobs[:5]:
            print(
                f"  - {j['status']} {j['job_type']} ({j['source']}) "
                f"group_id={j['group_id']} lane={j['lane']} run_after={format_timestamp(j.get('run_after'))} "
                f"attempts={j.get('attempts')}/{j.get('max_attempts')}"
            )
    else:
        print("\nNext runnable: none")

    running_jobs = queue.get("running") or []
    if running_jobs:
        print("\nRunning now:")
        for j in running_jobs[:5]:
            print(
                f"  - running {j['job_type']} ({j['source']}) "
                f"group_id={j['group_id']} lane={j['lane']} started={format_timestamp(j.get('last_started_at'))} "
                f"attempts={j.get('attempts')}/{j.get('max_attempts')}"
            )

    last_runs = queue.get("last_runs") or []
    last_runs = [r for r in last_runs if r.get("last_success_at") or r.get("last_error_at")]
    if last_runs:
        print("\nLast run (by job):")
        for r in last_runs[:5]:
            last_success = format_timestamp(r.get("last_success_at"))
            last_error_at = format_timestamp(r.get("last_error_at"))
            last_error = r.get("last_error") or ""
            print(
                f"  - {r['job_type']} ({r['source']}) group_id={r['group_id']} lane={r['lane']} "
                f"last_success={last_success} last_error_at={last_error_at} last_error={last_error}"
            )

    if args.verbose:
        if queue.get("last_runs"):
            print("\nLast run (full):")
            for r in queue.get("last_runs") or []:
                last_success = format_timestamp(r.get("last_success_at"))
                last_error_at = format_timestamp(r.get("last_error_at"))
                last_error = r.get("last_error") or ""
                print(
                    f"  - {r['job_type']} ({r['source']}) group_id={r['group_id']} lane={r['lane']} "
                    f"last_success={last_success} last_error_at={last_error_at} last_error={last_error}"
                )

    # Per-source details
    sources = registry.list_sources(args.sources)
    
    if sources:
        print(f"\n📁 Sources ({len(sources)} total)")
        print("-" * 60)
        
        for s in sources:
            print(f"\n{s.source_key}")
            print(f"  Type:         {s.source_type}")
            print(f"  Chunks:       {s.chunk_count}")
            print(f"  Watermark:    {s.watermark_str or 'not set'}")
            print(f"  Overlap:      {s.overlap_window}")
            print(f"  Last ingest:  {format_timestamp(s.last_ingested_at)}")
    else:
        print("\nNo sources tracked yet.")
        print("Run one of the ingest scripts to populate the registry:")
        print("  python scripts/mcp_ingest_chatgpt.py --group-id <id>")
        print("  python scripts/mcp_ingest_sessions.py --group-id <id>")


if __name__ == "__main__":
    main()
