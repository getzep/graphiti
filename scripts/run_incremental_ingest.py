#!/usr/bin/env python3
"""Background worker for the durable ingest queue (registry DB).

This worker:
- Enqueues the scheduled 30-minute sessions incremental ingest (main group) when due.
- Claims the next runnable queued job atomically (safe under concurrent cron invocations).
- Executes the corresponding incremental ingest command.
- Retries failures with bounded exponential backoff + jitter.
- Re-queues stale "running" jobs (best-effort recovery).

The queue is stored in: state/ingest_registry.db (table: ingest_jobs)
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

# Shared queue schema + helpers (single source of truth).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.queue import connect as _connect  # noqa: E402
from ingest.queue import ensure_schema as _ensure_queue_schema  # noqa: E402
from ingest.queue import validate_identifier  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "state" / "ingest_registry.db"

SESSIONS_MAIN_GROUP_ID = "s1_sessions_main"
DEFAULT_SESSIONS_OVERLAP_CHUNKS = 10
DEFAULT_SESSIONS_DIR = Path.home() / ".clawdbot"
DEFAULT_EVIDENCE_OUT = REPO_ROOT / "evidence"
DEFAULT_SESSIONS_AGENT_ID = "main"

SCHEDULE_PERIOD_S = 30 * 60
STALE_RUNNING_REQUEUE_S = 2 * 60 * 60

BACKOFF_BASE_S = 60
BACKOFF_CAP_S = 30 * 60
BACKOFF_JITTER_FRAC = 0.20


def _utc_now_dt() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _utc_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso(s: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).replace(microsecond=0)
    except Exception:
        return None


def _sessions_evidence_path(evidence_out: Path, *, agent_id: str = DEFAULT_SESSIONS_AGENT_ID) -> Path:
    return evidence_out / "sessions_v1" / agent_id / "all_evidence.json"


def _sessions_transcripts_latest_mtime_ns(sessions_dir: Path, *, agent_id: str = DEFAULT_SESSIONS_AGENT_ID) -> int:
    """Best-effort latest transcript mtime in ns under <sessions_dir>/agents/<agent_id>/sessions/*.jsonl.

    Returns 0 if no transcript files are present.
    """

    transcripts_dir = sessions_dir / "agents" / agent_id / "sessions"
    latest_ns = 0
    for p in transcripts_dir.glob("*.jsonl"):
        try:
            latest_ns = max(latest_ns, int(p.stat().st_mtime_ns))
        except FileNotFoundError:
            # Race: file deleted between glob and stat.
            continue
    return latest_ns


def _sessions_evidence_needs_refresh(sessions_dir: Path, evidence_out: Path, *, agent_id: str = DEFAULT_SESSIONS_AGENT_ID) -> bool:
    """Return True when sessions evidence should be regenerated (cheap stat-only gate).

    Rule:
    - refresh if evidence file missing
    - refresh if any transcript mtime > evidence mtime

    If the sessions directory is missing/unreadable, default to refresh=True so we
    surface the issue (and can fall back to last evidence if present).
    """

    evidence_path = _sessions_evidence_path(evidence_out, agent_id=agent_id)
    try:
        evidence_mtime_ns = int(evidence_path.stat().st_mtime_ns)
    except FileNotFoundError:
        return True

    if not sessions_dir.exists():
        return True

    try:
        latest_transcript_ns = _sessions_transcripts_latest_mtime_ns(sessions_dir, agent_id=agent_id)
    except Exception:
        return True

    return latest_transcript_ns > evidence_mtime_ns


def _refresh_sessions_evidence(*, sessions_dir: Path, evidence_out: Path) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "ingest" / "parse_sessions_v1.py"),
        "--agent",
        DEFAULT_SESSIONS_AGENT_ID,
        "--sessions-dir",
        str(sessions_dir),
        "--output",
        str(evidence_out),
    ]
    print(f"REFRESH sessions_evidence agent={DEFAULT_SESSIONS_AGENT_ID} cmd={' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    if proc.returncode == 0:
        # Keep logs small; parse tool prints a short JSON summary.
        out_lines = (proc.stdout or "").strip().splitlines()
        for line in out_lines[-8:]:
            print(line)
        return

    tail_out = (proc.stdout or "").strip().splitlines()[-20:]
    tail_err = (proc.stderr or "").strip().splitlines()[-20:]
    if tail_out:
        print("refresh_stdout_tail:")
        for line in tail_out:
            print(line)
    if tail_err:
        print("refresh_stderr_tail:")
        for line in tail_err:
            print(line, file=sys.stderr)
    raise RuntimeError(f"sessions_evidence_refresh_failed:exit_code={proc.returncode}")


def _compute_backoff_seconds(attempt: int) -> float:
    """attempt is 1-based (the attempt that just failed)."""
    if attempt <= 0:
        attempt = 1
    base = BACKOFF_BASE_S * (2 ** (attempt - 1))
    base = min(base, BACKOFF_CAP_S)
    jitter_mult = 1.0 + random.uniform(-BACKOFF_JITTER_FRAC, BACKOFF_JITTER_FRAC)
    seconds = base * jitter_mult
    seconds = min(seconds, BACKOFF_CAP_S)
    return max(1.0, float(seconds))


def _error_tag(exc: BaseException, *, exit_code: Optional[int] = None) -> str:
    # Avoid storing raw exception messages; keep stable + safe tags.
    name = type(exc).__name__
    if exit_code is None:
        return f"error_type:{name}"
    return f"error_type:{name}:exit_code={int(exit_code)}"


def _requeue_stale_running(conn: sqlite3.Connection, now_iso: str, cutoff_iso: str) -> int:
    cur = conn.execute(
        """
        UPDATE ingest_jobs
        SET
          status = 'failed',
          run_after = ?,
          updated_at = ?,
          last_error = ?,
          last_error_at = ?,
          last_finished_at = ?
        WHERE
          status = 'running'
          AND last_started_at IS NOT NULL
          AND last_started_at < ?
        """,
        (
            now_iso,
            now_iso,
            "error_type:stale_running",
            now_iso,
            now_iso,
            cutoff_iso,
        ),
    )
    return int(cur.rowcount or 0)


def _claim_next_runnable(conn: sqlite3.Connection, now_iso: str) -> Optional[dict[str, Any]]:
    """Atomically claim the next runnable job.

    IMPORTANT: `BEGIN IMMEDIATE` must happen before the SELECT to avoid TOCTOU races
    under concurrent `--once` cron workers.
    """

    now_dt = _parse_iso(now_iso) or _utc_now_dt()
    cutoff_iso = _utc_iso(now_dt - timedelta(seconds=STALE_RUNNING_REQUEUE_S))

    conn.execute("BEGIN IMMEDIATE")
    try:
        _requeue_stale_running(conn, now_iso=now_iso, cutoff_iso=cutoff_iso)

        row = conn.execute(
            """
            SELECT
              job_id, dedupe_key, created_at, updated_at,
              source, job_type, group_id, lane,
              session_key, requested_ts,
              status, run_after,
              attempts, max_attempts,
              payload_json
            FROM ingest_jobs
            WHERE
              status IN ('queued', 'failed')
              AND run_after <= ?
              AND attempts < max_attempts
            ORDER BY
              CASE
                WHEN source = 'done' THEN 0
                WHEN source = 'schedule' THEN 1
                ELSE 2
              END,
              run_after ASC,
              created_at ASC
            LIMIT 1
            """,
            (now_iso,),
        ).fetchone()

        if not row:
            conn.commit()
            return None

        next_attempt = int(row["attempts"] or 0) + 1
        updated = conn.execute(
            """
            UPDATE ingest_jobs
            SET
              status = 'running',
              updated_at = ?,
              last_started_at = ?,
              attempts = ?
            WHERE
              job_id = ?
              AND status IN ('queued', 'failed')
            """,
            (now_iso, now_iso, next_attempt, row["job_id"]),
        ).rowcount

        if updated != 1:
            conn.rollback()
            return None

        conn.commit()
        job = dict(row)
        job["attempts"] = next_attempt
        job["status"] = "running"
        job["updated_at"] = now_iso
        job["last_started_at"] = now_iso
        return job
    except Exception:
        conn.rollback()
        raise


def _mark_succeeded(conn: sqlite3.Connection, job_id: str, now_iso: str, duration_s: float) -> None:
    conn.execute(
        """
        UPDATE ingest_jobs
        SET
          status = 'succeeded',
          updated_at = ?,
          last_finished_at = ?,
          last_exit_code = 0,
          last_duration_s = ?,
          last_error = NULL,
          last_error_at = NULL
        WHERE job_id = ?
        """,
        (now_iso, now_iso, float(duration_s), job_id),
    )
    conn.commit()


def _mark_failed(
    conn: sqlite3.Connection,
    job_id: str,
    *,
    now_dt: datetime,
    attempts: int,
    max_attempts: int,
    error_tag: str,
    exit_code: Optional[int],
    duration_s: float,
) -> None:
    if attempts >= max_attempts:
        status = "dead"
        run_after = _utc_iso(now_dt)
    else:
        status = "failed"
        backoff_s = _compute_backoff_seconds(attempts)
        run_after = _utc_iso(now_dt + timedelta(seconds=backoff_s))

    now_iso = _utc_iso(now_dt)

    conn.execute(
        """
        UPDATE ingest_jobs
        SET
          status = ?,
          run_after = ?,
          updated_at = ?,
          last_finished_at = ?,
          last_exit_code = ?,
          last_duration_s = ?,
          last_error = ?,
          last_error_at = ?
        WHERE job_id = ?
        """,
        (
            status,
            run_after,
            now_iso,
            now_iso,
            (int(exit_code) if exit_code is not None else None),
            float(duration_s),
            error_tag,
            now_iso,
            job_id,
        ),
    )
    conn.commit()


def _maybe_enqueue_scheduled_sessions(conn: sqlite3.Connection, *, now_dt: datetime, dry_run: bool) -> None:
    # If a scheduled job is already pending (queued/running/failed), don't enqueue another.
    pending = conn.execute(
        """
        SELECT 1
        FROM ingest_jobs
        WHERE
          source = 'schedule'
          AND job_type = 'sessions_incremental'
          AND group_id = ?
          AND status IN ('queued', 'running', 'failed')
        LIMIT 1
        """,
        (SESSIONS_MAIN_GROUP_ID,),
    ).fetchone()
    if pending:
        return

    last_success = conn.execute(
        """
        SELECT MAX(last_finished_at) AS last_finished_at
        FROM ingest_jobs
        WHERE
          source = 'schedule'
          AND job_type = 'sessions_incremental'
          AND group_id = ?
          AND status = 'succeeded'
        """,
        (SESSIONS_MAIN_GROUP_ID,),
    ).fetchone()

    last_finished_iso = (last_success["last_finished_at"] if last_success else None) if last_success else None
    if (not dry_run) and last_finished_iso:
        last_dt = _parse_iso(str(last_finished_iso))
        if last_dt and (now_dt - last_dt).total_seconds() < SCHEDULE_PERIOD_S:
            return

    # Use a stable per-window dedupe key to avoid double-enqueue under concurrent cron.
    window_start_ts = int(now_dt.timestamp() // SCHEDULE_PERIOD_S) * SCHEDULE_PERIOD_S
    window_start_iso = _utc_iso(datetime.fromtimestamp(window_start_ts, tz=timezone.utc))
    dedupe_key = f"schedule:sessions_incremental:{SESSIONS_MAIN_GROUP_ID}:{window_start_iso}"

    payload: dict[str, Any] = {
        "job_type": "sessions_incremental",
        "group_id": SESSIONS_MAIN_GROUP_ID,
        "lane": "primary",
        "incremental": True,
        "overlap": DEFAULT_SESSIONS_OVERLAP_CHUNKS,
        "schedule_window_start": window_start_iso,
        "dry_run": bool(dry_run),
    }

    now_iso = _utc_iso(now_dt)
    conn.execute(
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
            uuid.uuid4().hex,
            dedupe_key,
            now_iso,
            now_iso,
            "schedule",
            "sessions_incremental",
            SESSIONS_MAIN_GROUP_ID,
            "primary",
            None,
            now_iso,
            "queued",
            now_iso,
            0,
            6,
            json.dumps(payload, sort_keys=True),
        ),
    )
    conn.commit()


def _build_command(
    job: dict[str, Any],
    *,
    dry_run: bool,
    sessions_evidence_path: Optional[Path] = None,
) -> list[str]:
    job_type = str(job.get("job_type") or "")
    payload_json = str(job.get("payload_json") or "{}")
    try:
        payload = json.loads(payload_json) if payload_json.strip() else {}
    except Exception:
        payload = {}

    # Job-level dry_run (from enqueuer) OR worker-level dry_run.
    exec_dry_run = bool(dry_run) or bool(payload.get("dry_run"))

    if job_type == "sessions_incremental":
        group_id = str(job.get("group_id") or payload.get("group_id") or "")
        if not group_id:
            raise ValueError("missing_group_id")
        validate_identifier(group_id, "group_id")

        overlap = payload.get("overlap", DEFAULT_SESSIONS_OVERLAP_CHUNKS)
        try:
            overlap_i = int(overlap)
        except Exception:
            overlap_i = DEFAULT_SESSIONS_OVERLAP_CHUNKS

        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "mcp_ingest_sessions.py")]
        if sessions_evidence_path is not None:
            cmd.extend(["--evidence", str(sessions_evidence_path)])
        cmd.extend(
            [
                "--group-id",
                group_id,
                "--incremental",
                "--overlap",
                str(overlap_i),
            ]
        )
        if exec_dry_run:
            cmd.append("--dry-run")
        return cmd

    if job_type == "curated_snapshot_ingest":
        group_id = str(job.get("group_id") or payload.get("group_id") or "s1_curated_refs")
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "curated_snapshot_ingest.py"),
            "--group-id",
            group_id,
        ]
        if exec_dry_run:
            cmd.append("--dry-run")
        return cmd

    raise ValueError(f"unsupported_job_type:{job_type or '(missing)'}")


def _execute_job(
    job: dict[str, Any],
    *,
    dry_run: bool,
    refresh_sessions_evidence: bool,
    force_refresh_sessions_evidence: bool,
    sessions_dir: Path,
    evidence_out: Path,
) -> tuple[bool, Optional[int]]:
    job_type = str(job.get("job_type") or "")
    sessions_evidence_path: Optional[Path] = None

    if job_type == "sessions_incremental":
        sessions_evidence_path = _sessions_evidence_path(evidence_out)

        if refresh_sessions_evidence:
            needs_refresh = bool(force_refresh_sessions_evidence) or _sessions_evidence_needs_refresh(
                sessions_dir,
                evidence_out,
                agent_id=DEFAULT_SESSIONS_AGENT_ID,
            )
            if not needs_refresh:
                print(f"SKIP sessions_evidence_refresh agent={DEFAULT_SESSIONS_AGENT_ID} reason=up_to_date")
            else:
                try:
                    _refresh_sessions_evidence(sessions_dir=sessions_dir, evidence_out=evidence_out)
                except Exception as e:
                    # Per PRD: refresh failure should not fail the job if last evidence exists.
                    if sessions_evidence_path.exists():
                        print(
                            f"WARN sessions_evidence_refresh_failed agent={DEFAULT_SESSIONS_AGENT_ID} "
                            f"action=continue_with_last_evidence {_error_tag(e)}"
                        )
                    else:
                        raise

    cmd = _build_command(
        job,
        dry_run=dry_run,
        sessions_evidence_path=sessions_evidence_path,
    )
    print(f"RUN job_id={job['job_id']} job_type={job['job_type']} group_id={job['group_id']} cmd={' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )

    if proc.returncode != 0:
        # Print output for operator debugging (not persisted).
        tail_out = (proc.stdout or "").strip().splitlines()[-20:]
        tail_err = (proc.stderr or "").strip().splitlines()[-20:]
        if tail_out:
            print("stdout_tail:")
            for line in tail_out:
                print(line)
        if tail_err:
            print("stderr_tail:")
            for line in tail_err:
                print(line, file=sys.stderr)
        return False, int(proc.returncode)

    return True, 0


def _run_once(
    *,
    db_path: Path,
    dry_run: bool,
    no_schedule: bool,
    refresh_sessions_evidence: bool,
    force_refresh_sessions_evidence: bool,
    sessions_dir: Path,
    evidence_out: Path,
) -> int:
    now_dt = _utc_now_dt()
    now_iso = _utc_iso(now_dt)

    try:
        conn = _connect(db_path)
        try:
            _ensure_queue_schema(conn)

            if not no_schedule:
                _maybe_enqueue_scheduled_sessions(conn, now_dt=now_dt, dry_run=dry_run)

            job = _claim_next_runnable(conn, now_iso=now_iso)
        finally:
            conn.close()
    except sqlite3.Error as e:
        print(f"error_type:{type(e).__name__}", file=sys.stderr)
        return 1

    if not job:
        print("No runnable queued jobs")
        return 0

    started = time.monotonic()
    ok = False
    exit_code: Optional[int] = None
    err_tag: Optional[str] = None

    try:
        ok, exit_code = _execute_job(
            job,
            dry_run=dry_run,
            refresh_sessions_evidence=refresh_sessions_evidence,
            force_refresh_sessions_evidence=force_refresh_sessions_evidence,
            sessions_dir=sessions_dir,
            evidence_out=evidence_out,
        )
    except Exception as e:
        ok = False
        exit_code = None
        err_tag = _error_tag(e)
    duration_s = time.monotonic() - started

    try:
        conn2 = _connect(db_path)
        try:
            _ensure_queue_schema(conn2)
            if ok:
                _mark_succeeded(conn2, job_id=str(job["job_id"]), now_iso=_utc_iso(_utc_now_dt()), duration_s=duration_s)
                print(f"SUCCEEDED job_id={job['job_id']} duration_s={duration_s:.2f}")
            else:
                if err_tag is None:
                    err_tag = _error_tag(Exception("subprocess_failed"), exit_code=exit_code)
                _mark_failed(
                    conn2,
                    str(job["job_id"]),
                    now_dt=_utc_now_dt(),
                    attempts=int(job.get("attempts") or 0),
                    max_attempts=int(job.get("max_attempts") or 6),
                    error_tag=err_tag,
                    exit_code=exit_code,
                    duration_s=duration_s,
                )
                print(
                    f"FAILED job_id={job['job_id']} attempts={job.get('attempts')}/{job.get('max_attempts')} "
                    f"exit_code={exit_code} duration_s={duration_s:.2f} {err_tag}"
                )
        finally:
            conn2.close()
    except sqlite3.Error as e:
        print(f"error_type:{type(e).__name__}", file=sys.stderr)
        return 1

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run incremental ingest queue worker")
    ap.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--once", action="store_true", help="Run a single claim/execute cycle and exit")
    ap.add_argument("--dry-run", action="store_true", help="Execute jobs in dry-run mode (safe)")
    ap.add_argument("--no-schedule", action="store_true", help="Do not enqueue the scheduled 30m sessions job")
    ap.add_argument("--sleep", type=float, default=30.0, help="Sleep between loop iterations (non --once)")
    ap.add_argument(
        "--refresh-sessions-evidence",
        dest="refresh_sessions_evidence",
        action="store_true",
        default=True,
        help="Refresh v1 sessions evidence before running sessions_incremental jobs (default: on)",
    )
    ap.add_argument(
        "--no-refresh-sessions-evidence",
        dest="refresh_sessions_evidence",
        action="store_false",
        help="Disable sessions evidence refresh (advanced)",
    )
    ap.add_argument(
        "--force-refresh-sessions-evidence",
        action="store_true",
        help="Force sessions v1 evidence refresh (bypass mtime up-to-date check)",
    )
    ap.add_argument(
        "--sessions-dir",
        type=Path,
        default=DEFAULT_SESSIONS_DIR,
        help="Base path to .clawdbot directory (expects agents/*/sessions/*.jsonl)",
    )
    ap.add_argument(
        "--evidence-out",
        type=Path,
        default=DEFAULT_EVIDENCE_OUT,
        help="Output directory for evidence files (default: evidence/)",
    )
    args = ap.parse_args()

    db_path = Path(args.db_path)
    sessions_dir = Path(args.sessions_dir).expanduser()
    evidence_out = Path(args.evidence_out).expanduser()

    if args.once:
        return _run_once(
            db_path=db_path,
            dry_run=bool(args.dry_run),
            no_schedule=bool(args.no_schedule),
            refresh_sessions_evidence=bool(args.refresh_sessions_evidence),
            force_refresh_sessions_evidence=bool(args.force_refresh_sessions_evidence),
            sessions_dir=sessions_dir,
            evidence_out=evidence_out,
        )

    while True:
        rc = _run_once(
            db_path=db_path,
            dry_run=bool(args.dry_run),
            no_schedule=bool(args.no_schedule),
            refresh_sessions_evidence=bool(args.refresh_sessions_evidence),
            force_refresh_sessions_evidence=bool(args.force_refresh_sessions_evidence),
            sessions_dir=sessions_dir,
            evidence_out=evidence_out,
        )
        if rc != 0:
            return rc
        time.sleep(float(args.sleep))


if __name__ == "__main__":
    raise SystemExit(main())
