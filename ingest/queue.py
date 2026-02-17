"""Shared ingest queue schema and helpers.

Both ``scripts/ingest_trigger_done.py`` and ``scripts/run_incremental_ingest.py`` use the
same queue table.  This module is the single source of truth for the schema DDL, connection
factory, and common validation so the two scripts stay DRY.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

# Identifiers (group_id, session_key, source) must match this pattern.
# Alphanumeric, hyphens, underscores, dots, colons — no path separators, no
# shell metacharacters, no whitespace beyond what a sane key would contain.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@\-]{0,254}$")


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


def validate_identifier(value: str, label: str = "identifier") -> str:
    """Validate that *value* is a safe queue identifier.

    Raises ``ValueError`` with a descriptive message on failure.  Returns the
    (stripped) value on success for convenient inline use::

        group_id = validate_identifier(args.session_key, "session_key")
    """
    value = value.strip()
    if not value:
        raise ValueError(f"{label} must not be empty")
    if not _IDENTIFIER_RE.match(value):
        raise ValueError(
            f"{label} contains disallowed characters "
            f"(must match {_IDENTIFIER_RE.pattern})"
        )
    return value


def connect(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the queue database with sensible defaults."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Idempotently create the queue table and indexes."""
    conn.executescript(QUEUE_SCHEMA_DDL)
