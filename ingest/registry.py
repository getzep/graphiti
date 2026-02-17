#!/usr/bin/env python3
"""
Ingest Registry - SQLite-based tracking for incremental ingestion.

Tracks:
- Per-source watermarks (last ingested timestamp/index)
- Per-chunk content hashes for deduplication
- Constantine exclusions (persisted conversation IDs to filter)

Database: tools/graphiti/state/ingest_registry.db
"""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence


DEFAULT_DB_PATH = Path(__file__).parent.parent / "state" / "ingest_registry.db"

EPISODE_UUID_NAMESPACE = uuid.UUID("ea9f6f6d-b8f4-4fb2-b37f-9ef4f71ab9a4")
EXTRACTION_STATUS_QUEUED = "queued"
EXTRACTION_STATUS_SUCCEEDED = "succeeded"
EXTRACTION_STATUS_FAILED = "failed"
EXTRACTION_STATUSES = (
    EXTRACTION_STATUS_QUEUED,
    EXTRACTION_STATUS_SUCCEEDED,
    EXTRACTION_STATUS_FAILED,
)


def utc_now_iso() -> str:
    """Current UTC timestamp in second precision, ISO-8601 Z format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class SourceState:
    """State for a single source (chatgpt:conversationId or session:agentId:sessionId)."""
    source_key: str
    source_type: str  # "chatgpt" or "session"
    watermark: Optional[float]  # timestamp for chatgpt, line_number for sessions
    watermark_str: Optional[str]  # human-readable representation
    overlap_window: int  # messages/lines to re-process for overlap
    last_ingested_at: Optional[str]
    chunk_count: int


@dataclass
class ChunkRecord:
    """Record for an ingested chunk."""
    chunk_uuid: str
    source_key: str
    chunk_key: str
    content_hash: str
    evidence_id: str
    ingested_at: str


@dataclass
class CuratedFileState:
    """State for a curated reference file (hash-gated snapshot ingest)."""

    path: str
    last_hash: Optional[str]
    last_snapshot_ts: Optional[str]
    last_ingested_at: Optional[str]


@dataclass
class ExtractionRecord:
    """Lifecycle record for a queued Graphiti extraction episode."""

    group_id: str
    episode_uuid: str
    chunk_uuid: Optional[str]
    source_key: Optional[str]
    chunk_key: Optional[str]
    status: str
    first_queued_at: str
    last_queued_at: str
    last_processing_at: Optional[str]
    last_succeeded_at: Optional[str]
    last_failed_at: Optional[str]
    last_failure_reason: Optional[str]
    queue_count: int
    success_count: int
    failure_count: int
    updated_at: str


class IngestRegistry:
    """SQLite-backed registry for incremental ingest tracking."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self._conn() as conn:
            conn.executescript("""
                -- Source watermark tracking
                CREATE TABLE IF NOT EXISTS sources (
                    source_key TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,  -- 'chatgpt' or 'session'
                    watermark REAL,             -- float timestamp or line number
                    watermark_str TEXT,         -- human-readable
                    overlap_window INTEGER DEFAULT 0,
                    last_ingested_at TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                -- Per-chunk deduplication
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_uuid TEXT PRIMARY KEY,
                    source_key TEXT NOT NULL,
                    chunk_key TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    evidence_id TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    FOREIGN KEY (source_key) REFERENCES sources(source_key)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_key);
                CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);

                -- Constantine exclusions (persisted conversation IDs to always filter)
                CREATE TABLE IF NOT EXISTS exclusions (
                    conversation_id TEXT PRIMARY KEY,
                    reason TEXT NOT NULL,  -- 'constantine_cutoff', 'manual', etc.
                    title TEXT,
                    create_time REAL,
                    excluded_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                -- Curated reference files: content-hash gating + snapshot bookkeeping
                CREATE TABLE IF NOT EXISTS curated_files (
                    path TEXT PRIMARY KEY,
                    last_hash TEXT,
                    last_snapshot_ts TEXT,
                    last_ingested_at TEXT
                );

                -- Extraction lifecycle tracking keyed by (group_id, episode_uuid)
                CREATE TABLE IF NOT EXISTS extraction_tracking (
                    group_id TEXT NOT NULL,
                    episode_uuid TEXT NOT NULL,
                    chunk_uuid TEXT,
                    source_key TEXT,
                    chunk_key TEXT,
                    status TEXT NOT NULL DEFAULT 'queued',
                    first_queued_at TEXT NOT NULL,
                    last_queued_at TEXT NOT NULL,
                    last_processing_at TEXT,
                    last_succeeded_at TEXT,
                    last_failed_at TEXT,
                    last_failure_reason TEXT,
                    queue_count INTEGER NOT NULL DEFAULT 1,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (group_id, episode_uuid),
                    FOREIGN KEY (chunk_uuid) REFERENCES chunks(chunk_uuid)
                );

                -- Sync cursors for incremental log replay
                CREATE TABLE IF NOT EXISTS extraction_sync_cursors (
                    cursor_key TEXT PRIMARY KEY,
                    cursor_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

            # Schema migration: ensure chunk_key exists for older registries.
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(chunks)")}
            if "chunk_key" not in cols:
                conn.execute("ALTER TABLE chunks ADD COLUMN chunk_key TEXT")
                conn.execute(
                    "UPDATE chunks SET chunk_key = chunk_uuid "
                    "WHERE chunk_key IS NULL OR chunk_key = ''"
                )

            # Dedupe semantics: per (source_key, chunk_key, content_hash)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_source_chunk ON chunks(source_key, chunk_key)"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_chunks_source_chunk_hash "
                "ON chunks(source_key, chunk_key, content_hash)"
            )

            # Schema migration: curated_files columns (older versions may omit fields).
            curated_cols = {r["name"] for r in conn.execute("PRAGMA table_info(curated_files)")}
            if curated_cols:
                if "last_hash" not in curated_cols:
                    conn.execute("ALTER TABLE curated_files ADD COLUMN last_hash TEXT")
                if "last_snapshot_ts" not in curated_cols:
                    conn.execute("ALTER TABLE curated_files ADD COLUMN last_snapshot_ts TEXT")
                if "last_ingested_at" not in curated_cols:
                    conn.execute("ALTER TABLE curated_files ADD COLUMN last_ingested_at TEXT")

            # Schema migration: extraction_tracking columns.
            extraction_cols = {r["name"] for r in conn.execute("PRAGMA table_info(extraction_tracking)")}
            if extraction_cols:
                if "status" not in extraction_cols:
                    conn.execute(
                        "ALTER TABLE extraction_tracking ADD COLUMN status TEXT NOT NULL DEFAULT 'queued'"
                    )
                if "first_queued_at" not in extraction_cols:
                    conn.execute("ALTER TABLE extraction_tracking ADD COLUMN first_queued_at TEXT")
                if "last_queued_at" not in extraction_cols:
                    conn.execute("ALTER TABLE extraction_tracking ADD COLUMN last_queued_at TEXT")
                if "updated_at" not in extraction_cols:
                    conn.execute("ALTER TABLE extraction_tracking ADD COLUMN updated_at TEXT")
                if "chunk_uuid" not in extraction_cols:
                    conn.execute("ALTER TABLE extraction_tracking ADD COLUMN chunk_uuid TEXT")
                if "source_key" not in extraction_cols:
                    conn.execute("ALTER TABLE extraction_tracking ADD COLUMN source_key TEXT")
                if "chunk_key" not in extraction_cols:
                    conn.execute("ALTER TABLE extraction_tracking ADD COLUMN chunk_key TEXT")
                if "last_processing_at" not in extraction_cols:
                    conn.execute("ALTER TABLE extraction_tracking ADD COLUMN last_processing_at TEXT")
                if "last_succeeded_at" not in extraction_cols:
                    conn.execute("ALTER TABLE extraction_tracking ADD COLUMN last_succeeded_at TEXT")
                if "last_failed_at" not in extraction_cols:
                    conn.execute("ALTER TABLE extraction_tracking ADD COLUMN last_failed_at TEXT")
                if "last_failure_reason" not in extraction_cols:
                    conn.execute("ALTER TABLE extraction_tracking ADD COLUMN last_failure_reason TEXT")
                if "queue_count" not in extraction_cols:
                    conn.execute("ALTER TABLE extraction_tracking ADD COLUMN queue_count INTEGER NOT NULL DEFAULT 1")
                if "success_count" not in extraction_cols:
                    conn.execute("ALTER TABLE extraction_tracking ADD COLUMN success_count INTEGER NOT NULL DEFAULT 0")
                if "failure_count" not in extraction_cols:
                    conn.execute("ALTER TABLE extraction_tracking ADD COLUMN failure_count INTEGER NOT NULL DEFAULT 0")

                conn.execute(
                    "UPDATE extraction_tracking SET status = ? "
                    "WHERE status NOT IN (?, ?, ?) OR status IS NULL OR status = ''",
                    (
                        EXTRACTION_STATUS_QUEUED,
                        EXTRACTION_STATUS_QUEUED,
                        EXTRACTION_STATUS_SUCCEEDED,
                        EXTRACTION_STATUS_FAILED,
                    ),
                )

                conn.execute(
                    "UPDATE extraction_tracking SET queue_count = COALESCE(queue_count, 1), "
                    "success_count = COALESCE(success_count, 0), "
                    "failure_count = COALESCE(failure_count, 0)"
                )

                conn.execute(
                    "UPDATE extraction_tracking SET first_queued_at = "
                    "COALESCE(first_queued_at, last_queued_at, updated_at, datetime('now'))"
                )
                conn.execute(
                    "UPDATE extraction_tracking SET last_queued_at = "
                    "COALESCE(last_queued_at, first_queued_at, updated_at, datetime('now'))"
                )
                conn.execute(
                    "UPDATE extraction_tracking SET updated_at = "
                    "COALESCE(updated_at, last_failed_at, last_succeeded_at, last_processing_at, last_queued_at, datetime('now'))"
                )

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_extraction_tracking_status "
                "ON extraction_tracking(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_extraction_tracking_group_status "
                "ON extraction_tracking(group_id, status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_extraction_tracking_chunk_uuid "
                "ON extraction_tracking(chunk_uuid)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_extraction_tracking_source_chunk "
                "ON extraction_tracking(source_key, chunk_key)"
            )

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Source watermark management
    # ─────────────────────────────────────────────────────────────────────────

    def get_source_state(self, source_key: str) -> Optional[SourceState]:
        """Get current state for a source."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT source_key, source_type, watermark, watermark_str, overlap_window, last_ingested_at FROM sources WHERE source_key = ?",
                (source_key,)
            ).fetchone()
            if not row:
                return None

            chunk_count = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE source_key = ?",
                (source_key,)
            ).fetchone()[0]

            return SourceState(
                source_key=row["source_key"],
                source_type=row["source_type"],
                watermark=row["watermark"],
                watermark_str=row["watermark_str"],
                overlap_window=row["overlap_window"] or 0,
                last_ingested_at=row["last_ingested_at"],
                chunk_count=chunk_count,
            )

    def update_source_watermark(
        self,
        source_key: str,
        source_type: str,
        watermark: float,
        watermark_str: str,
        overlap_window: int = 0,
    ):
        """Update or create source watermark."""
        now = utc_now_iso()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO sources (source_key, source_type, watermark, watermark_str, overlap_window, last_ingested_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_key) DO UPDATE SET
                    watermark = excluded.watermark,
                    watermark_str = excluded.watermark_str,
                    overlap_window = excluded.overlap_window,
                    last_ingested_at = excluded.last_ingested_at
            """, (source_key, source_type, watermark, watermark_str, overlap_window, now))

    def list_sources(self, source_type: Optional[str] = None) -> list[SourceState]:
        """List all tracked sources, optionally filtered by type."""
        with self._conn() as conn:
            if source_type:
                rows = conn.execute(
                    "SELECT source_key, source_type, watermark, watermark_str, overlap_window, last_ingested_at FROM sources WHERE source_type = ? ORDER BY source_key",
                    (source_type,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT source_key, source_type, watermark, watermark_str, overlap_window, last_ingested_at FROM sources ORDER BY source_type, source_key"
                ).fetchall()

            results = []
            for row in rows:
                chunk_count = conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE source_key = ?",
                    (row["source_key"],)
                ).fetchone()[0]
                results.append(SourceState(
                    source_key=row["source_key"],
                    source_type=row["source_type"],
                    watermark=row["watermark"],
                    watermark_str=row["watermark_str"],
                    overlap_window=row["overlap_window"] or 0,
                    last_ingested_at=row["last_ingested_at"],
                    chunk_count=chunk_count,
                ))
            return results

    # ─────────────────────────────────────────────────────────────────────────
    # Chunk deduplication
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute SHA256 hash of content for deduplication."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def compute_chunk_uuid(source_key: str, chunk_key: str, content_hash: str) -> str:
        """Compute deterministic chunk UUID (legacy 32-hex format)."""
        payload = f"{source_key}\n{chunk_key}\n{content_hash}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:32]

    @staticmethod
    def compute_episode_uuid(chunk_uuid: str) -> str:
        """Compute deterministic RFC-4122 UUID for Graphiti episode tracking."""
        return str(uuid.uuid5(EPISODE_UUID_NAMESPACE, str(chunk_uuid)))

    def is_chunk_ingested(self, source_key: str, chunk_key: str, content_hash: str) -> bool:
        """Check if this (source_key, chunk_key, content_hash) triple has been ingested."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM chunks WHERE source_key = ? AND chunk_key = ? AND content_hash = ?",
                (source_key, chunk_key, content_hash),
            ).fetchone()
            return row is not None

    @staticmethod
    def _infer_source_type(source_key: str) -> str:
        """Infer source_type from source_key prefix.

        'chatgpt:...' → 'chatgpt', 'session:...' / 'sessions:...' → 'session',
        otherwise 'unknown'.
        """
        if source_key.startswith("chatgpt:"):
            return "chatgpt"
        if source_key.startswith("session:") or source_key.startswith("sessions:"):
            return "session"
        return "unknown"

    def record_chunk(
        self,
        chunk_uuid: str,
        source_key: str,
        chunk_key: str,
        content_hash: str,
        evidence_id: str,
    ):
        """Record an ingested chunk.

        Also ensures a ``sources`` row exists for *source_key* so that
        ``get_stats()`` (which JOINs chunks ↔ sources) never under-counts.
        """
        now = utc_now_iso()
        source_type = self._infer_source_type(source_key)
        with self._conn() as conn:
            # Ensure a sources row exists (watermark fields left NULL).
            conn.execute("""
                INSERT OR IGNORE INTO sources (source_key, source_type, created_at)
                VALUES (?, ?, ?)
            """, (source_key, source_type, now))
            conn.execute("""
                INSERT OR REPLACE INTO chunks (chunk_uuid, source_key, chunk_key, content_hash, evidence_id, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (chunk_uuid, source_key, chunk_key, content_hash, evidence_id, now))

    def get_chunk_count(self, source_key: Optional[str] = None) -> int:
        """Get total chunk count, optionally filtered by source."""
        with self._conn() as conn:
            if source_key:
                return conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE source_key = ?",
                    (source_key,)
                ).fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # ─────────────────────────────────────────────────────────────────────────
    # Extraction lifecycle tracking (queue/success/failure)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _coerce_statuses(statuses: Optional[Sequence[str]]) -> Optional[list[str]]:
        if statuses is None:
            return None
        items = [str(s).strip().lower() for s in statuses if str(s).strip()]
        if not items:
            return None
        bad = [s for s in items if s not in EXTRACTION_STATUSES]
        if bad:
            raise ValueError(f"Invalid extraction status(es): {', '.join(sorted(set(bad)))}")
        return items

    @staticmethod
    def _row_to_extraction_record(row: sqlite3.Row) -> ExtractionRecord:
        return ExtractionRecord(
            group_id=row["group_id"],
            episode_uuid=row["episode_uuid"],
            chunk_uuid=row["chunk_uuid"],
            source_key=row["source_key"],
            chunk_key=row["chunk_key"],
            status=row["status"],
            first_queued_at=row["first_queued_at"],
            last_queued_at=row["last_queued_at"],
            last_processing_at=row["last_processing_at"],
            last_succeeded_at=row["last_succeeded_at"],
            last_failed_at=row["last_failed_at"],
            last_failure_reason=row["last_failure_reason"],
            queue_count=int(row["queue_count"] or 0),
            success_count=int(row["success_count"] or 0),
            failure_count=int(row["failure_count"] or 0),
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _seed_extraction_row(
        conn: sqlite3.Connection,
        *,
        group_id: str,
        episode_uuid: str,
        event_at: str,
    ) -> None:
        conn.execute(
            """
            INSERT OR IGNORE INTO extraction_tracking (
                group_id,
                episode_uuid,
                status,
                first_queued_at,
                last_queued_at,
                queue_count,
                success_count,
                failure_count,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, 1, 0, 0, ?)
            """,
            (
                group_id,
                episode_uuid,
                EXTRACTION_STATUS_QUEUED,
                event_at,
                event_at,
                event_at,
            ),
        )

    def record_extraction_queued(
        self,
        *,
        group_id: str,
        episode_uuid: str,
        chunk_uuid: Optional[str],
        source_key: Optional[str],
        chunk_key: Optional[str],
        queued_at: Optional[str] = None,
    ) -> None:
        """Record that an extraction episode was queued for Graphiti processing."""

        ts = queued_at or utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO extraction_tracking (
                    group_id,
                    episode_uuid,
                    chunk_uuid,
                    source_key,
                    chunk_key,
                    status,
                    first_queued_at,
                    last_queued_at,
                    queue_count,
                    success_count,
                    failure_count,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 0, 0, ?)
                ON CONFLICT(group_id, episode_uuid) DO UPDATE SET
                    chunk_uuid = COALESCE(excluded.chunk_uuid, extraction_tracking.chunk_uuid),
                    source_key = COALESCE(excluded.source_key, extraction_tracking.source_key),
                    chunk_key = COALESCE(excluded.chunk_key, extraction_tracking.chunk_key),
                    status = excluded.status,
                    last_queued_at = excluded.last_queued_at,
                    queue_count = extraction_tracking.queue_count + 1,
                    updated_at = excluded.updated_at
                """,
                (
                    group_id,
                    episode_uuid,
                    chunk_uuid,
                    source_key,
                    chunk_key,
                    EXTRACTION_STATUS_QUEUED,
                    ts,
                    ts,
                    ts,
                ),
            )

    def mark_extraction_processing(
        self,
        *,
        group_id: str,
        episode_uuid: str,
        event_at: Optional[str] = None,
    ) -> None:
        """Record a queue-service `processing` event for an extraction episode."""

        ts = event_at or utc_now_iso()
        with self._conn() as conn:
            self._seed_extraction_row(conn, group_id=group_id, episode_uuid=episode_uuid, event_at=ts)
            conn.execute(
                """
                UPDATE extraction_tracking
                SET status = ?,
                    last_processing_at = ?,
                    updated_at = ?
                WHERE group_id = ? AND episode_uuid = ?
                """,
                (
                    EXTRACTION_STATUS_QUEUED,
                    ts,
                    ts,
                    group_id,
                    episode_uuid,
                ),
            )

    def mark_extraction_succeeded(
        self,
        *,
        group_id: str,
        episode_uuid: str,
        event_at: Optional[str] = None,
    ) -> None:
        """Record a queue-service `succeeded` event for an extraction episode."""

        ts = event_at or utc_now_iso()
        with self._conn() as conn:
            self._seed_extraction_row(conn, group_id=group_id, episode_uuid=episode_uuid, event_at=ts)
            conn.execute(
                """
                UPDATE extraction_tracking
                SET status = ?,
                    last_succeeded_at = ?,
                    success_count = COALESCE(success_count, 0) + 1,
                    updated_at = ?
                WHERE group_id = ? AND episode_uuid = ?
                """,
                (
                    EXTRACTION_STATUS_SUCCEEDED,
                    ts,
                    ts,
                    group_id,
                    episode_uuid,
                ),
            )

    def mark_extraction_failed(
        self,
        *,
        group_id: str,
        episode_uuid: str,
        failure_reason: Optional[str],
        event_at: Optional[str] = None,
    ) -> None:
        """Record a queue-service `failed` event for an extraction episode."""

        ts = event_at or utc_now_iso()
        with self._conn() as conn:
            self._seed_extraction_row(conn, group_id=group_id, episode_uuid=episode_uuid, event_at=ts)
            conn.execute(
                """
                UPDATE extraction_tracking
                SET status = ?,
                    last_failed_at = ?,
                    last_failure_reason = ?,
                    failure_count = COALESCE(failure_count, 0) + 1,
                    updated_at = ?
                WHERE group_id = ? AND episode_uuid = ?
                """,
                (
                    EXTRACTION_STATUS_FAILED,
                    ts,
                    (failure_reason or "").strip() or None,
                    ts,
                    group_id,
                    episode_uuid,
                ),
            )

    def apply_extraction_event(
        self,
        *,
        group_id: str,
        episode_uuid: str,
        event_type: str,
        event_at: Optional[str] = None,
        failure_reason: Optional[str] = None,
    ) -> None:
        """Apply a parsed queue-service event to extraction lifecycle state."""

        et = str(event_type or "").strip().lower()
        if et == "processing":
            self.mark_extraction_processing(group_id=group_id, episode_uuid=episode_uuid, event_at=event_at)
            return
        if et == "succeeded":
            self.mark_extraction_succeeded(group_id=group_id, episode_uuid=episode_uuid, event_at=event_at)
            return
        if et == "failed":
            self.mark_extraction_failed(
                group_id=group_id,
                episode_uuid=episode_uuid,
                failure_reason=failure_reason,
                event_at=event_at,
            )
            return
        if et == "queued":
            self.record_extraction_queued(
                group_id=group_id,
                episode_uuid=episode_uuid,
                chunk_uuid=None,
                source_key=None,
                chunk_key=None,
                queued_at=event_at,
            )
            return
        raise ValueError(f"Unsupported extraction event_type: {event_type!r}")

    def list_extractions(
        self,
        *,
        statuses: Optional[Sequence[str]] = None,
        group_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[ExtractionRecord]:
        """List extraction lifecycle rows, newest-first."""

        normalized = self._coerce_statuses(statuses)

        where: list[str] = []
        params: list[Any] = []

        if group_id:
            where.append("group_id = ?")
            params.append(group_id)
        if normalized:
            placeholders = ",".join("?" for _ in normalized)
            where.append(f"status IN ({placeholders})")
            params.extend(normalized)

        sql = (
            "SELECT group_id, episode_uuid, chunk_uuid, source_key, chunk_key, status, "
            "first_queued_at, last_queued_at, last_processing_at, last_succeeded_at, "
            "last_failed_at, last_failure_reason, queue_count, success_count, failure_count, updated_at "
            "FROM extraction_tracking"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY updated_at DESC, group_id, episode_uuid"

        if isinstance(limit, int) and limit > 0:
            sql += " LIMIT ?"
            params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_extraction_record(r) for r in rows]

    def get_extraction_stats(self, *, group_id: Optional[str] = None) -> dict[str, Any]:
        """Get extraction lifecycle summary stats."""

        where = ""
        params: list[Any] = []
        if group_id:
            where = "WHERE group_id = ?"
            params.append(group_id)

        with self._conn() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM extraction_tracking {where}",
                params,
            ).fetchone()[0]

            by_status = {
                row["status"]: int(row["cnt"] or 0)
                for row in conn.execute(
                    f"SELECT status, COUNT(*) AS cnt FROM extraction_tracking {where} GROUP BY status",
                    params,
                ).fetchall()
            }

            newest_update = conn.execute(
                f"SELECT MAX(updated_at) FROM extraction_tracking {where}",
                params,
            ).fetchone()[0]

            unresolved_where = "status IN (?, ?)"
            unresolved_params: list[Any] = [EXTRACTION_STATUS_QUEUED, EXTRACTION_STATUS_FAILED]
            if group_id:
                unresolved_where += " AND group_id = ?"
                unresolved_params.append(group_id)

            oldest_unresolved = conn.execute(
                f"SELECT MIN(last_queued_at) FROM extraction_tracking WHERE {unresolved_where}",
                unresolved_params,
            ).fetchone()[0]

        queued = int(by_status.get(EXTRACTION_STATUS_QUEUED) or 0)
        failed = int(by_status.get(EXTRACTION_STATUS_FAILED) or 0)
        succeeded = int(by_status.get(EXTRACTION_STATUS_SUCCEEDED) or 0)

        return {
            "total": int(total or 0),
            "by_status": by_status,
            "queued": queued,
            "failed": failed,
            "succeeded": succeeded,
            "unresolved": queued + failed,
            "newest_update": newest_update,
            "oldest_unresolved": oldest_unresolved,
        }

    def get_replay_groups(
        self,
        *,
        statuses: Sequence[str] = (EXTRACTION_STATUS_FAILED,),
        group_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Return replay candidates grouped by group_id with chunk/source keys."""

        rows = self.list_extractions(statuses=statuses, group_id=group_id, limit=limit)
        grouped: dict[str, dict[str, Any]] = {}

        for row in rows:
            if not row.chunk_key or not row.source_key:
                continue

            grp = grouped.setdefault(
                row.group_id,
                {
                    "group_id": row.group_id,
                    "status_counts": {},
                    "source_keys": set(),
                    "chunk_keys": set(),
                    "episodes": [],
                },
            )
            grp["status_counts"][row.status] = int(grp["status_counts"].get(row.status, 0)) + 1
            grp["source_keys"].add(row.source_key)
            grp["chunk_keys"].add(row.chunk_key)
            grp["episodes"].append(
                {
                    "episode_uuid": row.episode_uuid,
                    "chunk_uuid": row.chunk_uuid,
                    "source_key": row.source_key,
                    "chunk_key": row.chunk_key,
                    "status": row.status,
                    "last_failed_at": row.last_failed_at,
                    "last_failure_reason": row.last_failure_reason,
                    "last_queued_at": row.last_queued_at,
                }
            )

        out: list[dict[str, Any]] = []
        for group in sorted(grouped):
            g = grouped[group]
            episodes = sorted(
                g["episodes"],
                key=lambda e: (
                    e.get("status") != EXTRACTION_STATUS_FAILED,
                    e.get("last_failed_at") or "",
                    e.get("chunk_key") or "",
                ),
            )
            out.append(
                {
                    "group_id": group,
                    "status_counts": dict(sorted(g["status_counts"].items())),
                    "source_keys": sorted(g["source_keys"]),
                    "chunk_keys": sorted(g["chunk_keys"]),
                    "episodes": episodes,
                }
            )

        return out

    def get_sync_cursor(self, cursor_key: str) -> Optional[str]:
        """Get stored extraction sync cursor value."""

        with self._conn() as conn:
            row = conn.execute(
                "SELECT cursor_value FROM extraction_sync_cursors WHERE cursor_key = ?",
                (cursor_key,),
            ).fetchone()
            return row["cursor_value"] if row else None

    def set_sync_cursor(self, cursor_key: str, cursor_value: str) -> None:
        """Set extraction sync cursor value."""

        now = utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO extraction_sync_cursors (cursor_key, cursor_value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cursor_key) DO UPDATE SET
                    cursor_value = excluded.cursor_value,
                    updated_at = excluded.updated_at
                """,
                (cursor_key, cursor_value, now),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Curated reference files (hash-gated snapshot ingest)
    # ─────────────────────────────────────────────────────────────────────────

    def get_curated_file_state(self, path: str) -> Optional[CuratedFileState]:
        """Get current state for a curated file path (absolute path)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT path, last_hash, last_snapshot_ts, last_ingested_at "
                "FROM curated_files WHERE path = ?",
                (path,),
            ).fetchone()
            if not row:
                return None
            return CuratedFileState(
                path=row["path"],
                last_hash=row["last_hash"],
                last_snapshot_ts=row["last_snapshot_ts"],
                last_ingested_at=row["last_ingested_at"],
            )

    def upsert_curated_file_state(
        self,
        path: str,
        *,
        last_hash: str,
        last_snapshot_ts: str,
        last_ingested_at: Optional[str] = None,
    ) -> None:
        """Insert or update curated file state."""
        now = utc_now_iso()
        ingested_at = last_ingested_at or now
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO curated_files (path, last_hash, last_snapshot_ts, last_ingested_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                  last_hash = excluded.last_hash,
                  last_snapshot_ts = excluded.last_snapshot_ts,
                  last_ingested_at = excluded.last_ingested_at
                """,
                (path, last_hash, last_snapshot_ts, ingested_at),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Constantine exclusions
    # ─────────────────────────────────────────────────────────────────────────

    def add_exclusion(
        self,
        conversation_id: str,
        reason: str,
        title: Optional[str] = None,
        create_time: Optional[float] = None,
    ):
        """Add a conversation to the exclusion list."""
        now = utc_now_iso()
        with self._conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO exclusions (conversation_id, reason, title, create_time, excluded_at)
                VALUES (?, ?, ?, ?, ?)
            """, (conversation_id, reason, title, create_time, now))

    def add_exclusions_batch(self, exclusions: list[dict]):
        """Add multiple exclusions at once."""
        now = utc_now_iso()
        with self._conn() as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO exclusions (conversation_id, reason, title, create_time, excluded_at)
                VALUES (?, ?, ?, ?, ?)
            """, [
                (e["conversation_id"], e["reason"], e.get("title"), e.get("create_time"), now)
                for e in exclusions
            ])

    def is_excluded(self, conversation_id: str) -> bool:
        """Check if a conversation is excluded."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM exclusions WHERE conversation_id = ?",
                (conversation_id,)
            ).fetchone()
            return row is not None

    def get_excluded_ids(self) -> set[str]:
        """Get all excluded conversation IDs."""
        with self._conn() as conn:
            rows = conn.execute("SELECT conversation_id FROM exclusions").fetchall()
            return {row[0] for row in rows}

    def get_exclusion_count(self) -> int:
        """Get total exclusion count."""
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM exclusions").fetchone()[0]

    def get_exclusions(self, reason: Optional[str] = None) -> list[dict]:
        """Get all exclusions, optionally filtered by reason."""
        with self._conn() as conn:
            if reason:
                rows = conn.execute(
                    "SELECT conversation_id, reason, title, create_time, excluded_at FROM exclusions WHERE reason = ?",
                    (reason,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT conversation_id, reason, title, create_time, excluded_at FROM exclusions"
                ).fetchall()
            return [dict(row) for row in rows]

    # ─────────────────────────────────────────────────────────────────────────
    # Summary statistics
    # ─────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get summary statistics for the registry."""
        with self._conn() as conn:
            source_count = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            exclusion_count = conn.execute("SELECT COUNT(*) FROM exclusions").fetchone()[0]

            sources_by_type = {}
            for row in conn.execute("SELECT source_type, COUNT(*) as cnt FROM sources GROUP BY source_type"):
                sources_by_type[row["source_type"]] = row["cnt"]

            chunks_by_type = {}
            for row in conn.execute("""
                SELECT s.source_type, COUNT(c.chunk_uuid) as cnt
                FROM chunks c JOIN sources s ON c.source_key = s.source_key
                GROUP BY s.source_type
            """):
                chunks_by_type[row["source_type"]] = row["cnt"]

            latest_ingest = conn.execute(
                "SELECT MAX(last_ingested_at) FROM sources"
            ).fetchone()[0]

            extraction_by_status = {
                row["status"]: int(row["cnt"] or 0)
                for row in conn.execute(
                    "SELECT status, COUNT(*) AS cnt FROM extraction_tracking GROUP BY status"
                ).fetchall()
            }
            extraction_total = int(sum(extraction_by_status.values()))

            return {
                "source_count": source_count,
                "chunk_count": chunk_count,
                "exclusion_count": exclusion_count,
                "sources_by_type": sources_by_type,
                "chunks_by_type": chunks_by_type,
                "latest_ingest": latest_ingest,
                "extraction": {
                    "total": extraction_total,
                    "by_status": extraction_by_status,
                    "queued": int(extraction_by_status.get(EXTRACTION_STATUS_QUEUED) or 0),
                    "succeeded": int(extraction_by_status.get(EXTRACTION_STATUS_SUCCEEDED) or 0),
                    "failed": int(extraction_by_status.get(EXTRACTION_STATUS_FAILED) or 0),
                },
            }


def get_registry(db_path: Optional[Path] = None) -> IngestRegistry:
    """Get the default registry instance."""
    return IngestRegistry(db_path or DEFAULT_DB_PATH)
