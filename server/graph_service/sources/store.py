from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import utc_now_iso


class SourceStore:
    """Small SQLite state store for sources, document fingerprints, and jobs."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys = ON')
        connection.execute('PRAGMA journal_mode = WAL')
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'idle',
                    last_sync_at TEXT,
                    last_error TEXT,
                    watermark_ms INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS source_items (
                    source_id TEXT NOT NULL,
                    external_id TEXT NOT NULL,
                    remote_version TEXT,
                    content_hash TEXT NOT NULL,
                    episode_uuid TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source_updated_at TEXT NOT NULL,
                    synced_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    deleted_at TEXT,
                    PRIMARY KEY (source_id, external_id),
                    FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS sync_jobs (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    full_sync INTEGER NOT NULL DEFAULT 0,
                    scanned INTEGER NOT NULL DEFAULT 0,
                    created INTEGER NOT NULL DEFAULT 0,
                    updated INTEGER NOT NULL DEFAULT 0,
                    skipped INTEGER NOT NULL DEFAULT 0,
                    failed INTEGER NOT NULL DEFAULT 0,
                    warnings_json TEXT NOT NULL DEFAULT '[]',
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_sync_jobs_created_at
                    ON sync_jobs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_source_items_synced_at
                    ON source_items(synced_at DESC);
                """
            )
            source_columns = {
                row['name'] for row in connection.execute('PRAGMA table_info(sources)').fetchall()
            }
            if 'connection_id' not in source_columns:
                connection.execute('ALTER TABLE sources ADD COLUMN connection_id TEXT')
            connection.execute(
                'CREATE INDEX IF NOT EXISTS idx_sources_connection_id ON sources(connection_id)'
            )

    @staticmethod
    def _source(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result['config'] = json.loads(result.pop('config_json'))
        result['enabled'] = bool(result['enabled'])
        return result

    @staticmethod
    def _job(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result['full_sync'] = bool(result['full_sync'])
        result['warnings'] = json.loads(result.pop('warnings_json'))
        return result

    def create_source(
        self,
        *,
        kind: str,
        name: str,
        group_id: str,
        connection_id: str | None = None,
        config: dict[str, Any],
        enabled: bool,
    ) -> dict[str, Any]:
        source_id = uuid4().hex
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sources (
                    id, kind, name, group_id, connection_id, config_json, enabled,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    kind,
                    name,
                    group_id,
                    connection_id,
                    json.dumps(config, ensure_ascii=False),
                    int(enabled),
                    now,
                    now,
                ),
            )
        return self.get_source(source_id)

    def get_source(self, source_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute('SELECT * FROM sources WHERE id = ?', (source_id,)).fetchone()
        if row is None:
            raise KeyError(source_id)
        return self._source(row)

    def list_sources(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute('SELECT * FROM sources ORDER BY created_at DESC').fetchall()
        return [self._source(row) for row in rows]

    def update_source(self, source_id: str, values: dict[str, Any]) -> dict[str, Any]:
        allowed = {'name', 'group_id', 'connection_id', 'enabled', 'config'}
        updates: list[str] = []
        parameters: list[Any] = []
        for key, value in values.items():
            if key not in allowed:
                continue
            column = 'config_json' if key == 'config' else key
            if key == 'config':
                value = json.dumps(value, ensure_ascii=False)
            elif key == 'enabled':
                value = int(value)
            updates.append(f'{column} = ?')
            parameters.append(value)
        if not updates:
            return self.get_source(source_id)
        updates.append('updated_at = ?')
        parameters.append(utc_now_iso())
        parameters.append(source_id)
        with self._connect() as connection:
            cursor = connection.execute(
                f'UPDATE sources SET {", ".join(updates)} WHERE id = ?',  # noqa: S608
                parameters,
            )
            if cursor.rowcount == 0:
                raise KeyError(source_id)
        return self.get_source(source_id)

    def sources_using_connection(self, connection_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT * FROM sources WHERE connection_id = ? ORDER BY created_at DESC',
                (connection_id,),
            ).fetchall()
        return [self._source(row) for row in rows]

    def reset_source_fingerprints(self, source_id: str) -> None:
        """Reset incremental state after changing the remote account or resource root."""
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute('DELETE FROM source_items WHERE source_id = ?', (source_id,))
            connection.execute(
                """
                UPDATE sources SET watermark_ms = NULL, last_sync_at = NULL,
                    last_error = NULL, status = 'idle', updated_at = ?
                WHERE id = ?
                """,
                (utc_now_iso(), source_id),
            )

    def delete_source(self, source_id: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute('DELETE FROM sources WHERE id = ?', (source_id,))
            if cursor.rowcount == 0:
                raise KeyError(source_id)

    def set_source_state(
        self,
        source_id: str,
        *,
        status: str,
        last_error: str | None = None,
        last_sync_at: str | None = None,
        watermark_ms: int | None = None,
    ) -> None:
        fields = ['status = ?', 'last_error = ?', 'updated_at = ?']
        parameters: list[Any] = [status, last_error, utc_now_iso()]
        if last_sync_at is not None:
            fields.append('last_sync_at = ?')
            parameters.append(last_sync_at)
        if watermark_ms is not None:
            fields.append('watermark_ms = ?')
            parameters.append(watermark_ms)
        parameters.append(source_id)
        with self._connect() as connection:
            connection.execute(
                f'UPDATE sources SET {", ".join(fields)} WHERE id = ?',  # noqa: S608
                parameters,
            )

    def get_item(self, source_id: str, external_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                'SELECT * FROM source_items WHERE source_id = ? AND external_id = ?',
                (source_id, external_id),
            ).fetchone()
        return dict(row) if row else None

    def list_items(self, source_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM source_items WHERE source_id = ?
                ORDER BY synced_at DESC LIMIT ?
                """,
                (source_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_item(
        self,
        *,
        source_id: str,
        external_id: str,
        remote_version: str,
        content_hash: str,
        episode_uuid: str,
        title: str,
        source_updated_at: str,
    ) -> None:
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO source_items (
                    source_id, external_id, remote_version, content_hash, episode_uuid,
                    title, source_updated_at, synced_at, last_seen_at, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(source_id, external_id) DO UPDATE SET
                    remote_version = excluded.remote_version,
                    content_hash = excluded.content_hash,
                    episode_uuid = excluded.episode_uuid,
                    title = excluded.title,
                    source_updated_at = excluded.source_updated_at,
                    synced_at = excluded.synced_at,
                    last_seen_at = excluded.last_seen_at,
                    deleted_at = NULL
                """,
                (
                    source_id,
                    external_id,
                    remote_version,
                    content_hash,
                    episode_uuid,
                    title,
                    source_updated_at,
                    now,
                    now,
                ),
            )

    def touch_item(self, source_id: str, external_id: str) -> None:
        self.touch_item_metadata(source_id, external_id)

    def touch_item_metadata(
        self,
        source_id: str,
        external_id: str,
        *,
        remote_version: str | None = None,
        title: str | None = None,
        source_updated_at: str | None = None,
    ) -> None:
        fields = ['last_seen_at = ?', 'deleted_at = NULL']
        parameters: list[Any] = [utc_now_iso()]
        for column, value in (
            ('remote_version', remote_version),
            ('title', title),
            ('source_updated_at', source_updated_at),
        ):
            if value is not None:
                fields.append(f'{column} = ?')
                parameters.append(value)
        parameters.extend((source_id, external_id))
        with self._connect() as connection:
            connection.execute(
                f'UPDATE source_items SET {", ".join(fields)} '  # noqa: S608
                'WHERE source_id = ? AND external_id = ?',
                parameters,
            )

    def touch_items(self, source_id: str, external_ids: set[str]) -> None:
        """Mark an inventory batch as seen without changing its content fingerprint."""
        if not external_ids:
            return
        seen_at = utc_now_iso()
        with self._connect() as connection:
            connection.executemany(
                """
                UPDATE source_items SET last_seen_at = ?, deleted_at = NULL
                WHERE source_id = ? AND external_id = ?
                """,
                ((seen_at, source_id, external_id) for external_id in external_ids),
            )

    def mark_items_missing(self, source_id: str, scan_started_at: str) -> int:
        """Tombstone items not observed by a complete source inventory.

        Graph facts are intentionally retained for temporal audit. A later source
        reappearance clears the tombstone through ``touch_item``/``upsert_item``.
        """
        now = utc_now_iso()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE source_items SET deleted_at = ?
                WHERE source_id = ? AND last_seen_at < ? AND deleted_at IS NULL
                """,
                (now, source_id, scan_started_at),
            )
        return cursor.rowcount

    def create_job(self, source_id: str, *, full_sync: bool) -> dict[str, Any]:
        """Create one queued/running job per source, atomically.

        ``BEGIN IMMEDIATE`` makes the read-then-insert decision safe across concurrent
        requests and multiple worker threads sharing the same SQLite file.
        """
        job_id = uuid4().hex
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            active = connection.execute(
                """
                SELECT * FROM sync_jobs
                WHERE source_id = ? AND status IN ('queued', 'running')
                ORDER BY created_at DESC LIMIT 1
                """,
                (source_id,),
            ).fetchone()
            if active is not None:
                if full_sync and not active['full_sync'] and active['status'] == 'queued':
                    connection.execute(
                        'UPDATE sync_jobs SET full_sync = 1 WHERE id = ?', (active['id'],)
                    )
                    active = connection.execute(
                        'SELECT * FROM sync_jobs WHERE id = ?', (active['id'],)
                    ).fetchone()
                return self._job(active)
            connection.execute(
                """
                INSERT INTO sync_jobs (id, source_id, status, full_sync, created_at)
                VALUES (?, ?, 'queued', ?, ?)
                """,
                (job_id, source_id, int(full_sync), utc_now_iso()),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute('SELECT * FROM sync_jobs WHERE id = ?', (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._job(row)

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT * FROM sync_jobs ORDER BY created_at DESC LIMIT ?', (limit,)
            ).fetchall()
        return [self._job(row) for row in rows]

    def active_job_for_source(self, source_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM sync_jobs
                WHERE source_id = ? AND status IN ('queued', 'running')
                ORDER BY created_at DESC LIMIT 1
                """,
                (source_id,),
            ).fetchone()
        return self._job(row) if row else None

    def fail_interrupted_jobs(self) -> int:
        now = utc_now_iso()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE sync_jobs
                SET status = 'failed', error = '服务重启导致任务中断', finished_at = ?
                WHERE status IN ('queued', 'running')
                """,
                (now,),
            )
            connection.execute(
                """
                UPDATE sources SET status = 'error', last_error = '服务重启导致任务中断'
                WHERE status = 'syncing'
                """
            )
        return cursor.rowcount

    def update_job(self, job_id: str, **values: Any) -> dict[str, Any]:
        allowed = {
            'status',
            'scanned',
            'created',
            'updated',
            'skipped',
            'failed',
            'warnings',
            'error',
            'started_at',
            'finished_at',
        }
        updates: list[str] = []
        parameters: list[Any] = []
        for key, value in values.items():
            if key not in allowed:
                continue
            column = 'warnings_json' if key == 'warnings' else key
            if key == 'warnings':
                value = json.dumps(value, ensure_ascii=False)
            updates.append(f'{column} = ?')
            parameters.append(value)
        if updates:
            parameters.append(job_id)
            with self._connect() as connection:
                connection.execute(
                    f'UPDATE sync_jobs SET {", ".join(updates)} WHERE id = ?',  # noqa: S608
                    parameters,
                )
        return self.get_job(job_id)

    def stats(self) -> dict[str, int]:
        with self._connect() as connection:
            sources = connection.execute('SELECT COUNT(*) FROM sources').fetchone()[0]
            items = connection.execute('SELECT COUNT(*) FROM source_items').fetchone()[0]
            running = connection.execute(
                "SELECT COUNT(*) FROM sync_jobs WHERE status IN ('queued', 'running')"
            ).fetchone()[0]
        return {'sources': sources, 'items': items, 'active_jobs': running}
