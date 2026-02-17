#!/usr/bin/env python3
"""Small durable key/value registry for ingest cursors.

This is intentionally separate from ingest/registry.py (the incremental ingest registry)
so we can store simple workflow cursors without coupling to chunk dedupe schemas.

Database (default): state/registry.db
Schema:
  kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)

Stdlib-only.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


def default_registry_db_path() -> Path:
    return Path(__file__).resolve().parents[1] / "state" / "registry.db"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class KVRow:
    key: str
    value: str
    updated_at: str


class KVRegistry:
    """Simple durable key/value store backed by SQLite.

    ⚠️  Security: Values are stored as **plain-text** in SQLite.  Do NOT store
    unencrypted PII, credentials, API keys, or other secrets.  If sensitive
    data must be persisted, encrypt it before calling ``set()`` / ``set_json()``.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_registry_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS kv (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                """
            )

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get(self, key: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute("SELECT key, value, updated_at FROM kv WHERE key=?", (key,)).fetchone()
            return str(row["value"]) if row else None

    def set(self, key: str, value: str) -> None:
        now = _utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO kv(key, value, updated_at)
                VALUES(?,?,?)
                ON CONFLICT(key) DO UPDATE SET
                  value=excluded.value,
                  updated_at=excluded.updated_at
                """,
                (key, value, now),
            )

    def set_many(self, items: dict[str, str]) -> None:
        """Set multiple key/value pairs atomically in a single transaction."""

        if not items:
            return

        now = _utc_now_iso()
        # Stable ordering for determinism in tests/debugging.
        pairs = sorted(((str(k), str(v)) for k, v in items.items()), key=lambda kv: kv[0])

        with self._conn() as conn:
            conn.executemany(
                """
                INSERT INTO kv(key, value, updated_at)
                VALUES(?,?,?)
                ON CONFLICT(key) DO UPDATE SET
                  value=excluded.value,
                  updated_at=excluded.updated_at
                """,
                [(k, v, now) for k, v in pairs],
            )

    def get_json(self, key: str) -> Optional[Any]:
        v = self.get(key)
        if v is None:
            return None
        try:
            return json.loads(v)
        except Exception:
            return None

    def set_json(self, key: str, value: Any) -> None:
        self.set(key, json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
