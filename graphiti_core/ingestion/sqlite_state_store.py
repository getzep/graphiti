"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import sqlite3
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from graphiti_core.errors import StaleCommitError, UnitExpiredError
from graphiti_core.ingestion.state_store import (
    FAILED_STATUSES,
    REASON_APPLY_FAILED,
    REASON_EXPIRED,
    STATUS_ADMITTED,
    STATUS_ADMITTING,
    STATUS_ANALYZED,
    STATUS_APPLY_FAILED,
    STATUS_APPLYING,
    STATUS_COMMITTED,
    STATUS_FAILED,
    TERMINAL_STATUSES,
    Admission,
    AnalysisStart,
    CommitRecover,
    CommitTurn,
    CommitWait,
    DeltaUnit,
    IngestionStateStore,
    is_released,
    is_stale_heartbeat,
)

T = TypeVar('T')

BUSY_TIMEOUT_SECONDS = 5.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS graphiti_ingest_group (
  group_id      TEXT PRIMARY KEY,
  next_seq      INTEGER NOT NULL,
  watermark     INTEGER NOT NULL,
  admit_seq     INTEGER
);
CREATE TABLE IF NOT EXISTS graphiti_ingest_unit (
  group_id      TEXT NOT NULL,
  seq           INTEGER NOT NULL,
  unit_id       TEXT NOT NULL,
  status        TEXT NOT NULL,
  snapshot_seq  INTEGER,
  started_at    REAL,
  ended_at      REAL,
  fence         INTEGER NOT NULL DEFAULT 0,
  deadline      REAL NOT NULL,
  fail_reason   TEXT,
  payload       BLOB,
  PRIMARY KEY (group_id, seq)
);
CREATE INDEX IF NOT EXISTS graphiti_ingest_unit_by_unit_id
  ON graphiti_ingest_unit (group_id, unit_id, seq);
CREATE TABLE IF NOT EXISTS graphiti_ingest_enrich (
  group_id      TEXT NOT NULL,
  node_uuid     TEXT NOT NULL,
  seq           INTEGER NOT NULL,
  PRIMARY KEY (group_id, node_uuid, seq)
);
"""

_TERMINAL_SQL = ', '.join(f"'{status}'" for status in sorted(TERMINAL_STATUSES))


class SQLiteIngestionStateStore(IngestionStateStore):
    """A state store in one SQLite file, with `sqlite3` from the standard library.

    More than one process on one host can use the same file. Each method is one
    `BEGIN IMMEDIATE` transaction that runs in a worker thread.
    """

    def __init__(self, path: str, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            path,
            timeout=BUSY_TIMEOUT_SECONDS,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.execute(f'PRAGMA busy_timeout={int(BUSY_TIMEOUT_SECONDS * 1000)}')
        with self._lock:
            self._conn.executescript(_SCHEMA)

    def _run_sync(self, func: Callable[[sqlite3.Connection], T]) -> T:
        with self._lock:
            conn = self._conn
            conn.execute('BEGIN IMMEDIATE')
            try:
                result = func(conn)
            except BaseException:
                conn.execute('ROLLBACK')
                raise
            conn.execute('COMMIT')
            return result

    async def _run(self, func: Callable[[sqlite3.Connection], T]) -> T:
        return await asyncio.to_thread(self._run_sync, func)

    # --- helpers that run inside a transaction ---

    @staticmethod
    def _group(conn: sqlite3.Connection, group_id: str) -> sqlite3.Row:
        conn.execute(
            'INSERT OR IGNORE INTO graphiti_ingest_group (group_id, next_seq, watermark) '
            'VALUES (?, 1, 0)',
            (group_id,),
        )
        return conn.execute(
            'SELECT * FROM graphiti_ingest_group WHERE group_id = ?', (group_id,)
        ).fetchone()

    @staticmethod
    def _unit(conn: sqlite3.Connection, group_id: str, seq: int) -> sqlite3.Row | None:
        return conn.execute(
            'SELECT * FROM graphiti_ingest_unit WHERE group_id = ? AND seq = ?', (group_id, seq)
        ).fetchone()

    def _live_unit(
        self,
        conn: sqlite3.Connection,
        group_id: str,
        seq: int,
        *,
        applying_exempt: bool = False,
    ) -> sqlite3.Row:
        unit = self._unit(conn, group_id, seq)
        if unit is None or unit['status'] in FAILED_STATUSES:
            raise UnitExpiredError(group_id, seq)
        if unit['deadline'] <= self._clock() and not (
            applying_exempt and unit['status'] == STATUS_APPLYING
        ):
            raise UnitExpiredError(group_id, seq)
        return unit

    @staticmethod
    def _update_unit(conn: sqlite3.Connection, group_id: str, seq: int, **values: Any) -> None:
        assignments = ', '.join(f'{column} = ?' for column in values)
        conn.execute(
            f'UPDATE graphiti_ingest_unit SET {assignments} WHERE group_id = ? AND seq = ?',
            (*values.values(), group_id, seq),
        )

    def _advance_watermark(self, conn: sqlite3.Connection, group_id: str) -> None:
        group = self._group(conn, group_id)
        watermark = group['watermark']
        while watermark + 1 < group['next_seq']:
            unit = self._unit(conn, group_id, watermark + 1)
            if unit is not None and unit['status'] not in TERMINAL_STATUSES:
                break
            watermark += 1
        if watermark != group['watermark']:
            conn.execute(
                'UPDATE graphiti_ingest_group SET watermark = ? WHERE group_id = ?',
                (watermark, group_id),
            )

    def _fail_unit(
        self,
        conn: sqlite3.Connection,
        group_id: str,
        seq: int,
        reason: str,
        *,
        keep_payload: bool,
    ) -> None:
        if keep_payload:
            self._update_unit(
                conn,
                group_id,
                seq,
                status=STATUS_APPLY_FAILED,
                fail_reason=reason,
                ended_at=self._clock(),
            )
        else:
            self._update_unit(
                conn,
                group_id,
                seq,
                status=STATUS_FAILED,
                fail_reason=reason,
                ended_at=self._clock(),
                payload=None,
            )
        conn.execute(
            'DELETE FROM graphiti_ingest_enrich WHERE group_id = ? AND seq = ?', (group_id, seq)
        )
        conn.execute(
            'UPDATE graphiti_ingest_group SET admit_seq = NULL WHERE group_id = ? AND admit_seq = ?',
            (group_id, seq),
        )
        self._advance_watermark(conn, group_id)

    def _delete_old_units(
        self, conn: sqlite3.Connection, group_id: str, retain_seconds: float
    ) -> None:
        group = self._group(conn, group_id)
        row = conn.execute(
            'SELECT MIN(snapshot_seq) AS snapshot FROM graphiti_ingest_unit '
            f'WHERE group_id = ? AND status NOT IN ({_TERMINAL_SQL}) AND snapshot_seq IS NOT NULL',
            (group_id,),
        ).fetchone()
        limit = group['watermark']
        if row['snapshot'] is not None:
            limit = min(limit, row['snapshot'])
        conn.execute(
            'DELETE FROM graphiti_ingest_unit '
            f'WHERE group_id = ? AND seq <= ? AND status IN ({_TERMINAL_SQL}) '
            'AND ended_at IS NOT NULL AND ended_at < ? '
            'AND NOT EXISTS (SELECT 1 FROM graphiti_ingest_enrich e '
            'WHERE e.group_id = graphiti_ingest_unit.group_id '
            'AND e.seq = graphiti_ingest_unit.seq)',
            (group_id, limit, self._clock() - retain_seconds),
        )

    def _commit_unit(
        self, conn: sqlite3.Connection, group_id: str, seq: int, fence: int
    ) -> sqlite3.Row:
        unit = self._unit(conn, group_id, seq)
        if unit is None or unit['status'] in FAILED_STATUSES:
            raise UnitExpiredError(group_id, seq)
        if unit['status'] == STATUS_COMMITTED or unit['fence'] != fence:
            raise StaleCommitError(group_id, seq)
        if unit['deadline'] <= self._clock():
            raise UnitExpiredError(group_id, seq)
        return unit

    # --- interface ---

    async def begin_admit(self, group_id: str, unit_id: str, ttl: float) -> Admission | None:
        def work(conn: sqlite3.Connection) -> Admission | None:
            group = self._group(conn, group_id)
            now = self._clock()
            # A `unit_id` can have more than one unit. Only the last one can be live.
            previous = conn.execute(
                'SELECT seq, status, deadline FROM graphiti_ingest_unit '
                'WHERE group_id = ? AND unit_id = ? ORDER BY seq DESC LIMIT 1',
                (group_id, unit_id),
            ).fetchone()
            if previous is not None:
                if not is_released(previous['status'], previous['deadline'], now):
                    return Admission(seq=previous['seq'], existing=True)
                if previous['status'] not in TERMINAL_STATUSES:
                    self._fail_unit(
                        conn, group_id, previous['seq'], REASON_EXPIRED, keep_payload=False
                    )
                group = self._group(conn, group_id)

            if group['admit_seq'] is not None:
                holder = self._unit(conn, group_id, group['admit_seq'])
                if holder is not None and holder['deadline'] > now:
                    return None
                if holder is not None:
                    self._fail_unit(
                        conn, group_id, holder['seq'], REASON_EXPIRED, keep_payload=False
                    )

            seq = group['next_seq']
            conn.execute(
                'INSERT INTO graphiti_ingest_unit (group_id, seq, unit_id, status, deadline) '
                'VALUES (?, ?, ?, ?, ?)',
                (group_id, seq, unit_id, STATUS_ADMITTING, now + ttl),
            )
            conn.execute(
                'UPDATE graphiti_ingest_group SET next_seq = ?, admit_seq = ? WHERE group_id = ?',
                (seq + 1, seq, group_id),
            )
            return Admission(
                seq=seq, previous_seq=previous['seq'] if previous is not None else None
            )

        return await self._run(work)

    async def end_admit(self, group_id: str, seq: int) -> None:
        def work(conn: sqlite3.Connection) -> None:
            unit = self._live_unit(conn, group_id, seq)
            if unit['status'] == STATUS_ADMITTING:
                self._update_unit(conn, group_id, seq, status=STATUS_ADMITTED)
            conn.execute(
                'UPDATE graphiti_ingest_group SET admit_seq = NULL '
                'WHERE group_id = ? AND admit_seq = ?',
                (group_id, seq),
            )

        await self._run(work)

    async def start_analysis(self, group_id: str, seq: int) -> AnalysisStart:
        def work(conn: sqlite3.Connection) -> AnalysisStart:
            group = self._group(conn, group_id)
            self._live_unit(conn, group_id, seq)
            started_at = self._clock()
            self._update_unit(
                conn, group_id, seq, snapshot_seq=group['watermark'], started_at=started_at
            )
            return AnalysisStart(snapshot_seq=group['watermark'], started_at=started_at)

        return await self._run(work)

    async def save_change_set(self, group_id: str, seq: int, payload: bytes) -> None:
        def work(conn: sqlite3.Connection) -> None:
            unit = self._live_unit(conn, group_id, seq, applying_exempt=True)
            if unit['status'] != STATUS_ADMITTED:
                # A second analysis does not replace a stored payload.
                raise StaleCommitError(group_id, seq)
            self._update_unit(conn, group_id, seq, payload=payload, status=STATUS_ANALYZED)

        await self._run(work)

    async def heartbeat(
        self, group_id: str, seq: int, ttl: float, fence: int | None = None
    ) -> None:
        def work(conn: sqlite3.Connection) -> None:
            unit = self._live_unit(conn, group_id, seq, applying_exempt=True)
            if is_stale_heartbeat(unit['status'], unit['fence'], fence):
                raise StaleCommitError(group_id, seq)
            self._update_unit(conn, group_id, seq, deadline=self._clock() + ttl)

        await self._run(work)

    async def begin_commit(
        self, group_id: str, seq: int, ttl: float
    ) -> CommitTurn | CommitWait | CommitRecover:
        def work(conn: sqlite3.Connection) -> CommitTurn | CommitWait | CommitRecover:
            group = self._group(conn, group_id)
            unit = self._live_unit(conn, group_id, seq, applying_exempt=True)
            if unit['status'] == STATUS_COMMITTED:
                raise StaleCommitError(group_id, seq)

            now = self._clock()
            earlier_units = conn.execute(
                'SELECT * FROM graphiti_ingest_unit '
                f'WHERE group_id = ? AND seq > ? AND seq < ? AND status NOT IN ({_TERMINAL_SQL}) '
                'ORDER BY seq',
                (group_id, group['watermark'], seq),
            ).fetchall()
            for earlier in earlier_units:
                if earlier['deadline'] > now:
                    return CommitWait()
                if earlier['status'] == STATUS_APPLYING and earlier['payload'] is not None:
                    fence = earlier['fence'] + 1
                    self._update_unit(
                        conn, group_id, earlier['seq'], fence=fence, deadline=now + ttl
                    )
                    return CommitRecover(
                        seq=earlier['seq'], fence=fence, final_payload=bytes(earlier['payload'])
                    )
                self._fail_unit(conn, group_id, earlier['seq'], REASON_EXPIRED, keep_payload=False)

            if unit['status'] == STATUS_APPLYING and unit['deadline'] > now:
                # A different attempt applies the final change set of the unit.
                return CommitWait()
            fence = unit['fence'] + 1
            self._update_unit(conn, group_id, seq, fence=fence, deadline=now + ttl)
            final_payload = (
                bytes(unit['payload'])
                if unit['status'] == STATUS_APPLYING and unit['payload'] is not None
                else None
            )
            return CommitTurn(fence=fence, final_payload=final_payload)

        return await self._run(work)

    async def load_delta(
        self, group_id: str, after_seq: int, before_seq: int, since: float
    ) -> list[DeltaUnit]:
        def work(conn: sqlite3.Connection) -> list[DeltaUnit]:
            rows = conn.execute(
                'SELECT seq, status, payload FROM graphiti_ingest_unit '
                'WHERE group_id = ? AND seq < ? AND status IN (?, ?) AND payload IS NOT NULL '
                'AND (seq > ? OR (ended_at IS NOT NULL AND ended_at >= ?)) ORDER BY seq',
                (group_id, before_seq, STATUS_COMMITTED, STATUS_APPLY_FAILED, after_seq, since),
            ).fetchall()
            return [
                DeltaUnit(seq=row['seq'], status=row['status'], payload=bytes(row['payload']))
                for row in rows
            ]

        return await self._run(work)

    async def save_final(self, group_id: str, seq: int, fence: int, payload: bytes) -> None:
        def work(conn: sqlite3.Connection) -> None:
            self._commit_unit(conn, group_id, seq, fence)
            self._update_unit(conn, group_id, seq, payload=payload, status=STATUS_APPLYING)

        await self._run(work)

    async def finish_commit(
        self,
        group_id: str,
        seq: int,
        fence: int,
        node_uuids: list[str],
        retain_seconds: float,
    ) -> None:
        def work(conn: sqlite3.Connection) -> None:
            self._commit_unit(conn, group_id, seq, fence)
            self._update_unit(conn, group_id, seq, status=STATUS_COMMITTED, ended_at=self._clock())
            conn.executemany(
                'INSERT OR IGNORE INTO graphiti_ingest_enrich (group_id, node_uuid, seq) '
                'VALUES (?, ?, ?)',
                [(group_id, node_uuid, seq) for node_uuid in dict.fromkeys(node_uuids)],
            )
            self._advance_watermark(conn, group_id)
            self._delete_old_units(conn, group_id, retain_seconds)

        await self._run(work)

    async def fail(self, group_id: str, seq: int, reason: str, fence: int | None = None) -> None:
        def work(conn: sqlite3.Connection) -> None:
            unit = self._unit(conn, group_id, seq)
            if unit is None or unit['status'] in TERMINAL_STATUSES:
                return
            if unit['status'] == STATUS_APPLYING:
                if reason != REASON_APPLY_FAILED:
                    raise ValueError(
                        'fail is not permitted for a unit with a stored final change set, '
                        f'with the exception of the reason {REASON_APPLY_FAILED!r}'
                    )
                if fence is None or fence != unit['fence']:
                    raise StaleCommitError(group_id, seq)
                self._fail_unit(conn, group_id, seq, reason, keep_payload=True)
                return
            self._fail_unit(conn, group_id, seq, reason, keep_payload=False)

        await self._run(work)

    async def ready_enrich(self, group_id: str, seq: int) -> list[str]:
        def work(conn: sqlite3.Connection) -> list[str]:
            self._live_unit(conn, group_id, seq)
            conn.execute(
                'DELETE FROM graphiti_ingest_enrich WHERE group_id = ? AND NOT EXISTS ('
                'SELECT 1 FROM graphiti_ingest_unit u WHERE u.group_id = graphiti_ingest_enrich.group_id '
                'AND u.seq = graphiti_ingest_enrich.seq AND u.deadline > ?)',
                (group_id, self._clock()),
            )
            rows = conn.execute(
                'SELECT e.node_uuid FROM graphiti_ingest_enrich e '
                'WHERE e.group_id = ? AND e.seq = ? AND NOT EXISTS ('
                'SELECT 1 FROM graphiti_ingest_enrich b WHERE b.group_id = e.group_id '
                'AND b.node_uuid = e.node_uuid AND b.seq < e.seq) ORDER BY e.rowid',
                (group_id, seq),
            ).fetchall()
            return [row['node_uuid'] for row in rows]

        return await self._run(work)

    async def finish_enrich(self, group_id: str, seq: int, node_uuids: list[str]) -> None:
        def work(conn: sqlite3.Connection) -> None:
            conn.executemany(
                'DELETE FROM graphiti_ingest_enrich WHERE group_id = ? AND seq = ? AND node_uuid = ?',
                [(group_id, seq, node_uuid) for node_uuid in node_uuids],
            )

        await self._run(work)

    async def close(self) -> None:
        with self._lock:
            self._conn.close()
