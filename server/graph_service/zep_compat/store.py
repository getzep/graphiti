"""Durable state for the Zep-compatible layer.

Graphiti itself owns the knowledge graph. This module owns only the metadata
Zep Cloud keeps outside the graph and that MiroFish depends on:

  * graph registry (so `GET graph/{id}` can 404 for an unknown graph)
  * per-graph ontology (Zep stores it server-side; Graphiti wants it per call)
  * batch + batch item lifecycle (MiroFish polls this and reconciles restarts)

SQLite, because MiroFish's reconciliation paths assume the server remembers a
batch across a lost response — and, after a restart, across a lost process.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid as uuid_lib
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS graphs (
    graph_id    TEXT PRIMARY KEY,
    uuid        TEXT NOT NULL,
    name        TEXT,
    description TEXT,
    time_zone   TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ontologies (
    graph_id     TEXT PRIMARY KEY,
    entity_types TEXT NOT NULL,
    edge_types   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS batches (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id     TEXT NOT NULL UNIQUE,
    status       TEXT NOT NULL,
    metadata     TEXT,
    ignore_roles TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    processed_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS batch_items (
    item_id            TEXT PRIMARY KEY,
    batch_id           TEXT NOT NULL,
    sequence_index     INTEGER NOT NULL,
    status             TEXT NOT NULL,
    graph_id           TEXT,
    episode_uuid       TEXT,
    kind               TEXT NOT NULL DEFAULT 'graph_episode',
    payload            TEXT,
    data_type          TEXT,
    name               TEXT,
    source_description TEXT,
    reference_time     TEXT,
    metadata           TEXT,
    error              TEXT,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    UNIQUE (batch_id, sequence_index)
);

CREATE TABLE IF NOT EXISTS uuid_index (
    uuid     TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    kind     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_items_batch ON batch_items (batch_id, sequence_index);
CREATE INDEX IF NOT EXISTS idx_uuid_graph ON uuid_index (graph_id);
"""

TERMINAL_ITEM_STATES = ('succeeded', 'failed', 'skipped', 'canceled')


def _now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def _dumps(value: Any) -> str | None:
    return None if value is None else json.dumps(value)


def _loads(value: str | None) -> Any:
    return None if value in (None, '') else json.loads(value)


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ':memory:':
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute('PRAGMA journal_mode=WAL')
            self._conn.execute('PRAGMA synchronous=NORMAL')
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- graphs -------------------------------------------------------------

    def create_graph(
        self,
        graph_id: str,
        name: str | None,
        description: str | None,
        time_zone: str | None,
    ) -> dict[str, Any]:
        created = _now()
        graph_uuid = str(uuid_lib.uuid4())
        with self._tx() as conn:
            conn.execute(
                'INSERT OR IGNORE INTO graphs '
                '(graph_id, uuid, name, description, time_zone, created_at) '
                'VALUES (?,?,?,?,?,?)',
                (graph_id, graph_uuid, name, description, time_zone, created),
            )
        found = self.get_graph(graph_id)
        assert found is not None
        return found

    def get_graph(self, graph_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                'SELECT * FROM graphs WHERE graph_id = ?', (graph_id,)
            ).fetchone()
        return dict(row) if row else None

    def delete_graph(self, graph_id: str) -> None:
        with self._tx() as conn:
            conn.execute('DELETE FROM graphs WHERE graph_id = ?', (graph_id,))
            conn.execute('DELETE FROM ontologies WHERE graph_id = ?', (graph_id,))

    # -- ontology -----------------------------------------------------------

    def set_ontology(
        self, graph_id: str, entity_types: list[dict], edge_types: list[dict]
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                'INSERT INTO ontologies (graph_id, entity_types, edge_types, updated_at) '
                'VALUES (?,?,?,?) ON CONFLICT(graph_id) DO UPDATE SET '
                'entity_types=excluded.entity_types, edge_types=excluded.edge_types, '
                'updated_at=excluded.updated_at',
                (graph_id, json.dumps(entity_types), json.dumps(edge_types), _now()),
            )

    def get_ontology(self, graph_id: str) -> tuple[list[dict], list[dict]]:
        with self._lock:
            row = self._conn.execute(
                'SELECT entity_types, edge_types FROM ontologies WHERE graph_id = ?',
                (graph_id,),
            ).fetchone()
        if not row:
            return [], []
        return json.loads(row['entity_types']), json.loads(row['edge_types'])

    def list_graph_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                'SELECT graph_id FROM graphs ORDER BY created_at DESC'
            ).fetchall()
        return [r['graph_id'] for r in rows]

    # -- uuid -> graph index ------------------------------------------------
    #
    # graphiti 0.29.3 maps group_id onto the DATABASE name, so a node or
    # episode UUID is only resolvable inside its own graph's database. Zep's
    # GET graph/node/{uuid} and GET graph/episodes/{uuid} carry no graph_id, so
    # remember which graph each UUID came from as we hand them out.

    def remember_uuids(self, graph_id: str, kind: str, uuids: list[str]) -> None:
        if not uuids:
            return
        with self._tx() as conn:
            conn.executemany(
                'INSERT INTO uuid_index (uuid, graph_id, kind) VALUES (?,?,?) '
                'ON CONFLICT(uuid) DO UPDATE SET graph_id=excluded.graph_id',
                [(u, graph_id, kind) for u in uuids if u],
            )

    def graph_id_for_uuid(self, uuid: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                'SELECT graph_id FROM uuid_index WHERE uuid = ?', (uuid,)
            ).fetchone()
        return row['graph_id'] if row else None

    # -- batches ------------------------------------------------------------

    def create_batch(
        self, metadata: dict[str, Any] | None, ignore_roles: list[str] | None
    ) -> str:
        batch_id = str(uuid_lib.uuid4())
        stamp = _now()
        with self._tx() as conn:
            conn.execute(
                'INSERT INTO batches '
                '(batch_id, status, metadata, ignore_roles, created_at, updated_at) '
                'VALUES (?,?,?,?,?,?)',
                (batch_id, 'draft', _dumps(metadata), _dumps(ignore_roles), stamp, stamp),
            )
        return batch_id

    def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                'SELECT * FROM batches WHERE batch_id = ?', (batch_id,)
            ).fetchone()
        if not row:
            return None
        batch = dict(row)
        batch['metadata'] = _loads(batch['metadata'])
        batch['ignore_roles'] = _loads(batch['ignore_roles'])
        return batch

    def list_batches(
        self, limit: int, cursor: int | None, status: str | None
    ) -> tuple[list[dict[str, Any]], int | None]:
        """Page batches by monotonic `seq`. Returns (batches, next_cursor)."""
        sql = 'SELECT * FROM batches WHERE seq > ?'
        params: list[Any] = [cursor or 0]
        if status:
            sql += ' AND status = ?'
            params.append(status)
        sql += ' ORDER BY seq ASC LIMIT ?'
        params.append(limit + 1)
        with self._lock:
            rows = [dict(r) for r in self._conn.execute(sql, params).fetchall()]
        has_more = len(rows) > limit
        rows = rows[:limit]
        for row in rows:
            row['metadata'] = _loads(row['metadata'])
            row['ignore_roles'] = _loads(row['ignore_roles'])
        next_cursor = rows[-1]['seq'] if (has_more and rows) else None
        return rows, next_cursor

    def set_batch_status(
        self,
        batch_id: str,
        status: str,
        *,
        processed: bool = False,
        completed: bool = False,
    ) -> None:
        stamp = _now()
        sets = ['status = ?', 'updated_at = ?']
        params: list[Any] = [status, stamp]
        if processed:
            sets.append('processed_at = ?')
            params.append(stamp)
        if completed:
            sets.append('completed_at = ?')
            params.append(stamp)
        params.append(batch_id)
        with self._tx() as conn:
            conn.execute(
                f'UPDATE batches SET {", ".join(sets)} WHERE batch_id = ?', params
            )

    def claim_draft_batches(self) -> list[str]:
        """Batches left mid-flight by a crash, so a restart can resume them."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT batch_id FROM batches WHERE status IN ('queued','processing')"
            ).fetchall()
        return [r['batch_id'] for r in rows]

    # -- batch items --------------------------------------------------------

    def add_items(self, batch_id: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Append items, assigning globally increasing sequence_index.

        MiroFish reconciles an ambiguous `batch.add` by listing items and
        asserting the indexes are exactly range(expected_count), so the index
        must be global across the whole batch, not per-request.
        """
        stamp = _now()
        with self._tx() as conn:
            row = conn.execute(
                'SELECT COALESCE(MAX(sequence_index) + 1, 0) AS next FROM batch_items '
                'WHERE batch_id = ?',
                (batch_id,),
            ).fetchone()
            start = int(row['next'])
            created: list[dict[str, Any]] = []
            for offset, item in enumerate(items):
                record = {
                    'item_id': str(uuid_lib.uuid4()),
                    'batch_id': batch_id,
                    'sequence_index': start + offset,
                    'status': 'pending',
                    'graph_id': item.get('graph_id'),
                    # Assign the episode UUID now: MiroFish reads episode_uuid
                    # off the *add* response, long before processing runs.
                    'episode_uuid': str(uuid_lib.uuid4()),
                    'kind': item.get('kind') or 'graph_episode',
                    'payload': item.get('payload'),
                    'data_type': item.get('data_type') or 'text',
                    'name': item.get('name'),
                    'source_description': item.get('source_description'),
                    'reference_time': item.get('reference_time'),
                    'metadata': _dumps(item.get('metadata')),
                    'error': None,
                    'created_at': stamp,
                    'updated_at': stamp,
                }
                conn.execute(
                    'INSERT INTO batch_items (item_id, batch_id, sequence_index, status, '
                    'graph_id, episode_uuid, kind, payload, data_type, name, '
                    'source_description, reference_time, metadata, error, created_at, '
                    'updated_at) VALUES (:item_id,:batch_id,:sequence_index,:status,'
                    ':graph_id,:episode_uuid,:kind,:payload,:data_type,:name,'
                    ':source_description,:reference_time,:metadata,:error,:created_at,'
                    ':updated_at)',
                    record,
                )
                created.append(record)
        return created

    def list_items(
        self, batch_id: str, limit: int, cursor: int | None
    ) -> tuple[list[dict[str, Any]], int | None]:
        """Page items by sequence_index. Cursor is exclusive."""
        start = -1 if cursor is None else cursor
        with self._lock:
            rows = [
                dict(r)
                for r in self._conn.execute(
                    'SELECT * FROM batch_items WHERE batch_id = ? AND sequence_index > ? '
                    'ORDER BY sequence_index ASC LIMIT ?',
                    (batch_id, start, limit + 1),
                ).fetchall()
            ]
        has_more = len(rows) > limit
        rows = rows[:limit]
        for row in rows:
            row['metadata'] = _loads(row['metadata'])
            row['error'] = _loads(row['error'])
        next_cursor = rows[-1]['sequence_index'] if (has_more and rows) else None
        return rows, next_cursor

    def pending_items(self, batch_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                'SELECT * FROM batch_items WHERE batch_id = ? '
                "AND status NOT IN ('succeeded','skipped','canceled') "
                'ORDER BY sequence_index ASC',
                (batch_id,),
            ).fetchall()
        out = []
        for raw in rows:
            row = dict(raw)
            row['metadata'] = _loads(row['metadata'])
            out.append(row)
        return out

    def find_item_by_episode_uuid(self, episode_uuid: str) -> dict[str, Any] | None:
        """Look up a batch item by the episode UUID handed out at add time.

        Needed because episode UUIDs are assigned when items are added, well
        before the episode exists in the graph. A poll for one of those must
        answer "not processed yet" rather than 404 — see the note in
        router.get_episode.
        """
        with self._lock:
            row = self._conn.execute(
                'SELECT * FROM batch_items WHERE episode_uuid = ? LIMIT 1',
                (episode_uuid,),
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item['metadata'] = _loads(item['metadata'])
        item['error'] = _loads(item['error'])
        return item

    def set_item_status(
        self, item_id: str, status: str, error: dict[str, Any] | None = None
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                'UPDATE batch_items SET status = ?, error = ?, updated_at = ? '
                'WHERE item_id = ?',
                (status, _dumps(error), _now(), item_id),
            )

    def mark_items_queued(self, batch_id: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE batch_items SET status = 'queued', updated_at = ? "
                "WHERE batch_id = ? AND status = 'pending'",
                (_now(), batch_id),
            )

    def item_counts(self, batch_id: str) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                'SELECT status, COUNT(*) AS n FROM batch_items WHERE batch_id = ? '
                'GROUP BY status',
                (batch_id,),
            ).fetchall()
        counts = {r['status']: int(r['n']) for r in rows}
        counts['total'] = sum(counts.values())
        return counts
