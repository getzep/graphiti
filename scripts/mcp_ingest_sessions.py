#!/usr/bin/env python3
"""Ingest session transcript evidence into Graphiti via the local Graphiti MCP server.

This script is an *enqueue* tool: it sends evidence chunks to Graphiti MCP's `add_memory`.

Supports:
- v1 session evidence (preferred): ingest/parse_sessions_v1.py
- legacy v0 evidence: ingest/parse_sessions.py

Also supports sharded enqueue for parallelism:
  --shards N --shard-index i

Usage (recommended):
  cd tools/graphiti
  python3 ingest/parse_sessions_v1.py --agent main
  python3 scripts/mcp_ingest_sessions.py --group-id s1_sessions_main --limit 500 --shards 4 --shard-index 0

Notes:
- Requires the Graphiti MCP server running locally (launchd native service or docker compose).
- Uses a strong sanitizer to avoid FalkorDB RediSearch syntax errors.
- Incremental mode uses a single (best-effort) watermark per --group-id (not per session source).
- For s1_sessions_main: large evidence (>10k chars) is deterministically sub-chunked into
  smaller pieces (default 10,000 chars) with :p0/:p1/... key suffixes. This avoids LLM
  context_length_exceeded errors without lossy truncation. Each sub-chunk gets its own
  registry entry for idempotent dedup.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.common import sanitize_for_graphiti
from ingest.registry import get_registry

_logger = logging.getLogger(__name__)

MCP_URL_DEFAULT = 'http://localhost:8000/mcp'
DEFAULT_OVERLAP_CHUNKS = 10  # for incremental mode

# --- Sub-chunking for large evidence (sessions_main) ---
# Default max chars per episode body sent to Graphiti. Evidence chunks exceeding
# this limit are deterministically split into sub-chunks with :p0/:p1/... suffixes.
# This avoids LLM context_length_exceeded errors without lossy truncation.
DEFAULT_SUBCHUNK_SIZE = 10_000

# Group IDs that get automatic sub-chunking when evidence exceeds subchunk_size.
_SUBCHUNK_GROUP_IDS = {'s1_sessions_main'}

# Compiled per-line regex for the inline form:
#   <graphiti-context>...</graphiti-context>  (both tags on the same line)
# The pattern is non-greedy and operates per-line so it is naturally bounded
# to a single line's length — no cross-line backtracking / ReDoS risk.
_GRAPHITI_INLINE_RE = re.compile(
    r'<graphiti-context>.*?</graphiti-context>',
    re.IGNORECASE,
)


def strip_graphiti_context(content: str) -> str:
    """Remove <graphiti-context> ... </graphiti-context> wrappers.

    Handles two forms:
    1. Block form  – opening tag alone on one line, closing tag alone on another.
    2. Inline form – both tags (and content) on the same line.

    Parses line-by-line so we only strip explicit wrapper blocks and avoid fragile,
    over-greedy regex behavior on large payloads.
    """
    lines = content.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        if lines[i].strip().lower() == '<graphiti-context>':
            # Block form: scan for a matching closing wrapper. Cap scan to avoid
            # pathological behavior on malformed input.
            k = i + 1
            found_end = False
            max_scan = min(n, i + 1 + 5000)
            while k < max_scan:
                if lines[k].strip().lower() == '</graphiti-context>':
                    found_end = True
                    break
                k += 1
            if found_end:
                i = k + 1
                continue

        # Inline form: substitute any <graphiti-context>…</graphiti-context>
        # spans that appear within the current line.
        cleaned = _GRAPHITI_INLINE_RE.sub('', lines[i])
        out.append(cleaned)
        i += 1

    return ''.join(out)


def strip_untrusted_metadata(content: str) -> str:
    """Remove untrusted metadata JSON blocks while preventing ReDoS.

    Parses line-by-line to correctly pair backtick fences and avoid early
    termination if the JSON payload itself contains embedded triple backticks
    inside strings. Collapses resulting multiple empty lines to maintain
    paragraph boundaries.
    """
    prefixes = (
        'Conversation info:',
        'Sender (untrusted metadata):',
        'Replied message (untrusted, for context):',
        'Conversation info (untrusted metadata):',
    )

    lines = content.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped_line = line.strip()

        matched = False
        for p in prefixes:
            if stripped_line == p:
                matched = True
                break

        if matched and i + 1 < n and lines[i + 1].strip() == '```json':
            # We found a header followed by ```json. Find the true closing ```
            # which is on its own line. Cap scan length to avoid O(N²) behavior
            # on adversarial payloads.
            k = i + 2
            found_end = False
            max_scan = min(n, i + 2 + 1000)
            while k < max_scan:
                if lines[k].strip() == '```':
                    found_end = True
                    break
                k += 1

            if found_end:
                # Skip all lines from i to k (inclusive).
                i = k + 1
                continue

        out.append(line)
        i += 1

    result = ''.join(out)
    return re.sub(r'\n{3,}', '\n\n', result).strip()


def strip_ingestion_noise(content: str) -> str:
    """Remove known wrapper/metadata noise while preserving user message content."""
    return strip_untrusted_metadata(strip_graphiti_context(content))


def subchunk_evidence(content: str, chunk_key: str, max_chars: int) -> list[tuple[str, str]]:
    """Split content into deterministic sub-chunks if it exceeds max_chars.

    Returns a list of (sub_chunk_key, sub_content) tuples.
    If content fits in a single chunk, returns [(chunk_key, content)].

    Sub-chunk keys use :p0, :p1, ... suffixes for deterministic idempotent keys.
    Strips known ingestion wrappers/noise before sub-chunking, then splits on
    paragraph boundaries (double newline) when possible to keep context coherent.
    Falls back to hard split at max_chars if no good boundary exists.

    Raises:
        ValueError: If max_chars <= 0 (would cause infinite loop).
    """
    if max_chars <= 0:
        raise ValueError(f'max_chars must be positive, got {max_chars}')

    # Strip wrappers/metadata noise before we count length
    content = strip_ingestion_noise(content)

    if len(content) <= max_chars:
        return [(chunk_key, content)]

    parts: list[str] = []
    remaining = content

    while remaining:
        if len(remaining) <= max_chars:
            parts.append(remaining)
            break

        # Try to find a paragraph boundary (double newline) near the split point
        # Search in the last 20% of the allowed window for a clean break
        search_start = int(max_chars * 0.8)
        split_pos = remaining.rfind('\n\n', search_start, max_chars)

        if split_pos == -1:
            # No paragraph break; try single newline
            split_pos = remaining.rfind('\n', search_start, max_chars)

        if split_pos == -1:
            # Hard split at max_chars
            split_pos = max_chars

        parts.append(remaining[:split_pos])
        remaining = remaining[split_pos:].lstrip('\n')

    return [(f'{chunk_key}:p{i}', part) for i, part in enumerate(parts)]


def _validate_mcp_url(url: str) -> str:
    """Validate and return the MCP server URL with basic SSRF hardening.

    Reuses the same validation pattern as om_compressor._llm_chat_base_url().

    Rules:
      - Must be an absolute http(s) URL with a non-empty netloc.
      - Must not embed credentials (user:pass@host).
      - Must not include a query string or fragment (structurally invalid for
        an MCP base URL; would indicate mis-configuration or injection).
      - Cloud metadata IP ranges (169.254.x.x / link-local) are always blocked.
        Loopback (localhost / 127.x.x.x / ::1) and private RFC-1918 ranges are
        allowed because the Graphiti MCP server is expected to run locally or
        on the same private network.

    Raises:
        ValueError: on any validation failure.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"MCP URL must be an absolute http(s) URL with a host, got: {url!r}"
        )
    if parsed.username or parsed.password:
        raise ValueError(
            f"MCP URL must not include embedded credentials: {url!r}"
        )
    # Reject query strings and fragments — structurally invalid for an MCP base URL
    # and a potential indicator of mis-configuration or injection attempt.
    if parsed.query:
        raise ValueError(
            f"MCP URL must not include a query string: {url!r}"
        )
    if parsed.fragment:
        raise ValueError(
            f"MCP URL must not include a fragment: {url!r}"
        )

    # Block cloud metadata / link-local addresses (169.254.x.x, fe80::, …).
    # We intentionally allow loopback and RFC-1918 (MCP server is typically local).
    # IPv6-safe: use parsed.hostname which correctly strips brackets from IPv6
    # literals (e.g. "[::1]:8000" → "::1") rather than naive split(":")[0].
    host = (parsed.hostname or "").strip()
    try:
        import ipaddress as _ipaddress
        addr = _ipaddress.ip_address(host)
        if addr.is_link_local:
            raise ValueError(
                f"MCP URL targets a link-local address (cloud metadata risk): {url!r}"
            )
    except ValueError as exc:
        if "cloud metadata" in str(exc) or "link-local" in str(exc):
            raise
        # Not a numeric IP address — hostname, allowed.

    return url


class MCPClient:
    """Minimal MCP (streamable HTTP) client for Graphiti (stdlib only)."""

    def __init__(self, url: str = MCP_URL_DEFAULT):
        self.url = _validate_mcp_url(url)
        self.session_id: str | None = None
        self.initialized = False

    def _decode(self, content_type: str, body: str, status: int) -> dict[str, Any]:
        ct = (content_type or '').lower()

        if ct.startswith('text/event-stream'):
            data_lines = [
                line[len('data:') :].strip()
                for line in body.splitlines()
                if line.startswith('data:')
            ]
            return json.loads(data_lines[-1]) if data_lines else {}

        return json.loads(body) if body.strip() else {}

    def _http_post_json(self, payload: dict[str, Any], extra_headers: dict[str, str] | None = None):
        data = json.dumps(payload).encode('utf-8')
        headers = {
            'Accept': 'application/json, text/event-stream',
            'Content-Type': 'application/json',
        }
        if extra_headers:
            headers.update(extra_headers)

        req = urllib.request.Request(self.url, data=data, headers=headers, method='POST')

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = getattr(resp, 'status', 200)
                resp_headers = dict(resp.headers.items())
                body = resp.read().decode('utf-8', errors='replace')
                return status, resp_headers, body
        except urllib.error.HTTPError as e:
            status = e.code
            resp_headers = dict(e.headers.items()) if e.headers else {}
            body = e.read().decode('utf-8', errors='replace')
            return status, resp_headers, body

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        extra = {}
        if self.session_id:
            extra['mcp-session-id'] = self.session_id

        status, headers, body = self._http_post_json(payload, extra)

        sid = headers.get('mcp-session-id')
        if sid and not self.session_id:
            self.session_id = sid

        # Retry once if missing session id (Graphiti returns 400 but provides session id in headers)
        if (
            status == 400
            and ('Missing session ID' in body)
            and self.session_id
            and 'mcp-session-id' not in extra
        ):
            status, headers, body = self._http_post_json(
                payload, {'mcp-session-id': self.session_id}
            )

        return self._decode(headers.get('content-type', ''), body, status)

    def initialize(self):
        if self.initialized:
            return

        self._post(
            {
                'jsonrpc': '2.0',
                'id': 1,
                'method': 'initialize',
                'params': {
                    'protocolVersion': '2024-11-05',
                    'capabilities': {},
                    'clientInfo': {'name': 'graphiti-mcp-ingest-sessions', 'version': '2'},
                },
            }
        )

        # notifications/initialized
        self._http_post_json(
            {'jsonrpc': '2.0', 'method': 'notifications/initialized', 'params': {}},
            {'mcp-session-id': self.session_id} if self.session_id else None,
        )

        self.initialized = True

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.initialize()
        return self._post(
            {
                'jsonrpc': '2.0',
                'id': 1,
                'method': 'tools/call',
                'params': {'name': name, 'arguments': arguments},
            }
        )


def get_evidence_timestamp(ev: dict) -> float:
    """Extract a best-effort start timestamp from evidence as float.

    Supports both v0 evidence (timestamp) and v1 evidence (timestamp_range.start).
    """

    ts = ev.get('timestamp')
    if not ts:
        ts = (ev.get('timestamp_range') or {}).get('start')
    if not ts:
        ts = (ev.get('timestampRange') or {}).get('start')

    if not ts:
        return 0.0

    if isinstance(ts, (int, float)):
        return float(ts)

    try:
        dt = datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
        return dt.timestamp()
    except Exception:
        return 0.0


def get_chunk_index(ev: dict) -> int:
    if 'chunkIndex' in ev and ev.get('chunkIndex') is not None:
        try:
            return int(ev.get('chunkIndex') or 0)
        except Exception:
            return 0

    ck = ev.get('chunk_key') or ev.get('chunkKey')
    if isinstance(ck, str):
        m = re.search(r':c(\d+)$', ck)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return 0

    return 0


def apply_shard(evidences: list[dict], shards: int, shard_index: int) -> list[dict]:
    if shards <= 1:
        return evidences
    if shard_index < 0 or shard_index >= shards:
        raise SystemExit(f'Invalid --shard-index {shard_index} for --shards {shards}')
    return [ev for i, ev in enumerate(evidences) if (i % shards) == shard_index]


# ---------------------------------------------------------------------------
# Neo4j helpers for FR-4 source mode
# ---------------------------------------------------------------------------

# Upper bound on Neo4j-eligible messages fetched per run. Prevents runaway
# queries when the graph contains hundreds of thousands of rows.
_NEO4J_FETCH_CEILING = 50_000

# Margin factor: fetch this many messages per requested chunk so Smart Cutter
# has enough context to assemble output chunks.
_MESSAGES_PER_CHUNK_ESTIMATE = 40


def _neo4j_conn_params() -> dict:
    """Return Neo4j connection params from environment."""
    return {
        'uri': os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
        'user': os.environ.get('NEO4J_USER', 'neo4j'),
        'password': os.environ.get('NEO4J_PASSWORD', ''),
        'database': os.environ.get('NEO4J_DATABASE', 'neo4j'),
    }


def _neo4j_driver_or_raise():
    """Return a Neo4j driver or raise RuntimeError if config is missing."""
    try:
        from neo4j import GraphDatabase  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            'neo4j Python driver is required for neo4j source mode. '
            'Install with: pip install neo4j'
        ) from exc

    p = _neo4j_conn_params()
    if not p['password']:
        raise RuntimeError(
            'NEO4J_PASSWORD environment variable is required for neo4j source mode.'
        )

    return GraphDatabase.driver(
        p['uri'],
        auth=(p['user'], p['password']),
        connection_timeout=10,
        max_connection_lifetime=60,
    )


def _fetch_neo4j_messages(limit: int, target_group_id: str) -> list[dict]:
    """Fetch up to *limit* Message nodes eligible for *target_group_id* extraction.

    Eligibility rule (group-aware):
      - include messages never marked (`graphiti_extracted_at IS NULL`), OR
      - include messages explicitly marked for a *different* group_id via
        `graphiti_extracted_group_id`.

    Legacy rows that only have global `graphiti_extracted_at` (without
    `graphiti_extracted_group_id`) are treated as already extracted and remain
    excluded unless operators intentionally backfill the group-id marker.

    Returns dicts with: message_id, content, created_at, content_embedding,
    source_session_id, role. Ordered by created_at ASC for chronological
    Smart Cutter input.
    """
    p = _neo4j_conn_params()
    effective_limit = min(limit, _NEO4J_FETCH_CEILING)

    driver = _neo4j_driver_or_raise()
    with driver, driver.session(database=p['database']) as session:
        rows = session.run(
            """
                MATCH (m:Message)
                WHERE m.graphiti_extracted_at IS NULL
                   OR (
                     m.graphiti_extracted_group_id IS NOT NULL
                     AND m.graphiti_extracted_group_id <> $target_group_id
                   )
                RETURN m.message_id AS message_id,
                       coalesce(m.content, '') AS content,
                       coalesce(m.created_at, '') AS created_at,
                       coalesce(m.content_embedding, []) AS content_embedding,
                       coalesce(m.source_session_id, '') AS source_session_id,
                       coalesce(m.role, 'message') AS role
                ORDER BY m.created_at ASC, m.message_id ASC
                LIMIT $n
                """,
            {'n': effective_limit, 'target_group_id': target_group_id},
        ).data()

    return [
        {
            'message_id': str(r['message_id']),
            'content': str(r['content']),
            'created_at': str(r['created_at']),
            'content_embedding': [float(v) for v in (r.get('content_embedding') or [])],
            'source_session_id': str(r.get('source_session_id') or ''),
            'role': str(r.get('role') or 'message'),
        }
        for r in rows
    ]


def _mark_neo4j_extracted(message_ids: list[str], target_group_id: str) -> None:
    """Mark Message nodes extracted for *target_group_id* in Neo4j."""
    if not message_ids:
        return

    p = _neo4j_conn_params()
    now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    driver = _neo4j_driver_or_raise()
    with driver, driver.session(database=p['database']) as session:
        session.run(
            'MATCH (m:Message) WHERE m.message_id IN $ids '
            'SET m.graphiti_extracted_at = $ts, '
            '    m.graphiti_extracted_group_id = $target_group_id',
            {
                'ids': list(message_ids),
                'ts': now,
                'target_group_id': target_group_id,
            },
        ).consume()


def _build_episode_body(message_ids: list[str], messages_by_id: dict[str, dict]) -> str:
    """Build a text episode body from a Smart Cutter chunk.

    Each message is formatted as ``[ISO-timestamp] role: content``.
    Messages are joined with double newlines to preserve paragraph boundaries.
    Missing messages are silently skipped (defensive; should not occur in normal flow).
    """
    parts: list[str] = []
    for mid in message_ids:
        msg = messages_by_id.get(mid)
        if not msg:
            continue
        ts = (msg.get('created_at') or '')[:19]  # trim to second precision
        role = msg.get('role') or 'message'
        content = strip_ingestion_noise(msg.get('content') or '')
        if ts:
            parts.append(f'[{ts}] {role}: {content}')
        else:
            parts.append(f'{role}: {content}')
    return '\n\n'.join(parts)


def _apply_smart_cutter(messages: list[dict]) -> list[Any]:
    """Apply Smart Cutter + Graphiti lane merge to a message list.

    Messages that lack a valid embedding vector are excluded from the cutter
    input (they cannot influence cosine similarity) but are still covered:
    the cutter may include them in adjacent chunks based on time ordering.

    Returns a list of ChunkBoundary objects (graphiti_lane_merge output).
    """
    from graphiti_core.utils.content_chunking import (  # type: ignore
        SmartCutterConfig,
        chunk_conversation_semantic,
        graphiti_lane_merge,
    )

    # Partition messages into those with a usable embedding and those without.
    cuttable: list[dict] = []
    dim: int = 0

    for msg in messages:
        emb = msg.get('content_embedding') or []
        if emb and len(emb) > 0 and all(isinstance(v, (int, float)) for v in emb):
            if dim == 0:
                dim = len(emb)
            if len(emb) == dim:
                cuttable.append(msg)

    if not cuttable:
        # No messages with valid embeddings — cannot run Smart Cutter.
        return []

    chunks = chunk_conversation_semantic(cuttable, SmartCutterConfig())
    return graphiti_lane_merge(chunks, cuttable)


def _load_manifest(manifest_path: Path) -> dict[str, dict]:
    """Load a JSONL manifest into a dict keyed by chunk_id.

    Each line is a JSON object with at least: chunk_id, message_ids, content.
    Blank lines are silently skipped.  Lines that fail JSON parsing are skipped
    and logged as WARNING (line number + parse error message only; raw payload
    is NOT logged to avoid leaking sensitive session content).
    """
    result: dict[str, dict] = {}
    with manifest_path.open('r', encoding='utf-8') as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                _logger.warning(
                    'manifest %s line %d: skipping malformed JSONL (%s)',
                    manifest_path.name,
                    lineno,
                    exc.msg,
                )
                continue
            cid = obj.get('chunk_id')
            if cid:
                result[str(cid)] = obj
    return result


def _query_neo4j_message_count(args: argparse.Namespace) -> int:
    """Query Neo4j for the total Message node count.

    Returns:
        >= 0: actual message count from Neo4j.
        -1:   could not connect or authenticate (error, not zero messages).
    """
    try:
        from neo4j import GraphDatabase  # type: ignore

        uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
        user = os.environ.get('NEO4J_USER', 'neo4j')
        password = os.environ.get('NEO4J_PASSWORD')
        database = os.environ.get('NEO4J_DATABASE', 'neo4j')

        if not password:
            # Missing credential is inconclusive — treat as unknown, not as zero messages,
            # to avoid false BOOTSTRAP_REQUIRED triggers.
            print(
                'WARNING: NEO4J_PASSWORD not set; cannot verify Neo4j message count.',
                file=sys.stderr,
            )
            return -1

        with GraphDatabase.driver(
            uri,
            auth=(user, password),
            connection_timeout=10,
            max_connection_lifetime=30,
        ) as driver, driver.session(database=database) as session:
            rec = session.run('MATCH (m:Message) RETURN count(m) AS cnt').single()
            if rec is None:
                return 0
            return int(rec.get('cnt', 0) or 0)
    except Exception as exc:
        print(f'WARNING: Could not query Neo4j message count ({type(exc).__name__})', file=sys.stderr)
        return -1


def check_bootstrap_guard(neo4j_message_count: int, evidence_files_exist: bool) -> bool:
    """Return True if BOOTSTRAP_REQUIRED guard should fire.

    The guard fires when Neo4j has zero messages but evidence files exist,
    indicating the graph has not been bootstrapped yet.

    Returns False (don't block) when neo4j_message_count is -1 (connection error),
    printing a warning so the operator knows the check was inconclusive.
    """
    if neo4j_message_count < 0:
        print(
            'WARNING: Could not verify Neo4j state. Proceeding with caution.',
            file=sys.stderr,
        )
        return False
    return neo4j_message_count == 0 and evidence_files_exist


def init_claim_db(path: str) -> sqlite3.Connection:
    """Initialize SQLite claim-state DB."""
    conn = sqlite3.connect(path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS chunk_claims (
            chunk_id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'pending',
            worker_id TEXT,
            claimed_at TEXT,
            completed_at TEXT,
            fail_count INTEGER NOT NULL DEFAULT 0,
            error TEXT
        )
    ''')
    conn.commit()
    return conn


def seed_claims(conn: sqlite3.Connection, chunk_ids: list[str]) -> None:
    """Seed claim DB with pending chunks."""
    for cid in chunk_ids:
        conn.execute(
            'INSERT OR IGNORE INTO chunk_claims (chunk_id, status) VALUES (?, ?)',
            (cid, 'pending'),
        )
    conn.commit()


def claim_chunk(conn: sqlite3.Connection, worker_id: str) -> str | None:
    """Atomically claim one pending chunk. Returns chunk_id or None."""
    now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    conn.execute('BEGIN IMMEDIATE')

    if sqlite3.sqlite_version_info >= (3, 35, 0):
        cursor = conn.execute(
            "UPDATE chunk_claims SET status='claimed', worker_id=?, claimed_at=? "
            "WHERE chunk_id = (SELECT chunk_id FROM chunk_claims WHERE status='pending' LIMIT 1) "
            'RETURNING chunk_id',
            (worker_id, now),
        )
        row = cursor.fetchone()
        conn.commit()
        return row[0] if row else None
    else:
        # Fallback for SQLite < 3.35.0: SELECT then UPDATE
        cursor = conn.execute(
            "SELECT chunk_id FROM chunk_claims WHERE status='pending' LIMIT 1"
        )
        row = cursor.fetchone()
        if row is None:
            conn.commit()
            return None
        chunk_id = row[0]
        conn.execute(
            "UPDATE chunk_claims SET status='claimed', worker_id=?, claimed_at=? "
            "WHERE chunk_id=? AND status='pending'",
            (worker_id, now, chunk_id),
        )
        conn.commit()
        return chunk_id


def _claim_done(conn: sqlite3.Connection, chunk_id: str) -> None:
    """Mark a claimed chunk as successfully done."""
    now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    conn.execute(
        "UPDATE chunk_claims SET status='done', completed_at=? WHERE chunk_id=?",
        (now, chunk_id),
    )
    conn.commit()


def _claim_fail(conn: sqlite3.Connection, chunk_id: str, error: str) -> None:
    """Mark a claimed chunk as failed with an error message."""
    now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    conn.execute(
        "UPDATE chunk_claims SET status='failed', "
        "fail_count=COALESCE(fail_count, 0) + 1, error=?, completed_at=? "
        'WHERE chunk_id=?',
        (error[:500], now, chunk_id),
    )
    conn.commit()


def _claim_sent_not_marked(conn: sqlite3.Connection, chunk_id: str) -> None:
    """Transition a claimed chunk to 'sent_not_marked'.

    Called after add_memory succeeds but before _mark_neo4j_extracted is called.
    This intermediate state lets retry workers skip re-sending to add_memory
    and only retry the Neo4j mark — preventing duplicate memory extraction when
    the Neo4j mark fails transiently.

    Claim-ordering guarantee preserved: the chunk stays in a pending-like state
    until _claim_done() is called after a successful Neo4j mark.
    """
    conn.execute(
        "UPDATE chunk_claims SET status='sent_not_marked', error=NULL WHERE chunk_id=?",
        (chunk_id,),
    )
    conn.commit()


def _snapshot_sent_not_marked(conn: sqlite3.Connection) -> list[str]:
    """Return all chunk IDs currently in 'sent_not_marked' status.

    Called once at the start of Phase A to bound the retry set for this run.
    Processing each chunk at most once prevents a persistently-failing chunk
    from starving Phase B pending work (liveness guarantee).
    """
    cursor = conn.execute(
        "SELECT chunk_id FROM chunk_claims WHERE status='sent_not_marked'"
    )
    return [row[0] for row in cursor.fetchall()]


def _claim_neo4j_retry_targeted(
    conn: sqlite3.Connection, worker_id: str, chunk_id: str
) -> bool:
    """Atomically claim a specific 'sent_not_marked' chunk for Neo4j-mark-only retry.

    Returns True if the chunk was successfully claimed (status was 'sent_not_marked'),
    False if the chunk is no longer in that state (e.g. claimed by another worker).
    Workers must NOT call add_memory for claimed chunks — only _mark_neo4j_extracted.
    """
    now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    conn.execute('BEGIN IMMEDIATE')
    conn.execute(
        "UPDATE chunk_claims SET status='claimed', worker_id=?, claimed_at=? "
        "WHERE chunk_id=? AND status='sent_not_marked'",
        (worker_id, now, chunk_id),
    )
    rows_changed = conn.execute('SELECT changes()').fetchone()[0]
    conn.commit()
    return rows_changed > 0


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for mcp_ingest_sessions."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--mcp-url', default=MCP_URL_DEFAULT)
    ap.add_argument(
        '--evidence',
        default=str(
            Path(__file__).resolve().parents[1]
            / 'evidence'
            / 'sessions_v1'
            / 'main'
            / 'all_evidence.json'
        ),
        help='Evidence JSON file (v1 default: evidence/sessions_v1/<agent>/all_evidence.json)',
    )
    ap.add_argument('--group-id', required=True)
    ap.add_argument('--limit', type=int, default=500)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--sleep', type=float, default=0.02)

    # Sharding (parallel enqueue): take every Nth chunk.
    ap.add_argument('--shards', type=int, default=1, help='Total shard count (default: 1)')
    ap.add_argument(
        '--shard-index', type=int, default=0, help="This worker's shard index [0..shards-1]"
    )

    # Incremental mode options
    ap.add_argument(
        '--incremental',
        action='store_true',
        help='Enable incremental mode: only ingest new/changed chunks beyond a watermark',
    )
    ap.add_argument(
        '--overlap',
        type=int,
        default=DEFAULT_OVERLAP_CHUNKS,
        help=(
            'Number of evidence chunks to overlap for incremental mode'
            f' (default: {DEFAULT_OVERLAP_CHUNKS})'
        ),
    )
    ap.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be ingested without sending to Graphiti',
    )
    ap.add_argument(
        '--force', action='store_true', help='Force re-ingest even if chunk already in registry'
    )
    ap.add_argument(
        '--subchunk-size',
        type=int,
        default=DEFAULT_SUBCHUNK_SIZE,
        help=(
            f'Max chars per sub-chunk for large evidence (default: {DEFAULT_SUBCHUNK_SIZE}). '
            'Only applies to group IDs in _SUBCHUNK_GROUP_IDS (e.g. s1_sessions_main).'
        ),
    )

    # FR-4: source mode
    ap.add_argument(
        '--source-mode',
        choices=['neo4j', 'evidence'],
        default='neo4j',
        help='Source mode: neo4j (default) reads from graph DB; evidence reads from JSON files',
    )

    # FR-10: manifest and claim-based processing
    ap.add_argument(
        '--build-manifest',
        default=None,
        help='Build frozen chunk manifest and write to this path',
    )
    ap.add_argument(
        '--manifest',
        default=None,
        help='Use pre-built manifest file for chunk IDs',
    )
    ap.add_argument(
        '--claim-mode',
        action='store_true',
        help='Enable SQLite claim-based processing for high-throughput batch extraction',
    )
    ap.add_argument(
        '--claim-state-check',
        action='store_true',
        help='Check integrity of claim-state DB and report status',
    )

    return ap


_OM_GROUP_ID_PREFIX = "s1_observational_memory"


def _check_om_path_guard(args: argparse.Namespace) -> None:
    """Part B guardrail: warn (or refuse) when --group-id targets the OM namespace.

    Observational Memory extraction in production MUST go through om_compressor
    (scripts/om_compressor.py), NOT through this script's add_memory MCP path.
    Using add_memory for OM would bypass:
      - OM ontology constraints (MOTIVATES/GENERATES/SUPERSEDES/ADDRESSES/RESOLVES)
      - OM node deduplication + provenance writes
      - om_extractor schema-version pinning
      - Dead-letter isolation for failed OM chunks

    This function prints a prominent warning. Set OM_PATH_GUARD=strict to
    abort instead of warn (production hardening option).

    See docs/runbooks/om-operations.md for the authoritative OM path runbook.
    """
    if not args.group_id.startswith(_OM_GROUP_ID_PREFIX):
        return

    msg = (
        f"\n{'='*70}\n"
        f"⚠  OM PATH GUARD: --group-id '{args.group_id}' looks like an OM namespace.\n"
        f"\n"
        f"   Observational Memory extraction MUST use om_compressor, NOT add_memory:\n"
        f"   uv run python scripts/om_compressor.py --force --max-chunks-per-run 10\n"
        f"\n"
        f"   This script (mcp_ingest_sessions.py) uses Graphiti MCP's add_memory,\n"
        f"   which bypasses OM ontology constraints and provenance tracking.\n"
        f"\n"
        f"   Runbook: docs/runbooks/om-operations.md\n"
        f"{'='*70}\n"
    )
    print(msg, file=sys.stderr)

    strict = os.environ.get("OM_PATH_GUARD", "").strip().lower() == "strict"
    if strict:
        print(
            "OM_PATH_GUARD=strict: refusing to ingest into OM namespace via add_memory path.",
            file=sys.stderr,
        )
        sys.exit(2)


def main():
    ap = build_parser()
    args = ap.parse_args()

    if args.subchunk_size <= 0:
        ap.error('--subchunk-size must be a positive integer')

    # Part B: OM path guard — warn when targeting the OM namespace
    _check_om_path_guard(args)

    # --- FR-10: claim-state-check mode ---
    if args.claim_state_check:
        if not args.manifest and not args.build_manifest:
            ap.error('--claim-state-check requires --manifest or --build-manifest')
        claim_db_path = (args.manifest or args.build_manifest) + '.claims.db'
        if not Path(claim_db_path).exists():
            print(f'Claim DB not found: {claim_db_path}')
            return
        conn = init_claim_db(claim_db_path)
        try:
            cursor = conn.execute(
                'SELECT status, COUNT(*) FROM chunk_claims GROUP BY status'
            )
            rows = cursor.fetchall()
            print('Claim state summary:')
            for status, count in rows:
                print(f'  {status}: {count}')
        finally:
            conn.close()
        return

    # --- FR-4: source mode routing ---
    if args.source_mode == 'neo4j':
        _run_neo4j_mode(args, ap)
    else:
        _run_evidence_mode(args, ap)


def _run_neo4j_mode(args: argparse.Namespace, ap: argparse.ArgumentParser) -> None:
    """Neo4j source mode: read extraction-eligible Message nodes from Neo4j.

    Dispatch order:
    1. --build-manifest  → FR-10 build path: query Neo4j, cut chunks, write JSONL + seed claim DB.
    2. --claim-mode      → FR-10 worker path: claim chunks from manifest, send to MCP, mark Neo4j.
    3. default           → FR-4 inline path: fetch → Smart Cutter → MCP send → mark Neo4j.
    """
    evidence_path = Path(args.evidence)
    evidence_files_exist = evidence_path.exists()

    # BOOTSTRAP_REQUIRED guard: if Neo4j has 0 messages but evidence files exist,
    # the graph hasn't been bootstrapped yet — run evidence mode first.
    neo4j_message_count = _query_neo4j_message_count(args)
    if check_bootstrap_guard(neo4j_message_count, evidence_files_exist):
        print(
            'BOOTSTRAP_REQUIRED: Neo4j has no messages but evidence files exist.\n'
            'Run with --source-mode evidence first to bootstrap the graph,\n'
            'or populate Neo4j before using neo4j source mode.',
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # FR-10 build-manifest path
    # ------------------------------------------------------------------
    if args.build_manifest:
        manifest_path = Path(args.build_manifest)
        print(f'Building manifest from Neo4j (group_id={args.group_id}) …')

        fetch_limit = _NEO4J_FETCH_CEILING  # fetch all extraction-eligible messages
        messages = _fetch_neo4j_messages(fetch_limit, args.group_id)
        print(f'  Fetched {len(messages)} extraction-eligible message(s) from Neo4j')

        if not messages:
            print('  Nothing to manifest. Exiting.')
            if not args.dry_run:
                manifest_path.write_text('', encoding='utf-8')
            return

        messages_by_id: dict[str, dict] = {m['message_id']: m for m in messages}
        chunks = _apply_smart_cutter(messages)
        print(f'  Smart Cutter produced {len(chunks)} chunk(s)')

        if args.dry_run:
            print(f'DRY RUN: would write {len(chunks)} chunk(s) to {manifest_path}')
            for c in chunks[:5]:
                print(f'  chunk {c.chunk_index}: {len(c.message_ids)} msg(s), '
                      f'{c.token_count} tokens, reason={c.boundary_reason}')
            if len(chunks) > 5:
                print(f'  … and {len(chunks) - 5} more')
            return

        # Write JSONL manifest — one line per chunk, self-contained for workers.
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_ids: list[str] = []
        with manifest_path.open('w', encoding='utf-8') as fh:
            for chunk in chunks:
                content = _build_episode_body(chunk.message_ids, messages_by_id)
                record = {
                    'chunk_id': chunk.chunk_id,
                    'chunk_index': chunk.chunk_index,
                    'message_ids': chunk.message_ids,
                    'content': content,
                    'token_count': chunk.token_count,
                    'time_range_start': chunk.time_range_start,
                    'time_range_end': chunk.time_range_end,
                    'boundary_reason': chunk.boundary_reason,
                    'boundary_score': chunk.boundary_score,
                }
                fh.write(json.dumps(record, ensure_ascii=True) + '\n')
                chunk_ids.append(chunk.chunk_id)

        print(f'Wrote manifest with {len(chunk_ids)} chunk(s) to {manifest_path}')

        # Seed claim DB if worker-based processing is requested.
        if args.claim_mode:
            claim_db_path = str(manifest_path) + '.claims.db'
            conn = init_claim_db(claim_db_path)
            try:
                seed_claims(conn, chunk_ids)
                print(f'Seeded claim DB at {claim_db_path} with {len(chunk_ids)} chunk(s)')
            finally:
                conn.close()
        return

    # ------------------------------------------------------------------
    # FR-10 claim-mode worker path
    # ------------------------------------------------------------------
    if args.claim_mode:
        if not args.manifest:
            ap.error('--claim-mode requires --manifest (or --build-manifest to create one)')
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            raise SystemExit(f'Manifest not found: {manifest_path}')

        print(f'Loading manifest from {manifest_path} …')
        manifest = _load_manifest(manifest_path)
        print(f'  {len(manifest)} chunk(s) available in manifest')

        claim_db_path = str(manifest_path) + '.claims.db'

        # ------------------------------------------------------------------
        # Dry-run: read-only scan — do NOT touch claim DB or call claim_chunk.
        # Just report what would be processed without mutating any state.
        # ------------------------------------------------------------------
        if args.dry_run:
            conn_ro = init_claim_db(claim_db_path)
            try:
                pending_rows = conn_ro.execute(
                    "SELECT chunk_id FROM chunk_claims WHERE status='pending'"
                ).fetchall()
                pending_ids = [row[0] for row in pending_rows]
                total_rows = conn_ro.execute(
                    'SELECT COUNT(*) FROM chunk_claims'
                ).fetchone()[0]
            finally:
                conn_ro.close()

            print(f'DRY RUN: claim DB has {total_rows} total row(s), '
                  f'{len(pending_ids)} pending')
            preview = pending_ids[:10]
            for chunk_id in preview:
                chunk_data = manifest.get(chunk_id) or {}
                message_ids_preview: list[str] = chunk_data.get('message_ids') or []
                print(f'  would send chunk {chunk_id} ({len(message_ids_preview)} msg(s)) to MCP')
            if len(pending_ids) > 10:
                print(f'  … and {len(pending_ids) - 10} more pending chunk(s)')
            print('DRY RUN: no claim state was modified.')
            return

        conn = init_claim_db(claim_db_path)

        # ------------------------------------------------------------------
        # Manifest/claim handoff robustness: if claim DB has no pending chunks
        # but the manifest is non-empty, auto-seed missing chunk IDs.
        # This handles: fresh DB, DB loss, or first worker after --build-manifest
        # without --claim-mode.  Already in-progress/done chunks are NOT re-seeded.
        # ------------------------------------------------------------------
        try:
            pending_count = conn.execute(
                "SELECT COUNT(*) FROM chunk_claims WHERE status='pending'"
            ).fetchone()[0]

            if pending_count == 0 and manifest:
                existing_ids = {
                    row[0]
                    for row in conn.execute('SELECT chunk_id FROM chunk_claims').fetchall()
                }
                missing_ids = [cid for cid in manifest if cid not in existing_ids]
                if missing_ids:
                    seed_claims(conn, missing_ids)
                    print(
                        f'  Auto-seeded {len(missing_ids)} missing chunk(s) into claim DB '
                        f'(skipped {len(existing_ids)} already-tracked chunk(s))'
                    )
                elif existing_ids:
                    print(
                        f'  All {len(existing_ids)} manifest chunk(s) already tracked in claim DB '
                        f'(none pending — possibly all done/in-progress).'
                    )
        except Exception as seed_exc:
            print(
                f'WARNING: claim DB auto-seed check failed: {seed_exc}',
                file=sys.stderr,
            )

        client = MCPClient(args.mcp_url)

        ok = errors = 0
        worker_id = f'w{args.shard_index}'

        try:
            # ------------------------------------------------------------------
            # Two-phase loop for idempotent retry safety:
            #
            # Phase A — drain 'sent_not_marked' chunks first.
            #   These chunks already had add_memory called successfully on a
            #   prior attempt; only the Neo4j mark failed.  We must NOT call
            #   add_memory again (would duplicate extraction).  Just retry the
            #   Neo4j mark.
            #
            # Phase B — process fresh 'pending' chunks.
            #   Full pipeline: add_memory → sent_not_marked → mark → done.
            #   Transitioning to 'sent_not_marked' before the Neo4j mark
            #   ensures that if the mark fails the chunk lands in Phase A on
            #   the next run (skip re-send).
            # ------------------------------------------------------------------

            # --- Phase A: Neo4j-mark-only retries ---
            # Snapshot all sent_not_marked IDs upfront so each chunk is attempted at
            # most once per run.  A persistently-failing chunk is put back to
            # sent_not_marked but will NOT be re-visited in this run because the
            # snapshot is bounded — Phase B pending chunks are always reachable.
            phase_a_ids = _snapshot_sent_not_marked(conn)
            for chunk_id in phase_a_ids:
                claimed = _claim_neo4j_retry_targeted(conn, worker_id, chunk_id)
                if not claimed:
                    # Another worker already claimed or completed this chunk; skip.
                    continue

                chunk_data = manifest.get(chunk_id)
                if not chunk_data:
                    _claim_fail(conn, chunk_id, 'chunk_id not found in manifest (neo4j-retry)')
                    errors += 1
                    continue

                message_ids_retry: list[str] = chunk_data.get('message_ids') or []
                try:
                    _mark_neo4j_extracted(message_ids_retry, args.group_id)
                    _claim_done(conn, chunk_id)
                    ok += 1
                    _logger.info('neo4j-retry ok: chunk %s', chunk_id[:12])
                except Exception as neo4j_exc:
                    # Increment fail_count and restore sent_not_marked for future runs.
                    # This chunk will NOT be re-processed in Phase A of the current run
                    # (snapshot was taken upfront), so Phase B is always reachable.
                    conn.execute(
                        "UPDATE chunk_claims SET status='sent_not_marked', "
                        "fail_count=COALESCE(fail_count, 0) + 1, error=? "
                        'WHERE chunk_id=?',
                        (str(neo4j_exc)[:500], chunk_id),
                    )
                    conn.commit()
                    errors += 1
                    _logger.warning(
                        'neo4j-retry still failing for chunk %s (fail_count incremented): %s',
                        chunk_id[:12],
                        neo4j_exc,
                    )

            # --- Phase B: fresh pending chunks ---
            # EVAL-3 fix: honour --limit in claim mode so bounded pilots stay bounded.
            # args.limit <= 0 means unlimited (consistent with evidence-mode semantics).
            phase_b_limit = args.limit if args.limit > 0 else None
            phase_b_count = 0

            while True:
                if phase_b_limit is not None and phase_b_count >= phase_b_limit:
                    _logger.info(
                        "Worker %s: reached --limit %d in claim mode, stopping Phase B",
                        worker_id,
                        phase_b_limit,
                    )
                    print(
                        f"Worker {worker_id}: --limit {phase_b_limit} reached, "
                        f"stopping claim-mode Phase B.",
                        file=sys.stderr,
                    )
                    break

                chunk_id = claim_chunk(conn, worker_id)
                if chunk_id is None:
                    break  # no more pending chunks for this worker

                chunk_data = manifest.get(chunk_id)
                if not chunk_data:
                    _claim_fail(conn, chunk_id, 'chunk_id not found in manifest')
                    errors += 1
                    phase_b_count += 1
                    continue

                message_ids: list[str] = chunk_data.get('message_ids') or []
                content: str = chunk_data.get('content') or ''

                # Note: dry-run is handled above (read-only path). If we reach here,
                # we are always in live mode.

                # --- Send chunk to MCP add_memory ---
                ep_name = f'neo4j:{chunk_data.get("chunk_index", 0)}:{chunk_id[:8]}'
                src_desc = (
                    f'neo4j chunk {chunk_id[:12]} '
                    f'({chunk_data.get("time_range_start", "")[:10]} to '
                    f'{chunk_data.get("time_range_end", "")[:10]})'
                )
                body = sanitize_for_graphiti(content)

                res = client.call_tool(
                    'add_memory',
                    {
                        'name': ep_name,
                        'episode_body': body,
                        'group_id': args.group_id,
                        'source': 'text',
                        'source_description': src_desc[:200],
                    },
                )

                if 'error' in res:
                    err_msg = str(res['error'])
                    print(f'ERROR sending chunk {chunk_id}: {err_msg}', file=sys.stderr)
                    _claim_fail(conn, chunk_id, f'mcp_error: {err_msg[:200]}')
                    errors += 1
                    phase_b_count += 1
                    continue

                # --- Transition to sent_not_marked before Neo4j mark ---
                # If the process dies between here and _claim_done, the next
                # run will find this chunk in Phase A and only retry the mark.
                _claim_sent_not_marked(conn, chunk_id)

                # --- Mark Neo4j messages as extracted ---
                try:
                    _mark_neo4j_extracted(message_ids, args.group_id)
                except Exception as neo4j_exc:
                    # MCP send succeeded; leave at 'sent_not_marked' so the next
                    # run retries only the Neo4j mark (Phase A), not add_memory.
                    # Increment fail_count so persistent failures are observable
                    # (mirrors Phase A accounting for retry completeness).
                    #
                    # Guard: only mutate if the row is still in 'sent_not_marked'
                    # state owned by *this* worker.  If another worker already
                    # resolved the row, rowcount will be 0 and we skip telemetry
                    # overwrite to avoid corrupting a resolved row.
                    cur = conn.execute(
                        "UPDATE chunk_claims SET "
                        "fail_count=COALESCE(fail_count, 0) + 1, error=? "
                        "WHERE chunk_id=? AND status='sent_not_marked' AND worker_id=?",
                        (f'neo4j_mark_failed: {neo4j_exc}'[:500], chunk_id, worker_id),
                    )
                    conn.commit()
                    errors += 1
                    if cur.rowcount == 0:
                        _logger.warning(
                            'Phase-B fail_count update skipped for chunk %s: '
                            'row already resolved by another worker',
                            chunk_id[:12],
                        )
                    else:
                        print(
                            f'WARNING: MCP send ok but Neo4j mark failed for chunk {chunk_id[:12]}: '
                            f'{neo4j_exc}',
                            file=sys.stderr,
                        )
                    continue

                # --- Both MCP send and Neo4j mark succeeded ---
                _claim_done(conn, chunk_id)
                ok += 1
                phase_b_count += 1
                time.sleep(args.sleep)

        finally:
            conn.close()

        print(f'\nWorker {worker_id}: ok={ok} errors={errors}')
        return

    # ------------------------------------------------------------------
    # FR-4 inline path (default neo4j mode — no manifest, no claim DB)
    # ------------------------------------------------------------------
    # Fetch enough messages for Smart Cutter to produce args.limit chunks.
    fetch_limit = min(
        max(args.limit, 1) * _MESSAGES_PER_CHUNK_ESTIMATE + args.offset * _MESSAGES_PER_CHUNK_ESTIMATE,
        _NEO4J_FETCH_CEILING,
    )
    print(f'Neo4j source mode: fetching up to {fetch_limit} extraction-eligible message(s) …')
    messages = _fetch_neo4j_messages(fetch_limit, args.group_id)
    print(f'  Fetched {len(messages)} message(s) from Neo4j')

    if not messages:
        print('  No extraction-eligible messages found. Nothing to do.')
        return

    messages_by_id: dict[str, dict] = {m['message_id']: m for m in messages}
    chunks = _apply_smart_cutter(messages)
    print(f'  Smart Cutter produced {len(chunks)} chunk(s)')

    # Apply --offset and --limit to the chunk list (not the message list).
    page = chunks[args.offset : args.offset + args.limit]
    print(f'  Processing {len(page)} chunk(s) (offset={args.offset} limit={args.limit})')

    if args.dry_run:
        for c in page:
            print(f'  DRY RUN chunk {c.chunk_index}: {len(c.message_ids)} msg(s), '
                  f'{c.token_count} tokens, reason={c.boundary_reason}')
        return

    client = MCPClient(args.mcp_url)
    ok = errors = 0
    extracted_ids: list[str] = []  # accumulated for bulk Neo4j mark

    for i, chunk in enumerate(page, start=1):
        content = _build_episode_body(chunk.message_ids, messages_by_id)
        body = sanitize_for_graphiti(content)
        ep_name = f'neo4j:{chunk.chunk_index}:{chunk.chunk_id[:8]}'
        src_desc = (
            f'neo4j chunk {chunk.chunk_id[:12]} '
            f'({chunk.time_range_start[:10]} to {chunk.time_range_end[:10]})'
        )

        res = client.call_tool(
            'add_memory',
            {
                'name': ep_name,
                'episode_body': body,
                'group_id': args.group_id,
                'source': 'text',
                'source_description': src_desc[:200],
            },
        )

        if 'error' in res:
            print(
                f'[{i}/{len(page)}] ERROR chunk {chunk.chunk_id[:12]}: {res["error"]}',
                file=sys.stderr,
            )
            errors += 1
        else:
            extracted_ids.extend(chunk.message_ids)
            ok += 1
            if i <= 5 or i == len(page) or i % 50 == 0:
                print(f'[{i}/{len(page)}] queued chunk {chunk.chunk_id[:12]} '
                      f'({len(chunk.message_ids)} msg(s))')
            time.sleep(args.sleep)

    # Bulk-mark extracted messages in Neo4j.
    if extracted_ids:
        try:
            _mark_neo4j_extracted(extracted_ids, args.group_id)
            print(
                f'Marked {len(extracted_ids)} message(s) as graphiti_extracted_at '
                f'for group_id={args.group_id} in Neo4j'
            )
        except Exception as exc:
            print(
                f'WARNING: Failed to mark {len(extracted_ids)} message(s) in Neo4j: {exc}',
                file=sys.stderr,
            )

    print(f'\nQueued: {ok} chunk(s) into group_id={args.group_id}')
    if errors:
        print(f'Errors: {errors}')


def _run_evidence_mode(args: argparse.Namespace, ap: argparse.ArgumentParser) -> None:
    """Evidence source mode: the original JSON-file-based ingestion path.

    Uses the already-parsed args from main() — does NOT re-parse sys.argv.
    """
    if getattr(args, 'subchunk_size', 0) <= 0:
        ap.error('--subchunk-size must be a positive integer')

    evidence_path = Path(args.evidence)
    if not evidence_path.exists():
        raise SystemExit(
            f'Evidence file not found: {evidence_path}\n'
            'Run: python3 ingest/parse_sessions_v1.py --agent main  (preferred)\n'
            '  or: python3 ingest/parse_sessions.py --agent main      (legacy)'
        )

    # Use json.load(fp) instead of json.loads(read_text()) to avoid the intermediate
    # string allocation — the JSON array is still loaded into memory but without
    # holding both the raw bytes and the parsed result simultaneously.
    with evidence_path.open('r', encoding='utf-8') as _efp:
        evidences = json.load(_efp)

    # Sort for consistent ordering.
    evidences.sort(
        key=lambda e: (
            get_evidence_timestamp(e),
            (e.get('source_key') or ''),
            get_chunk_index(e),
            (e.get('chunk_key') or e.get('chunkKey') or ''),
        )
    )

    registry = get_registry()

    # Registry source key for this *enqueue stream* (not per session).
    stream_source_key = f'sessions:{args.group_id}'

    # Incremental mode: filter to evidence beyond watermark minus overlap.
    if args.incremental:
        state = registry.get_source_state(stream_source_key)
        watermark = state.watermark if state else 0.0

        overlap_ts = 0.0
        if watermark > 0:
            watermark_idx = next(
                (i for i, e in enumerate(evidences) if get_evidence_timestamp(e) >= watermark),
                len(evidences),
            )
            overlap_start = max(0, watermark_idx - args.overlap)
            overlap_ts = (
                get_evidence_timestamp(evidences[overlap_start])
                if overlap_start < len(evidences)
                else 0.0
            )

        if overlap_ts > 0:
            original_count = len(evidences)
            evidences = [e for e in evidences if get_evidence_timestamp(e) >= overlap_ts]
            print(
                f'📊 Incremental mode: {original_count} total → {len(evidences)} after watermark (overlap={args.overlap})'
            )
        else:
            print(f'📊 Incremental mode: first run, processing all {len(evidences)} evidence')

    # Shard BEFORE offset/limit.
    if args.shards > 1:
        original_count = len(evidences)
        evidences = apply_shard(evidences, args.shards, args.shard_index)
        print(
            f'🧩 Shard {args.shard_index}/{args.shards}: {original_count} total → {len(evidences)}'
        )

    batch = evidences[args.offset : args.offset + args.limit]

    # Determine whether sub-chunking is active for this group.
    do_subchunk = args.group_id in _SUBCHUNK_GROUP_IDS
    subchunk_size = args.subchunk_size

    if args.dry_run:
        print(f'\n🔍 DRY RUN: Would process {len(batch)} evidence chunks')
        if do_subchunk:
            print(f'   Sub-chunking enabled (max {subchunk_size} chars per sub-chunk)')
        skipped = 0
        new = 0
        total_subchunks = 0
        for ev in batch:
            content = ev.get('content', '')

            chunk_key = ev.get('chunk_key') or ev.get('chunkKey')
            source_key = ev.get('source_key')
            chunk_idx = get_chunk_index(ev)

            if not (isinstance(chunk_key, str) and chunk_key):
                base = source_key or stream_source_key
                chunk_key = f'{base}:c{chunk_idx}'

            chunk_source_key = source_key or (
                chunk_key.split(':c', 1)[0] if ':c' in str(chunk_key) else stream_source_key
            )

            # Expand sub-chunks for counting
            if do_subchunk:
                sub_parts = subchunk_evidence(content, str(chunk_key), subchunk_size)
            else:
                sub_parts = [(str(chunk_key), strip_ingestion_noise(content))]

            for sub_key, sub_content in sub_parts:
                total_subchunks += 1
                sub_hash = registry.compute_content_hash(sub_content)
                if not args.force and registry.is_chunk_ingested(
                    chunk_source_key, sub_key, sub_hash
                ):
                    skipped += 1
                else:
                    new += 1
        print(f'   Total episodes (after sub-chunking): {total_subchunks}')
        print(f'   New: {new}, Already ingested: {skipped}')
        return

    client = MCPClient(args.mcp_url)

    ok = 0
    skipped = 0
    errors = 0
    max_ts = 0.0
    subchunk_count = 0  # tracks how many sub-chunks were created from oversized evidence

    for i, ev in enumerate(batch, start=1):
        evidence_id = ev.get('evidence_id') or ev.get('id') or ''
        content = ev.get('content', '')
        chunk_key = ev.get('chunk_key') or ev.get('chunkKey')
        source_key = ev.get('source_key')
        scope = ev.get('scope')

        chunk_idx = get_chunk_index(ev)

        # Ensure we have a stable chunk_key for dedupe/registry (v1) or legacy fallback.
        if not (isinstance(chunk_key, str) and chunk_key):
            base = source_key or stream_source_key
            chunk_key = f'{base}:c{chunk_idx}'

        chunk_source_key = source_key or (
            chunk_key.split(':c', 1)[0] if ':c' in str(chunk_key) else stream_source_key
        )

        # Sub-chunk large evidence for sessions_main to avoid LLM context overflow.
        # Each sub-chunk gets a deterministic :p0/:p1/... key suffix.
        if do_subchunk:
            sub_parts = subchunk_evidence(content, str(chunk_key), subchunk_size)
            if len(sub_parts) > 1:
                subchunk_count += len(sub_parts)
        else:
            sub_parts = [(str(chunk_key), strip_ingestion_noise(content))]

        for sub_key, sub_content in sub_parts:
            # Registry content hash (shortened) for dedup.
            content_hash = registry.compute_content_hash(sub_content)
            if not args.force and registry.is_chunk_ingested(
                chunk_source_key, sub_key, content_hash
            ):
                skipped += 1
                continue

            ep_name = f'{sub_key}:{evidence_id[:8] or content_hash[:8]}'
            src_desc = f'session chunk: {sub_key} (scope={scope or "unknown"})'

            chunk_uuid = registry.compute_chunk_uuid(
                source_key=chunk_source_key,
                chunk_key=sub_key,
                content_hash=content_hash,
            )
            episode_uuid = registry.compute_episode_uuid(chunk_uuid)

            body = sanitize_for_graphiti(sub_content)

            res = client.call_tool(
                'add_memory',
                {
                    'name': ep_name,
                    'episode_body': body,
                    'group_id': args.group_id,
                    'source': 'text',
                    'source_description': src_desc[:200],
                    # NOTE: do NOT pass "uuid" — standalone MCP breaks with client-generated UUIDs
                    # ("node X not found" error). Let the server generate its own.
                },
            )

            if 'error' in res:
                print(f'[{i}/{len(batch)}] ERROR enqueue {ep_name}: {res["error"]}')
                errors += 1
                continue

            ok += 1

            registry.record_chunk(
                chunk_uuid=chunk_uuid,
                source_key=chunk_source_key,
                chunk_key=sub_key,
                content_hash=content_hash,
                evidence_id=evidence_id or '(missing)',
            )
            registry.record_extraction_queued(
                group_id=args.group_id,
                episode_uuid=episode_uuid,
                chunk_uuid=chunk_uuid,
                source_key=chunk_source_key,
                chunk_key=sub_key,
            )

            time.sleep(args.sleep)

        ts = get_evidence_timestamp(ev)
        if ts > max_ts:
            max_ts = ts

        if i <= 5 or i == len(batch) or i % 100 == 0:
            print(
                f'[{i}/{len(batch)}] queued {chunk_key} ({len(sub_parts)} part{"s" if len(sub_parts) > 1 else ""})'
            )

    # Update watermark for this enqueue stream.
    if ok > 0 and max_ts > 0:
        registry.update_source_watermark(
            source_key=stream_source_key,
            source_type='session',
            watermark=max_ts,
            watermark_str=datetime.fromtimestamp(max_ts, tz=timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace('+00:00', 'Z'),
            overlap_window=args.overlap,
        )

    print(
        f'\nQueued: {ok} episodes into group_id={args.group_id} (from {len(batch)} evidence chunks)'
    )
    if subchunk_count:
        print(f'Sub-chunked: {subchunk_count} sub-chunks created from oversized evidence')
    if skipped:
        print(f'Skipped: {skipped} (already ingested)')
    if errors:
        print(f'Errors: {errors}')


if __name__ == '__main__':
    main()
