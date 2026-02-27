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
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.common import sanitize_for_graphiti
from ingest.registry import get_registry

MCP_URL_DEFAULT = 'http://localhost:8000/mcp'
DEFAULT_OVERLAP_CHUNKS = 10  # for incremental mode

# --- Sub-chunking for large evidence (sessions_main) ---
# Default max chars per episode body sent to Graphiti. Evidence chunks exceeding
# this limit are deterministically split into sub-chunks with :p0/:p1/... suffixes.
# This avoids LLM context_length_exceeded errors without lossy truncation.
DEFAULT_SUBCHUNK_SIZE = 10_000

# Group IDs that get automatic sub-chunking when evidence exceeds subchunk_size.
_SUBCHUNK_GROUP_IDS = {'s1_sessions_main'}


def strip_untrusted_metadata(content: str) -> str:
    """Remove untrusted metadata blocks while preventing ReDoS and early termination.
    
    Parses line-by-line to correctly pair backtick fences and avoid early termination
    if the JSON payload itself contains embedded triple backticks inside strings.
    Collapses resulting multiple empty lines to maintain paragraph boundaries.
    """
    prefixes = (
        "Conversation info:",
        "Sender (untrusted metadata):",
        "Replied message (untrusted, for context):",
        "Conversation info (untrusted metadata):",
    )
    
    lines = content.splitlines(keepends=True)
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped_line = line.strip()
        
        # Fast path prefix check
        matched = False
        for p in prefixes:
            if stripped_line == p:
                matched = True
                break
                
        if matched and i + 1 < n and lines[i+1].strip() == "```json":
            # We found a header followed by ```json
            # Find the true closing ``` which is unindented and on its own line.
            # Real JSON doesn't contain unescaped newlines, so any "```" on its own line is the end.
            k = i + 2
            found_end = False
            while k < n:
                if lines[k].strip() == "```":
                    found_end = True
                    break
                k += 1
                
            if found_end:
                # Skip all lines from i to k
                i = k + 1
                continue
                
        out.append(line)
        i += 1
        
    result = "".join(out)
    # Collapse 3+ newlines into 2 to prevent empty paragraph chunks
    return re.sub(r'\n{3,}', '\n\n', result).strip()


def subchunk_evidence(content: str, chunk_key: str, max_chars: int) -> list[tuple[str, str]]:
    """Split content into deterministic sub-chunks if it exceeds max_chars.

    Returns a list of (sub_chunk_key, sub_content) tuples.
    If content fits in a single chunk, returns [(chunk_key, content)].

    Sub-chunk keys use :p0, :p1, ... suffixes for deterministic idempotent keys.
    Strips enormous untrusted metadata JSON payloads before sub-chunking, then
    splits on paragraph boundaries (double newline) when possible to keep context
    coherent. Falls back to hard split at max_chars if no good boundary exists.

    Raises:
        ValueError: If max_chars <= 0 (would cause infinite loop).
    """
    if max_chars <= 0:
        raise ValueError(f'max_chars must be positive, got {max_chars}')

    # Strip enormous metadata injections before we count length
    content = strip_untrusted_metadata(content)

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


class MCPClient:
    """Minimal MCP (streamable HTTP) client for Graphiti (stdlib only)."""

    def __init__(self, url: str = MCP_URL_DEFAULT):
        self.url = url
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


def check_bootstrap_guard(neo4j_message_count: int, evidence_files_exist: bool) -> bool:
    """Return True if BOOTSTRAP_REQUIRED guard should fire.

    The guard fires when Neo4j has zero messages but evidence files exist,
    indicating the graph has not been bootstrapped yet.
    """
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
    cursor = conn.execute(
        "UPDATE chunk_claims SET status='claimed', worker_id=?, claimed_at=? "
        "WHERE chunk_id = (SELECT chunk_id FROM chunk_claims WHERE status='pending' LIMIT 1) "
        'RETURNING chunk_id',
        (worker_id, datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')),
    )
    row = cursor.fetchone()
    conn.commit()
    return row[0] if row else None


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


def main():
    ap = build_parser()
    args = ap.parse_args()

    if args.subchunk_size <= 0:
        ap.error('--subchunk-size must be a positive integer')

    # --- FR-10: claim-state-check mode ---
    if args.claim_state_check:
        if not args.manifest and not args.build_manifest:
            ap.error('--claim-state-check requires --manifest or --build-manifest')
        claim_db_path = (args.manifest or args.build_manifest) + '.claims.db'
        if not Path(claim_db_path).exists():
            print(f'Claim DB not found: {claim_db_path}')
            return
        conn = init_claim_db(claim_db_path)
        cursor = conn.execute(
            'SELECT status, COUNT(*) FROM chunk_claims GROUP BY status'
        )
        rows = cursor.fetchall()
        print('Claim state summary:')
        for status, count in rows:
            print(f'  {status}: {count}')
        conn.close()
        return

    # --- FR-4: source mode routing ---
    if args.source_mode == 'neo4j':
        _run_neo4j_mode(args, ap)
    else:
        _run_evidence_mode(args, ap)


def _run_neo4j_mode(args: argparse.Namespace, ap: argparse.ArgumentParser) -> None:
    """Neo4j source mode: read candidate chunks from the graph DB."""
    evidence_path = Path(args.evidence)
    evidence_files_exist = evidence_path.exists()

    # BOOTSTRAP_REQUIRED guard: if Neo4j has 0 messages but evidence files exist,
    # the graph hasn't been bootstrapped yet.
    # In a real implementation, neo4j_message_count would query the graph.
    # For now, we assume 0 when neo4j is not reachable.
    neo4j_message_count = 0
    if check_bootstrap_guard(neo4j_message_count, evidence_files_exist):
        print(
            'BOOTSTRAP_REQUIRED: Neo4j has no messages but evidence files exist.\n'
            'Run with --source-mode evidence first to bootstrap the graph,\n'
            'or populate Neo4j before using neo4j source mode.'
        )
        return

    # --- build-manifest mode ---
    if args.build_manifest:
        manifest_path = Path(args.build_manifest)
        # Placeholder: in a full implementation, chunk IDs would come from Neo4j query.
        chunk_ids: list[str] = []
        print(
            f'Neo4j source mode: would query graph for candidate chunks '
            f'(group_id={args.group_id})'
        )
        if args.dry_run:
            print(f'DRY RUN: would write {len(chunk_ids)} chunk IDs to {manifest_path}')
            return
        manifest_path.write_text(json.dumps(chunk_ids), encoding='utf-8')
        print(f'Wrote manifest with {len(chunk_ids)} chunk IDs to {manifest_path}')

        if args.claim_mode:
            claim_db_path = str(manifest_path) + '.claims.db'
            conn = init_claim_db(claim_db_path)
            seed_claims(conn, chunk_ids)
            print(f'Seeded claim DB at {claim_db_path} with {len(chunk_ids)} chunks')
            conn.close()
        return

    # --- claim mode with existing manifest ---
    if args.claim_mode:
        if not args.manifest:
            ap.error('--claim-mode requires --manifest (or --build-manifest to create one)')
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            raise SystemExit(f'Manifest not found: {manifest_path}')
        claim_db_path = str(manifest_path) + '.claims.db'
        conn = init_claim_db(claim_db_path)

        worker_id = f'w{args.shard_index}'
        claimed = 0
        while True:
            chunk_id = claim_chunk(conn, worker_id)
            if chunk_id is None:
                break
            claimed += 1
            if args.dry_run:
                print(f'DRY RUN: would process chunk {chunk_id}')
            else:
                print(f'Processing claimed chunk: {chunk_id}')
                # Placeholder: actual processing would happen here
            conn.execute(
                "UPDATE chunk_claims SET status='completed', completed_at=? "
                'WHERE chunk_id=?',
                (
                    datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                    chunk_id,
                ),
            )
            conn.commit()

        print(f'Worker {worker_id} processed {claimed} chunks')
        conn.close()
        return

    # Default neo4j mode: placeholder
    print(
        f'Neo4j source mode (group_id={args.group_id}): '
        'would read candidate chunks from the graph.\n'
        'Use --dry-run to preview candidate chunks.\n'
        'Use --build-manifest <path> to freeze chunk IDs to a manifest file.'
    )


def _run_evidence_mode(args: argparse.Namespace, ap: argparse.ArgumentParser) -> None:
    """Evidence source mode: the original JSON-file-based ingestion path."""
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
        help=f'Number of evidence chunks to overlap for incremental mode (default: {DEFAULT_OVERLAP_CHUNKS})',
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
        help=f'Max chars per sub-chunk for large evidence (default: {DEFAULT_SUBCHUNK_SIZE}). '
        'Only applies to group IDs in _SUBCHUNK_GROUP_IDS (e.g. s1_sessions_main).',
    )
    args = ap.parse_args()

    if args.subchunk_size <= 0:
        ap.error('--subchunk-size must be a positive integer')

    evidence_path = Path(args.evidence)
    if not evidence_path.exists():
        raise SystemExit(
            f'Evidence file not found: {evidence_path}\n'
            'Run: python3 ingest/parse_sessions_v1.py --agent main  (preferred)\n'
            '  or: python3 ingest/parse_sessions.py --agent main      (legacy)'
        )

    evidences = json.loads(evidence_path.read_text(encoding='utf-8'))

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
                sub_parts = [(str(chunk_key), content)]

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
            sub_parts = [(str(chunk_key), content)]

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
