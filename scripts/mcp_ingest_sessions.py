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
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import urllib.error
import urllib.request

# shared sanitizer + registry (stdlib only)
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
from common import sanitize_for_graphiti
from registry import get_registry


MCP_URL_DEFAULT = "http://localhost:8000/mcp"
DEFAULT_OVERLAP_CHUNKS = 10  # for incremental mode


class MCPClient:
    """Minimal MCP (streamable HTTP) client for Graphiti (stdlib only)."""

    def __init__(self, url: str = MCP_URL_DEFAULT):
        self.url = url
        self.session_id: Optional[str] = None
        self.initialized = False

    def _decode(self, content_type: str, body: str, status: int) -> Dict[str, Any]:
        ct = (content_type or "").lower()

        if ct.startswith("text/event-stream"):
            data_lines = [l[len("data:") :].strip() for l in body.splitlines() if l.startswith("data:")]
            return json.loads(data_lines[-1]) if data_lines else {}

        return json.loads(body) if body.strip() else {}

    def _http_post_json(self, payload: Dict[str, Any], extra_headers: Optional[Dict[str, str]] = None):
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = getattr(resp, "status", 200)
                resp_headers = dict(resp.headers.items())
                body = resp.read().decode("utf-8", errors="replace")
                return status, resp_headers, body
        except urllib.error.HTTPError as e:
            status = e.code
            resp_headers = dict(e.headers.items()) if e.headers else {}
            body = e.read().decode("utf-8", errors="replace")
            return status, resp_headers, body

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        extra = {}
        if self.session_id:
            extra["mcp-session-id"] = self.session_id

        status, headers, body = self._http_post_json(payload, extra)

        sid = headers.get("mcp-session-id")
        if sid and not self.session_id:
            self.session_id = sid

        # Retry once if missing session id (Graphiti returns 400 but provides session id in headers)
        if status == 400 and ("Missing session ID" in body) and self.session_id and "mcp-session-id" not in extra:
            status, headers, body = self._http_post_json(payload, {"mcp-session-id": self.session_id})

        return self._decode(headers.get("content-type", ""), body, status)

    def initialize(self):
        if self.initialized:
            return

        self._post(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "graphiti-mcp-ingest-sessions", "version": "2"},
                },
            }
        )

        # notifications/initialized
        self._http_post_json(
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            {"mcp-session-id": self.session_id} if self.session_id else None,
        )

        self.initialized = True

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self.initialize()
        return self._post(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )


def get_evidence_timestamp(ev: dict) -> float:
    """Extract a best-effort start timestamp from evidence as float.

    Supports both v0 evidence (timestamp) and v1 evidence (timestamp_range.start).
    """

    ts = ev.get("timestamp")
    if not ts:
        ts = (ev.get("timestamp_range") or {}).get("start")
    if not ts:
        ts = (ev.get("timestampRange") or {}).get("start")

    if not ts:
        return 0.0

    if isinstance(ts, (int, float)):
        return float(ts)

    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return 0.0


def get_chunk_index(ev: dict) -> int:
    if "chunkIndex" in ev and ev.get("chunkIndex") is not None:
        try:
            return int(ev.get("chunkIndex") or 0)
        except Exception:
            return 0

    ck = ev.get("chunk_key") or ev.get("chunkKey")
    if isinstance(ck, str):
        m = re.search(r":c(\d+)$", ck)
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
        raise SystemExit(f"Invalid --shard-index {shard_index} for --shards {shards}")
    return [ev for i, ev in enumerate(evidences) if (i % shards) == shard_index]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcp-url", default=MCP_URL_DEFAULT)
    ap.add_argument(
        "--evidence",
        default=str(Path(__file__).resolve().parents[1] / "evidence" / "sessions_v1" / "main" / "all_evidence.json"),
        help="Evidence JSON file (v1 default: evidence/sessions_v1/<agent>/all_evidence.json)",
    )
    ap.add_argument("--group-id", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.02)

    # Sharding (parallel enqueue): take every Nth chunk.
    ap.add_argument("--shards", type=int, default=1, help="Total shard count (default: 1)")
    ap.add_argument("--shard-index", type=int, default=0, help="This worker's shard index [0..shards-1]")

    # Incremental mode options
    ap.add_argument(
        "--incremental",
        action="store_true",
        help="Enable incremental mode: only ingest new/changed chunks beyond a watermark",
    )
    ap.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP_CHUNKS,
        help=f"Number of evidence chunks to overlap for incremental mode (default: {DEFAULT_OVERLAP_CHUNKS})",
    )
    ap.add_argument("--dry-run", action="store_true", help="Preview what would be ingested without sending to Graphiti")
    ap.add_argument("--force", action="store_true", help="Force re-ingest even if chunk already in registry")
    args = ap.parse_args()

    evidence_path = Path(args.evidence)
    if not evidence_path.exists():
        raise SystemExit(
            f"Evidence file not found: {evidence_path}\n"
            "Run: python3 ingest/parse_sessions_v1.py --agent main  (preferred)\n"
            "  or: python3 ingest/parse_sessions.py --agent main      (legacy)"
        )

    evidences = json.loads(evidence_path.read_text(encoding="utf-8"))

    # Sort for consistent ordering.
    evidences.sort(
        key=lambda e: (
            get_evidence_timestamp(e),
            (e.get("source_key") or ""),
            get_chunk_index(e),
            (e.get("chunk_key") or e.get("chunkKey") or ""),
        )
    )

    registry = get_registry()

    # Registry source key for this *enqueue stream* (not per session).
    stream_source_key = f"sessions:{args.group_id}"

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
            overlap_ts = get_evidence_timestamp(evidences[overlap_start]) if overlap_start < len(evidences) else 0.0

        if overlap_ts > 0:
            original_count = len(evidences)
            evidences = [e for e in evidences if get_evidence_timestamp(e) >= overlap_ts]
            print(
                f"📊 Incremental mode: {original_count} total → {len(evidences)} after watermark (overlap={args.overlap})"
            )
        else:
            print(f"📊 Incremental mode: first run, processing all {len(evidences)} evidence")

    # Shard BEFORE offset/limit.
    if args.shards > 1:
        original_count = len(evidences)
        evidences = apply_shard(evidences, args.shards, args.shard_index)
        print(f"🧩 Shard {args.shard_index}/{args.shards}: {original_count} total → {len(evidences)}")

    batch = evidences[args.offset : args.offset + args.limit]

    if args.dry_run:
        print(f"\n🔍 DRY RUN: Would process {len(batch)} evidence chunks")
        skipped = 0
        new = 0
        for ev in batch:
            content = ev.get("content", "")
            content_hash = registry.compute_content_hash(content)

            chunk_key = ev.get("chunk_key") or ev.get("chunkKey")
            source_key = ev.get("source_key")
            chunk_idx = get_chunk_index(ev)

            if not (isinstance(chunk_key, str) and chunk_key):
                base = source_key or stream_source_key
                chunk_key = f"{base}:c{chunk_idx}"

            chunk_source_key = source_key or (
                chunk_key.split(":c", 1)[0] if ":c" in str(chunk_key) else stream_source_key
            )

            if not args.force and registry.is_chunk_ingested(chunk_source_key, str(chunk_key), content_hash):
                skipped += 1
            else:
                new += 1
        print(f"   New: {new}, Already ingested: {skipped}")
        return

    client = MCPClient(args.mcp_url)

    ok = 0
    skipped = 0
    max_ts = 0.0

    for i, ev in enumerate(batch, start=1):
        evidence_id = (ev.get("evidence_id") or ev.get("id") or "")
        content = ev.get("content", "")
        chunk_key = ev.get("chunk_key") or ev.get("chunkKey")
        source_key = ev.get("source_key")
        scope = ev.get("scope")

        chunk_idx = get_chunk_index(ev)

        # Ensure we have a stable chunk_key for dedupe/registry (v1) or legacy fallback.
        if not (isinstance(chunk_key, str) and chunk_key):
            base = source_key or stream_source_key
            chunk_key = f"{base}:c{chunk_idx}"

        chunk_source_key = source_key or (
            chunk_key.split(":c", 1)[0] if ":c" in str(chunk_key) else stream_source_key
        )

        # Registry content hash (shortened) for dedup.
        content_hash = registry.compute_content_hash(content)
        if not args.force and registry.is_chunk_ingested(chunk_source_key, str(chunk_key), content_hash):
            skipped += 1
            continue


        ep_name = f"{chunk_key}:{evidence_id[:8] or content_hash[:8]}"
        src_desc = f"session chunk: {chunk_key} (scope={scope or 'unknown'})"

        chunk_uuid = registry.compute_chunk_uuid(
            source_key=chunk_source_key,
            chunk_key=str(chunk_key),
            content_hash=content_hash,
        )
        episode_uuid = registry.compute_episode_uuid(chunk_uuid)

        body = sanitize_for_graphiti(content)

        res = client.call_tool(
            "add_memory",
            {
                "name": ep_name,
                "episode_body": body,
                "group_id": args.group_id,
                "source": "text",
                "source_description": src_desc[:200],
                # NOTE: do NOT pass "uuid" — standalone MCP breaks with client-generated UUIDs
                # ("node X not found" error). Let the server generate its own.
            },
        )

        if "error" in res:
            print(f"[{i}/{len(batch)}] ERROR enqueue {ep_name}: {res['error']}")
            continue

        ok += 1

        registry.record_chunk(
            chunk_uuid=chunk_uuid,
            source_key=chunk_source_key,
            chunk_key=str(chunk_key),
            content_hash=content_hash,
            evidence_id=evidence_id or "(missing)",
        )
        registry.record_extraction_queued(
            group_id=args.group_id,
            episode_uuid=episode_uuid,
            chunk_uuid=chunk_uuid,
            source_key=chunk_source_key,
            chunk_key=str(chunk_key),
        )

        ts = get_evidence_timestamp(ev)
        if ts > max_ts:
            max_ts = ts

        if i <= 5 or i == len(batch) or i % 100 == 0:
            print(f"[{i}/{len(batch)}] queued {ep_name}")

        time.sleep(args.sleep)

    # Update watermark for this enqueue stream.
    if ok > 0 and max_ts > 0:
        registry.update_source_watermark(
            source_key=stream_source_key,
            source_type="session",
            watermark=max_ts,
            watermark_str=datetime.fromtimestamp(max_ts, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            overlap_window=args.overlap,
        )

    print(f"\nQueued: {ok}/{len(batch)} episodes into group_id={args.group_id}")
    if skipped:
        print(f"Skipped: {skipped} (already ingested)")


if __name__ == "__main__":
    main()
