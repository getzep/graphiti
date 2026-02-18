#!/usr/bin/env python3
"""Strict sequential content-group ingestion into Graphiti MCP.

Processes one group at a time: enqueue → drain → assert → next.
Stops on excessive cross-contamination or extraction stall.

Usage:
  python3 scripts/ingest_content_groups.py --dry-run
  python3 scripts/ingest_content_groups.py
  python3 scripts/ingest_content_groups.py --groups s1_content_strategy
  python3 scripts/ingest_content_groups.py --force   # skip dedup
  python3 scripts/ingest_content_groups.py --evidence-dir /path/to/evidence
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MCP_URL = "http://localhost:8000/mcp"
FALKORDB_CONTAINER = "graphiti-falkordb"
MCP_CONTAINER = "graphiti-mcp"
SUBPROCESS_TIMEOUT = 30  # seconds for redis-cli calls

# Allowlist pattern for graph/group names (prevents Cypher injection)
SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")

CONTENT_GROUPS = [  # smallest first — fast feedback, catch problems early
    "s1_content_strategy",
    "s1_writing_samples",
    "s1_inspiration_short_form",
    "s1_inspiration_long_form",
]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_EVIDENCE_PATH = _REPO_ROOT.parent / "graphiti-openclaw-private" / "evidence" / "content_batch_v1"
# Allowed roots for evidence dir (env var or --evidence-dir)
_SAFE_EVIDENCE_ROOTS = [_REPO_ROOT.parent, Path.home()]

DEFAULT_EVIDENCE_DIR = Path(
    os.environ.get("GRAPHITI_EVIDENCE_DIR", str(_DEFAULT_EVIDENCE_PATH))
)

GROUP_EVIDENCE_FILES = {
    "s1_content_strategy": "content_strategy_v1.json",
    "s1_writing_samples": "writing_samples_v1.json",
    "s1_inspiration_short_form": "inspiration_short_form_v1.json",
    "s1_inspiration_long_form": "inspiration_long_form_v1.json",
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_name(name: str) -> str:
    """Validate a graph/group name against the allowlist pattern."""
    if not SAFE_NAME_RE.match(name):
        raise ValueError(f"unsafe graph/group name: {name!r}")
    return name


def _validate_evidence_dir(p: Path) -> Path:
    """Ensure evidence dir resolves under an allowed root."""
    resolved = p.resolve()
    for root in _SAFE_EVIDENCE_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue
    raise ValueError(f"evidence dir {p} resolves outside allowed roots: {resolved}")


# ---------------------------------------------------------------------------
# FalkorDB helpers
# ---------------------------------------------------------------------------

def _cypher(graph: str, query: str) -> str:
    """Run a Cypher query against a FalkorDB graph via redis-cli."""
    _validate_name(graph)
    return subprocess.check_output(
        ["docker", "exec", FALKORDB_CONTAINER, "redis-cli", "-p", "6379",
         "GRAPH.QUERY", graph, query],
        text=True, timeout=SUBPROCESS_TIMEOUT,
    )


def _count(output: str) -> int:
    """Parse a single-row integer result from GRAPH.QUERY output.

    Redis-cli output for `RETURN count(e)` looks like:
        count(e)
        42
        ...stats lines...
    We grab the first purely-numeric non-header line.
    """
    lines = output.splitlines()
    for line in lines[1:]:  # skip header row
        s = line.strip()
        if s.isdigit():
            return int(s)
    return 0


def correct_chunk_keys(group_id: str) -> set[str]:
    """Chunk keys present in the right graph with matching group_id."""
    gid = _validate_name(group_id)
    out = _cypher(gid, f"MATCH (e:Episodic) WHERE e.group_id = '{gid}' RETURN e.source_description")
    keys: set[str] = set()
    for line in out.splitlines():
        m = re.search(r"session chunk: (.+?) \(scope=", line)
        if m:
            keys.add(m.group(1).strip())
    return keys


def misplaced_total(groups: list[str]) -> int:
    """Count episodes sitting in the wrong graph across given groups."""
    total = 0
    for g in groups:
        gid = _validate_name(g)
        total += _count(_cypher(
            gid, f"MATCH (e:Episodic) WHERE e.group_id IS NOT NULL AND e.group_id <> '{gid}' RETURN count(e)",
        ))
    return total


# ---------------------------------------------------------------------------
# MCP client (JSON-RPC over HTTP, streamable transport)
# ---------------------------------------------------------------------------

class MCPClient:
    def __init__(self, url: str = MCP_URL):
        self.url = url
        self.sid: str | None = None
        self._req_id = 0

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"}
        if self.sid:
            headers["Mcp-Session-Id"] = self.sid
        req = urllib.request.Request(
            self.url, json.dumps(payload).encode(), headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            self.sid = resp.headers.get("Mcp-Session-Id", self.sid)
            body, ct = resp.read().decode(), resp.headers.get("Content-Type", "")
        if "text/event-stream" in ct:
            data_lines = [line[5:].strip() for line in body.splitlines() if line.startswith("data:")]
            if not data_lines:
                return {}
            parsed = json.loads(data_lines[-1])
        else:
            parsed = json.loads(body) if body.strip() else {}
        if not parsed:
            raise RuntimeError("empty response from MCP server")
        return parsed

    def init(self) -> None:
        self._post({"jsonrpc": "2.0", "id": self._next_id(), "method": "initialize",
                     "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                "clientInfo": {"name": "ingest-content-groups", "version": "3"}}})
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    def add_memory(self, *, name: str, body: str, group_id: str, source_description: str) -> dict:
        return self._post({"jsonrpc": "2.0", "id": self._next_id(), "method": "tools/call",
                           "params": {"name": "add_memory", "arguments": {
                               "name": name, "episode_body": body, "group_id": group_id,
                               "source": "text", "source_description": source_description}}})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[ingest] {ts} {msg}", flush=True)


def sanitize(text: str) -> str:
    """Strip chars that break FalkorDB Cypher parsing."""
    text = re.sub(r'[{}()\[\]|<>@#$%^&*~`"\'\\]', " ", text)
    return re.sub(r"\s+", " ", text).strip()[:8000]


def last_openai_ts() -> float:
    """Epoch of most recent OpenAI call in MCP container logs (assumes UTC)."""
    try:
        logs = subprocess.check_output(
            ["docker", "logs", "--tail", "200", MCP_CONTAINER],
            text=True, stderr=subprocess.STDOUT, timeout=10)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return 0.0
    lines = [l for l in logs.splitlines() if "api.openai.com" in l]
    if not lines:
        return 0.0
    m = re.match(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})", lines[-1])
    if not m:
        return 0.0
    return datetime.fromisoformat(f"{m.group(1)} {m.group(2)}").replace(
        tzinfo=timezone.utc).timestamp()


def load_evidence(group_id: str, evidence_dir: Path) -> dict[str, dict]:
    if group_id not in GROUP_EVIDENCE_FILES:
        log(f"  ⚠ unknown group {group_id!r} — known: {list(GROUP_EVIDENCE_FILES)}")
        return {}
    path = evidence_dir / GROUP_EVIDENCE_FILES[group_id]
    if not path.exists():
        log(f"  ⚠ evidence file not found: {path}")
        return {}
    return {e["chunk_key"]: e for e in json.loads(path.read_text()) if e.get("chunk_key")}


# ---------------------------------------------------------------------------
# Drain waiter
# ---------------------------------------------------------------------------

def wait_for_drain(
    group_id: str, expected: int, *, poll_s: int, stable_checks: int, max_wait_s: int,
) -> bool:
    """Block until correct episode count reaches expected, or stall detected."""
    log(f"  draining {group_id}: target={expected}")
    start = time.time()
    prev, stable = -1, 0

    while True:
        current = len(correct_chunk_keys(group_id))
        elapsed = int(time.time() - start)
        log(f"    {current}/{expected} ({elapsed}s)")

        if current >= expected:
            log("  ✅ drain complete")
            return True
        if time.time() - start > max_wait_s:
            log(f"  ⏰ timeout after {max_wait_s}s")
            return False

        stable = stable + 1 if current == prev else 0
        prev = current

        if stable >= stable_checks:
            quiet = time.time() - (last_openai_ts() or 0)
            if quiet > 180:
                log(f"  stalled: no progress + OpenAI quiet {int(quiet)}s")
                return False

        time.sleep(poll_s)


# ---------------------------------------------------------------------------
# Group processor
# ---------------------------------------------------------------------------

def process_group(
    group_id: str, mcp: MCPClient, evidence_dir: Path, *,
    force: bool, sleep_s: float, poll_s: int, stable_checks: int,
    max_wait_s: int, max_misplace_growth: int,
) -> dict:
    """Enqueue missing chunks for one group, wait for drain, assert placement."""
    evidence = load_evidence(group_id, evidence_dir)
    if not evidence:
        log(f"\n== {group_id} == no evidence, skipping")
        return {"group": group_id, "queued": 0, "errors": 0, "drained": True}

    present = correct_chunk_keys(group_id)
    to_queue = sorted(evidence.keys()) if force else sorted(set(evidence.keys()) - present)

    log(f"\n== {group_id} ==")
    log(f"  expected={len(evidence)} correct={len(present)} to_queue={len(to_queue)}")

    if not to_queue:
        log("  nothing to queue ✓")
        return {"group": group_id, "queued": 0, "errors": 0, "drained": True}

    before = misplaced_total(CONTENT_GROUPS)
    log(f"  misplaced_before={before}")

    ok = err = 0
    for i, ck in enumerate(to_queue, 1):
        ev = evidence[ck]
        name = f"{ck}:{str(ev.get('id', ''))[:8]}"
        body = sanitize(str(ev.get("content", "") or ""))
        src = f"session chunk: {ck} (scope={ev.get('scope', 'private')})"[:200]
        try:
            res = mcp.add_memory(name=name, body=body, group_id=group_id, source_description=src)
            if "error" in res:
                err += 1
                if err <= 3:
                    log(f"  ⚠ {ck}: {json.dumps(res.get('error', ''))[:200]}")
            else:
                ok += 1
        except Exception as ex:
            err += 1
            if err <= 3:
                log(f"  ⚠ {ck}: {ex}")
        if i <= 3 or i == len(to_queue) or i % 25 == 0:
            log(f"  [{i}/{len(to_queue)}] {ck}")
        time.sleep(sleep_s)

    log(f"  enqueue done: ok={ok} err={err}")

    growth = misplaced_total(CONTENT_GROUPS) - before
    log(f"  misplacement_growth={growth}")
    if growth > max_misplace_growth:
        log(f"  ❌ growth {growth} > threshold {max_misplace_growth}, stopping")
        return {"group": group_id, "queued": ok, "errors": err, "drained": False}

    drained = wait_for_drain(
        group_id, len(evidence),
        poll_s=poll_s, stable_checks=stable_checks, max_wait_s=max_wait_s)
    return {"group": group_id, "queued": ok, "errors": err, "drained": drained}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Strict sequential content-group ingestion")
    ap.add_argument("--groups", default=",".join(CONTENT_GROUPS))
    ap.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between enqueue calls")
    ap.add_argument("--poll", type=int, default=30, help="drain poll interval (s)")
    ap.add_argument("--stable-checks", type=int, default=8)
    ap.add_argument("--max-wait", type=int, default=5400, help="max drain wait per group (s)")
    ap.add_argument("--max-misplace-growth", type=int, default=5)
    ap.add_argument("--force", action="store_true", help="re-ingest all (skip dedup)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    # Validate all group names upfront
    for g in groups:
        _validate_name(g)

    # Validate evidence dir resolves under allowed roots
    args.evidence_dir = _validate_evidence_dir(args.evidence_dir)

    log(f"groups={groups} force={args.force} dry_run={args.dry_run}")

    if args.dry_run:
        for gid in groups:
            ev = load_evidence(gid, args.evidence_dir)
            present = correct_chunk_keys(gid)
            missing = sorted(ev.keys()) if args.force else sorted(set(ev.keys()) - present)
            log(f"\n{gid}: expected={len(ev)} correct={len(present)} to_queue={len(missing)}")
            for k in missing[:5]:
                log(f"  {k}")
            if len(missing) > 5:
                log(f"  ... +{len(missing) - 5} more")
        return

    mcp = MCPClient()
    mcp.init()
    log(f"MCP session={mcp.sid}")

    results: list[dict] = []
    for gid in groups:
        r = process_group(
            gid, mcp, args.evidence_dir,
            force=args.force, sleep_s=args.sleep, poll_s=args.poll,
            stable_checks=args.stable_checks, max_wait_s=args.max_wait,
            max_misplace_growth=args.max_misplace_growth)
        results.append(r)
        if r["queued"] > 0 and not r["drained"]:
            log("  stopping before next group")
            break

    log("\n== Summary ==")
    for r in results:
        icon = "✅" if r["drained"] else "⏳"
        log(f"  {icon} {r['group']}: queued={r['queued']} errors={r['errors']}")

    if any(not r["drained"] for r in results if r["queued"] > 0):
        log("⚠ re-run to continue remaining groups")
        sys.exit(1)
    log("✅ all groups complete")


if __name__ == "__main__":
    main()
