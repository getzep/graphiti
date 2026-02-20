#!/usr/bin/env python3
"""Graphiti extraction progress monitor.

Outputs a Telegram-ready status message.

Design goals:
- Never mis-report completion as "in progress" due to transient DB timeouts.
- Finish quickly under load (uses small per-query timeouts).
- When queries time out, show "busy" plus the last known count from a local cache.

Supports both Neo4j and FalkorDB via --backend flag.

Exit code is always 0 (alerts are in the message body, not the exit code).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from graph_cli import check_health, parse_count, run_cypher

# Keep these small: this script is run by cron with a short timeout.
# If the DB is write-saturated, we *prefer* fast timeouts + cached last-known counts.
PING_TIMEOUT = 2  # seconds
QUERY_TIMEOUT = 4  # seconds per graph count query

GRAPHS = [
    ("s1_sessions_main", 3646),
    ("s1_inspiration_long_form", 494),
    ("s1_inspiration_short_form", 190),
    ("s1_writing_samples", 83),
    ("s1_content_strategy", 10),
    ("engineering_learnings", 51),
    ("learning_self_audit", 19),
]

MCP_PORTS = [8000, 8001, 8002, 8003, 8004, 8005, 8006]

BACKEND_LABELS = {"neo4j": "Neo4j", "falkordb": "FalkorDB"}


def _clawd_root() -> Path:
    return Path(__file__).resolve().parents[3]


CACHE_PATH = (
    _clawd_root()
    / "projects"
    / "graphiti-openclaw-runtime"
    / "state"
    / "extraction_monitor_cache.json"
)


def load_cache() -> dict:
    try:
        if CACHE_PATH.exists():
            return json.loads(CACHE_PATH.read_text())
    except Exception:
        pass
    return {}


def save_cache(cache: dict) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=2, sort_keys=True))
        tmp.replace(CACHE_PATH)
    except Exception:
        pass


def fmt_time(ts: str | None) -> str:
    if not ts:
        return "?"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%H:%M")
    except Exception:
        return "?"


def main() -> None:
    ap = argparse.ArgumentParser(description="Graphiti extraction progress monitor")
    ap.add_argument("--backend", choices=["neo4j", "falkordb"], default="neo4j",
                    help="graph database backend (default: neo4j)")
    args = ap.parse_args()

    backend = args.backend
    db_label = BACKEND_LABELS[backend]

    now_hhmm = datetime.now().strftime("%I:%M %p")
    lines: list[str] = [f"\U0001f4ca Graphiti extraction \u2014 {now_hhmm}"]

    cache = load_cache()
    cache.setdefault("graphs", {})
    cache["last_run_at"] = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    # --- 1) DB health ---
    db_alive = False
    for _attempt in (1, 2):
        if check_health(backend, timeout=PING_TIMEOUT):
            db_alive = True
            break

    if db_alive:
        lines.append(f"{db_label}: \u2705 alive")
        cache["last_ping_ok_at"] = cache["last_run_at"]
    else:
        last_ok = fmt_time(cache.get("last_ping_ok_at"))
        lines.append(
            f"{db_label}: \u23f3 busy/unresponsive \u2014 last OK at {last_ok}"
        )

    # --- 2) Episode counts ---
    lines.append("")
    lines.append("Episodes:")

    for graph, target in GRAPHS:
        try:
            # For Neo4j (single DB): scope by group_id.
            # For FalkorDB: query runs against the named graph.
            if backend == "neo4j":
                q = f"MATCH (e:Episodic) WHERE e.group_id = '{graph}' RETURN count(e);"
            else:
                q = "MATCH (e:Episodic) RETURN count(e)"

            out = run_cypher(backend, graph, q, timeout=QUERY_TIMEOUT)
            count: int | None = parse_count(out) if out.strip() else None

            if count is None:
                lines.append(f"  {graph}: \u26a0\ufe0f parse error")
                continue

            cache["graphs"][graph] = {
                "count": count,
                "target": target,
                "at": cache["last_run_at"],
            }

            if count >= target:
                lines.append(f"  {graph}: \u2705 {count}/{target}")
            else:
                pct = int(count / target * 100)
                lines.append(f"  {graph}: \U0001f504 {count}/{target} ({pct}%)")

        except subprocess.TimeoutExpired:
            last = cache["graphs"].get(graph) or {}
            last_count = last.get("count")
            last_at = fmt_time(last.get("at"))
            if last_count is None:
                lines.append(f"  {graph}: \u23f3 busy (no recent sample)")
            elif int(last_count) >= int(target):
                lines.append(f"  {graph}: \u2705 {last_count}/{target} (cached {last_at})")
            else:
                pct = int(int(last_count) / target * 100)
                lines.append(
                    f"  {graph}: \u23f3 busy (cached {last_count}/{target} ({pct}%) @ {last_at})"
                )

        except Exception:
            lines.append(f"  {graph}: \u26a0\ufe0f error")

    save_cache(cache)

    # --- 3) MCP health ---
    lines.append("")
    mcp_status: list[str] = []
    for port in MCP_PORTS:
        try:
            r = subprocess.run(
                [
                    "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                    "--max-time", "0.8", f"http://localhost:{port}/health",
                ],
                capture_output=True, text=True, timeout=1,
            )
            code = r.stdout.strip()
            icon = '\u2705' if code == '200' else '\U0001f534'
            mcp_status.append(f"{port}:{icon}")
        except Exception:
            mcp_status.append(f"{port}:\U0001f534")

    lines.append("MCP: " + " ".join(mcp_status))

    # --- 4) Enqueue driver processes ---
    try:
        r1 = subprocess.run(
            ["pgrep", "-f", r"mcp_ingest_sessions\.py"],
            capture_output=True, text=True, timeout=1,
        )
        sessions_drivers = len(r1.stdout.strip().splitlines()) if r1.stdout.strip() else 0

        r2 = subprocess.run(
            ["pgrep", "-f", r"ingest_compound_notes\.py"],
            capture_output=True, text=True, timeout=1,
        )
        compound_drivers = len(r2.stdout.strip().splitlines()) if r2.stdout.strip() else 0

        if sessions_drivers or compound_drivers:
            lines.append(f"Drivers: sessions={sessions_drivers} compound={compound_drivers}")
        else:
            lines.append("Drivers: none (MCP draining async)")
    except Exception:
        lines.append("Drivers: unknown")

    print("\n".join(lines))
    sys.exit(0)


if __name__ == "__main__":
    main()
