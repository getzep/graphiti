#!/usr/bin/env python3
"""Graphiti extraction progress monitor.

Outputs a Telegram-ready status message.

Design goals:
- Never mis-report completion as "in progress" due to transient FalkorDB timeouts.
- Finish quickly under load (uses small per-query timeouts).
- When queries time out, show "busy" plus the last known count from a local cache.

Exit code is always 0 (alerts are in the message body, not the exit code).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REDIS_CLI = "/opt/homebrew/opt/redis/bin/redis-cli"
PORT = 6379

# Keep these small: this script is run by cron with a short timeout.
# If FalkorDB is write-saturated, we *prefer* fast timeouts + cached last-known counts.
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


def _clawd_root() -> Path:
    # /Users/archibald/clawd/projects/graphiti-openclaw/scripts/extraction_monitor.py
    # parents[3] => /Users/archibald/clawd
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
        # Cache is best-effort.
        pass


def fmt_time(ts: str | None) -> str:
    if not ts:
        return "?"
    try:
        # ts stored as ISO-8601.
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%H:%M")
    except Exception:
        return "?"


def main() -> None:
    now_hhmm = datetime.now().strftime("%I:%M %p")
    lines: list[str] = [f"📊 Graphiti extraction — {now_hhmm}"]

    cache = load_cache()
    cache.setdefault("graphs", {})
    cache["last_run_at"] = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    # --- 1) FalkorDB health ---
    db_alive = False
    ping_err: str | None = None
    for attempt in (1, 2):
        try:
            r = subprocess.run(
                [REDIS_CLI, "-p", str(PORT), "PING"],
                capture_output=True,
                text=True,
                timeout=PING_TIMEOUT,
            )
            if r.stdout.strip() == "PONG":
                db_alive = True
                break
            ping_err = f"unexpected response: {r.stdout.strip()!r}"
        except subprocess.TimeoutExpired:
            ping_err = "PING timed out"
        except Exception as e:
            ping_err = f"error: {e}"

        # short retry
        if attempt == 1:
            continue

    if db_alive:
        lines.append("FalkorDB: ✅ alive")
        cache["last_ping_ok_at"] = cache["last_run_at"]
    else:
        last_ok = fmt_time(cache.get("last_ping_ok_at"))
        lines.append(
            f"FalkorDB: ⏳ busy/unresponsive ({ping_err or 'unknown'}) — last PONG at {last_ok}"
        )

    # --- 2) Episode counts ---
    lines.append("")
    lines.append("Episodes:")

    for graph, target in GRAPHS:
        try:
            r = subprocess.run(
                [
                    REDIS_CLI,
                    "-p",
                    str(PORT),
                    "GRAPH.QUERY",
                    graph,
                    "MATCH (e:Episodic) RETURN count(e)",
                ],
                capture_output=True,
                text=True,
                timeout=QUERY_TIMEOUT,
            )

            nums = [
                ln
                for ln in (r.stdout or "").strip().split("\n")
                if ln.strip().lstrip("-").isdigit()
            ]
            count = int(nums[0]) if nums else None

            if count is None:
                lines.append(f"  {graph}: ⚠️ parse error")
                continue

            cache["graphs"][graph] = {
                "count": count,
                "target": target,
                "at": cache["last_run_at"],
            }

            if count >= target:
                lines.append(f"  {graph}: ✅ {count}/{target}")
            else:
                pct = int(count / target * 100)
                lines.append(f"  {graph}: 🔄 {count}/{target} ({pct}%)")

        except subprocess.TimeoutExpired:
            # Do NOT treat this as "in progress".
            last = cache["graphs"].get(graph) or {}
            last_count = last.get("count")
            last_at = fmt_time(last.get("at"))
            if last_count is None:
                lines.append(f"  {graph}: ⏳ busy (no recent sample)")
            else:
                # If last sample already hit target, call it complete.
                if int(last_count) >= int(target):
                    lines.append(f"  {graph}: ✅ {last_count}/{target} (cached {last_at})")
                else:
                    pct = int(int(last_count) / target * 100)
                    lines.append(
                        f"  {graph}: ⏳ busy (cached {last_count}/{target} ({pct}%) @ {last_at})"
                    )

        except Exception:
            lines.append(f"  {graph}: ⚠️ error")

    save_cache(cache)

    # --- 3) MCP health ---
    lines.append("")
    mcp_status: list[str] = []
    for port in MCP_PORTS:
        try:
            r = subprocess.run(
                [
                    "curl",
                    "-s",
                    "-o",
                    "/dev/null",
                    "-w",
                    "%{http_code}",
                    "--max-time",
                    "0.8",
                    f"http://localhost:{port}/health",
                ],
                capture_output=True,
                text=True,
                timeout=1,
            )
            code = r.stdout.strip()
            mcp_status.append(f"{port}:{'✅' if code == '200' else '🔴'}")
        except Exception:
            mcp_status.append(f"{port}:🔴")

    lines.append("MCP: " + " ".join(mcp_status))

    # --- 4) Enqueue driver processes ---
    try:
        r1 = subprocess.run(
            ["pgrep", "-f", r"mcp_ingest_sessions\.py"],
            capture_output=True,
            text=True,
            timeout=1,
        )
        sessions_drivers = len(r1.stdout.strip().splitlines()) if r1.stdout.strip() else 0

        r2 = subprocess.run(
            ["pgrep", "-f", r"ingest_compound_notes\.py"],
            capture_output=True,
            text=True,
            timeout=1,
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
