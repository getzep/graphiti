#!/usr/bin/env python3
"""
Graphiti extraction progress monitor.
Outputs a Telegram-ready status message. No interpretation needed.
Exit 0 always (alerts are in the message body, not exit code).
"""
import subprocess
import sys
from datetime import datetime

REDIS_CLI = "/opt/homebrew/opt/redis/bin/redis-cli"
PORT = 6379
QUERY_TIMEOUT = 8  # seconds per graph query

GRAPHS = [
    ("s1_sessions_main",          3646),
    ("s1_inspiration_long_form",  494),
    ("s1_inspiration_short_form", 190),
    ("s1_writing_samples",        83),
    ("s1_content_strategy",       10),
    ("engineering_learnings",     51),
    ("learning_self_audit",       19),
]

MCP_PORTS = [8000, 8001, 8002, 8003, 8004, 8005, 8006]

now = datetime.now().strftime("%I:%M %p")
lines = [f"📊 Graphiti extraction — {now}"]

# --- 1. FalkorDB health ---
try:
    r = subprocess.run(
        [REDIS_CLI, "-p", str(PORT), "PING"],
        capture_output=True, text=True, timeout=4
    )
    if r.stdout.strip() == "PONG":
        lines.append("FalkorDB: ✅ alive")
        db_alive = True
    else:
        lines.append(f"FalkorDB: 🔴 unexpected response: {r.stdout.strip()!r}")
        db_alive = False
except subprocess.TimeoutExpired:
    lines.append("FalkorDB: 🚨 PING timed out — service may be down")
    print("\n".join(lines))
    sys.exit(0)
except Exception as e:
    lines.append(f"FalkorDB: 🚨 error: {e}")
    print("\n".join(lines))
    sys.exit(0)

# --- 2. Episode counts ---
lines.append("")
lines.append("Episodes:")
for graph, target in GRAPHS:
    try:
        r = subprocess.run(
            [REDIS_CLI, "-p", str(PORT), "GRAPH.QUERY", graph,
             "MATCH (e:Episodic) RETURN count(e)"],
            capture_output=True, text=True, timeout=QUERY_TIMEOUT
        )
        nums = [l for l in r.stdout.strip().split("\n")
                if l.strip().lstrip("-").isdigit()]
        count = int(nums[0]) if nums else None
        if count is None:
            lines.append(f"  {graph}: ⚠️ parse error")
        elif count >= target:
            lines.append(f"  {graph}: ✅ {count}/{target}")
        else:
            pct = int(count / target * 100)
            lines.append(f"  {graph}: 🔄 {count}/{target} ({pct}%)")
    except subprocess.TimeoutExpired:
        lines.append(f"  {graph}: ⏳ writes in progress")

# --- 3. MCP health ---
lines.append("")
mcp_status = []
for port in MCP_PORTS:
    try:
        r = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
             "--max-time", "3", f"http://localhost:{port}/health"],
            capture_output=True, text=True, timeout=5
        )
        code = r.stdout.strip()
        mcp_status.append(f"{port}:{'✅' if code == '200' else '🔴'}")
    except Exception:
        mcp_status.append(f"{port}:🔴")
lines.append("MCP: " + " ".join(mcp_status))

# --- 4. Workers ---
try:
    r = subprocess.run(
        ["pgrep", "-f", "mcp_ingest_sessions"],
        capture_output=True, text=True
    )
    worker_count = len(r.stdout.strip().splitlines()) if r.stdout.strip() else 0
    lines.append(f"Workers: {worker_count} ingest processes running" if worker_count else "Workers: none (MCP draining async)")
except Exception:
    lines.append("Workers: unknown")

print("\n".join(lines))
