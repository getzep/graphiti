#!/usr/bin/env python3
"""Delete misplaced episodic nodes across all FalkorDB graphs.

A node is misplaced if it lives in graph G but has group_id != G.
Always run --dry-run first to inspect what would be deleted.

Usage:
  python3 scripts/cleanup_misplacements_all_graphs.py --dry-run
  python3 scripts/cleanup_misplacements_all_graphs.py
  python3 scripts/cleanup_misplacements_all_graphs.py --only-prefix s1_content
"""

from __future__ import annotations

import argparse
import re
import subprocess

FALKORDB_CONTAINER = "graphiti-falkordb"
SUBPROCESS_TIMEOUT = 30
SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")


def _cypher(graph: str, query: str) -> str:
    if not SAFE_NAME_RE.match(graph):
        raise ValueError(f"unsafe graph name: {graph!r}")
    return subprocess.check_output(
        ["docker", "exec", FALKORDB_CONTAINER, "redis-cli", "-p", "6379",
         "GRAPH.QUERY", graph, query],
        text=True, timeout=SUBPROCESS_TIMEOUT,
    )


def _count(output: str) -> int:
    for line in output.splitlines()[1:]:
        if line.strip().isdigit():
            return int(line.strip())
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Delete misplaced episodes from all graphs")
    ap.add_argument("--dry-run", action="store_true", help="show what would be deleted")
    ap.add_argument("--only-prefix", default="", help="filter graphs by name prefix")
    args = ap.parse_args()

    raw = subprocess.check_output(
        ["docker", "exec", FALKORDB_CONTAINER, "redis-cli", "-p", "6379", "GRAPH.LIST"],
        text=True, timeout=SUBPROCESS_TIMEOUT)
    graphs = [x.strip() for x in raw.splitlines() if x.strip() and SAFE_NAME_RE.match(x.strip())]

    if args.only_prefix:
        if not SAFE_NAME_RE.match(args.only_prefix):
            raise ValueError(f"unsafe --only-prefix value: {args.only_prefix!r}")
        graphs = [g for g in graphs if g.startswith(args.only_prefix)]

    total = touched = 0
    for g in graphs:
        bad = _count(_cypher(
            g, f"MATCH (e:Episodic) WHERE e.group_id IS NOT NULL AND e.group_id <> '{g}' RETURN count(e)"))
        if bad <= 0:
            continue
        touched += 1
        total += bad
        label = "(would delete)" if args.dry_run else "(deleted)"
        print(f"{g}: {bad} misplaced {label}")
        if not args.dry_run:
            out = _cypher(
                g, f"MATCH (e:Episodic) WHERE e.group_id IS NOT NULL AND e.group_id <> '{g}' DETACH DELETE e")
            for line in out.splitlines():
                if "deleted" in line.lower():
                    print(f"  {line.strip()}")

    print(f"\ngraphs_touched={touched}  total_misplaced={total}"
          f"{'  (dry-run)' if args.dry_run else ''}")
    if not args.dry_run and total > 0:
        print("done — run scan_misplacements_all_graphs.py to verify")


if __name__ == "__main__":
    main()
