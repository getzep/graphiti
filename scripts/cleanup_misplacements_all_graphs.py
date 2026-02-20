#!/usr/bin/env python3
"""Delete misplaced episodic nodes across all graphs.

A node is misplaced if it lives in graph G but has group_id != G.
Always run --dry-run first to inspect what would be deleted.

On Neo4j (single DB) misplacements cannot occur — the script reports zero and exits.

Supports both Neo4j and FalkorDB via --backend flag.

Usage:
  python3 scripts/cleanup_misplacements_all_graphs.py --dry-run
  python3 scripts/cleanup_misplacements_all_graphs.py
  python3 scripts/cleanup_misplacements_all_graphs.py --only-prefix s1_content
  python3 scripts/cleanup_misplacements_all_graphs.py --backend falkordb --dry-run
"""

from __future__ import annotations

import argparse

from graph_cli import SAFE_NAME_RE, list_graphs, parse_count, run_cypher


def main() -> None:
    ap = argparse.ArgumentParser(description="Delete misplaced episodes from all graphs")
    ap.add_argument("--backend", choices=["neo4j", "falkordb"], default="neo4j",
                    help="graph database backend (default: neo4j)")
    ap.add_argument("--dry-run", action="store_true", help="show what would be deleted")
    ap.add_argument("--only-prefix", default="", help="filter graphs by name prefix")
    args = ap.parse_args()

    backend = args.backend

    if backend == "neo4j":
        # Neo4j uses a single database — nodes are scoped by group_id, so
        # cross-graph misplacement cannot occur.  Nothing to clean up.
        print("backend=neo4j — single DB, misplacement N/A")
        print("\ngraphs_touched=0  total_misplaced=0")
        return

    graphs = list_graphs(backend)

    if args.only_prefix:
        if not SAFE_NAME_RE.match(args.only_prefix):
            raise ValueError(f"unsafe --only-prefix value: {args.only_prefix!r}")
        graphs = [g for g in graphs if g.startswith(args.only_prefix)]

    total = touched = 0
    for g in graphs:
        bad = parse_count(run_cypher(
            backend, g,
            f"MATCH (e:Episodic) WHERE e.group_id IS NOT NULL AND e.group_id <> '{g}' RETURN count(e)"))
        if bad is None or bad <= 0:
            continue
        touched += 1
        total += bad
        label = "(would delete)" if args.dry_run else "(deleted)"
        print(f"{g}: {bad} misplaced {label}")
        if not args.dry_run:
            out = run_cypher(
                backend, g,
                f"MATCH (e:Episodic) WHERE e.group_id IS NOT NULL AND e.group_id <> '{g}' DETACH DELETE e")
            for line in out.splitlines():
                if "deleted" in line.lower():
                    print(f"  {line.strip()}")

    print(f"\ngraphs_touched={touched}  total_misplaced={total}"
          f"{'  (dry-run)' if args.dry_run else ''}")
    if not args.dry_run and total > 0:
        print("done — run scan_misplacements_all_graphs.py to verify")


if __name__ == "__main__":
    main()
