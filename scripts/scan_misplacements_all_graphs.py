#!/usr/bin/env python3
"""Scan all FalkorDB graphs for misplaced episodic nodes.

A node is misplaced if it lives in graph G but has group_id != G.
Outputs a JSON report and prints a summary table.

Usage:
  python3 scripts/scan_misplacements_all_graphs.py
  python3 scripts/scan_misplacements_all_graphs.py -o /tmp/report.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

FALKORDB_CONTAINER = "graphiti-falkordb"
SUBPROCESS_TIMEOUT = 30
SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")
SAFE_OUTPUT_ROOTS = [Path.cwd(), Path("_tmp").resolve(), Path("/tmp")]
DEFAULT_OUT = Path("_tmp/misplacements_all_graphs.json")


def _validate_output_path(p: Path) -> Path:
    """Ensure output path resolves under an allowed root directory."""
    resolved = p.resolve()
    for root in SAFE_OUTPUT_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue
    raise ValueError(f"output path {p} resolves outside allowed directories: {resolved}")


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
    ap = argparse.ArgumentParser(description="Scan all graphs for misplaced episodes")
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    raw = subprocess.check_output(
        ["docker", "exec", FALKORDB_CONTAINER, "redis-cli", "-p", "6379", "GRAPH.LIST"],
        text=True, timeout=SUBPROCESS_TIMEOUT)
    graphs = [x.strip() for x in raw.splitlines() if x.strip() and SAFE_NAME_RE.match(x.strip())]

    rows = []
    for g in graphs:
        total = _count(_cypher(g, "MATCH (e:Episodic) RETURN count(e)"))
        if total == 0:
            continue
        bad = _count(_cypher(
            g, f"MATCH (e:Episodic) WHERE e.group_id IS NOT NULL AND e.group_id <> '{g}' RETURN count(e)"))
        if bad > 0:
            rows.append({"graph": g, "episodes": total, "misplaced": bad})

    rows.sort(key=lambda r: r["misplaced"], reverse=True)
    report = {"graphs_scanned": len(graphs), "affected": len(rows),
              "total_misplaced": sum(r["misplaced"] for r in rows), "details": rows}

    out_path = _validate_output_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"scanned={report['graphs_scanned']}  affected={report['affected']}  "
          f"total_misplaced={report['total_misplaced']}  report={out_path}")
    for r in rows:
        print(f"  {r['graph']:40s}  eps={r['episodes']:>5d}  misplaced={r['misplaced']:>5d}")


if __name__ == "__main__":
    main()
