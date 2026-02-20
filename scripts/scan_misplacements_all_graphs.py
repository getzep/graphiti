#!/usr/bin/env python3
"""Scan all graphs for misplaced episodic nodes.

A node is misplaced if it lives in graph G but has group_id != G.
Outputs a JSON report and prints a summary table.

On Neo4j (single DB) misplacements cannot occur — the script reports zero and exits.

Supports both Neo4j and FalkorDB via --backend flag.

Usage:
  python3 scripts/scan_misplacements_all_graphs.py
  python3 scripts/scan_misplacements_all_graphs.py -o /tmp/report.json
  python3 scripts/scan_misplacements_all_graphs.py --backend falkordb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from graph_cli import list_graphs, parse_count, run_cypher

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan all graphs for misplaced episodes")
    ap.add_argument("--backend", choices=["neo4j", "falkordb"], default="neo4j",
                    help="graph database backend (default: neo4j)")
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    backend = args.backend

    if backend == "neo4j":
        # Neo4j uses a single database — cross-graph misplacement cannot occur.
        report = {"graphs_scanned": 0, "affected": 0, "total_misplaced": 0, "details": []}
        out_path = _validate_output_path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print("backend=neo4j — single DB, misplacement N/A")
        print(f"scanned=0  affected=0  total_misplaced=0  report={out_path}")
        return

    graphs = list_graphs(backend)

    rows = []
    for g in graphs:
        total = parse_count(run_cypher(
            backend, g, "MATCH (e:Episodic) RETURN count(e)"))
        if total is None or total == 0:
            continue
        bad = parse_count(run_cypher(
            backend, g,
            f"MATCH (e:Episodic) WHERE e.group_id IS NOT NULL AND e.group_id <> '{g}' RETURN count(e)"))
        if bad is None:
            continue
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
