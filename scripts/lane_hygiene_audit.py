#!/usr/bin/env python3
"""Read-only Phase 4 lane hygiene audit.

Audits legacy ``s1_*`` lane dependencies and enforces explicit keep/deprecate
classification per lane.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_truth_candidates = importlib.import_module('truth.candidates')
LANE_CORROBORATION_ONLY = _truth_candidates.LANE_CORROBORATION_ONLY
LANE_RETRIEVAL_ELIGIBLE_GLOBAL = _truth_candidates.LANE_RETRIEVAL_ELIGIBLE_GLOBAL
LANE_RETRIEVAL_ELIGIBLE_VC_SCOPED = _truth_candidates.LANE_RETRIEVAL_ELIGIBLE_VC_SCOPED

DECISIONS_BY_LANE: dict[str, str] = {
    's1_sessions_main': 'keep',
    's1_observational_memory': 'keep',
    's1_chatgpt_history': 'keep-scoped',
    's1_curated_refs': 'keep-corroboration-only',
    's1_memory_day1': 'deprecate-after-migration',
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _run_git_grep(repo_root: Path, needle: str) -> list[dict[str, Any]]:
    cmd = ['git', 'grep', '-n', '--fixed-strings', needle, '--', '.']
    proc = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
    if proc.returncode not in (0, 1):
        raise RuntimeError(f'git grep failed for {needle!r}: {proc.stderr.strip()}')

    matches: list[dict[str, Any]] = []
    for raw in proc.stdout.splitlines():
        # Format: path:line:snippet
        try:
            path, line, snippet = raw.split(':', 2)
        except ValueError:
            continue

        category = 'other'
        if path.startswith('mcp_server/'):
            category = 'reader'
        elif path.startswith('scripts/runtime_pack_router.py') or path.startswith('scripts/run_incremental_ingest.py'):
            category = 'router_or_cron'
        elif path.startswith('scripts/'):
            category = 'writer_or_tooling'

        matches.append(
            {
                'path': path,
                'line': int(line),
                'snippet': snippet.strip(),
                'category': category,
            }
        )

    return matches


def build_lane_hygiene_audit(repo_root: Path) -> dict[str, Any]:
    lane_ids = sorted(
        set(LANE_RETRIEVAL_ELIGIBLE_GLOBAL)
        | set(LANE_RETRIEVAL_ELIGIBLE_VC_SCOPED)
        | set(LANE_CORROBORATION_ONLY)
    )

    lane_entries: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for lane_id in lane_ids:
        refs = _run_git_grep(repo_root, lane_id)
        decision = DECISIONS_BY_LANE.get(lane_id)
        if decision is None:
            unresolved.append(lane_id)

        lane_entries.append(
            {
                'lane_id': lane_id,
                'decision': decision,
                'references': refs,
                'reference_count': len(refs),
            }
        )

    return {
        'timestamp': _utc_now(),
        'repo_root': str(repo_root),
        'lanes': lane_entries,
        'unresolved_lanes': sorted(unresolved),
        'ok': len(unresolved) == 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='Read-only lane hygiene audit')
    ap.add_argument('--output', required=True, help='Output JSON path')
    ap.add_argument('--repo-root', default='.', help='Path inside target git repo')
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    audit = build_lane_hygiene_audit(repo_root)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')

    print(f'lane hygiene artifact: {out_path}')
    if not audit['ok']:
        print('lane hygiene FAILED: unresolved lane decisions')
        for lane in audit['unresolved_lanes']:
            print(f'  unresolved: {lane}')
        return 1

    print('lane hygiene ok')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
