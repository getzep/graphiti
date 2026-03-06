#!/usr/bin/env python3
"""Build OM closeout markdown report from generated artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


_RUN_ID_RE = re.compile(r'^[A-Za-z0-9_-]+$')
_RUN_ID_MAX_LEN = 128


def _validate_run_id(run_id: str) -> str:
    if len(run_id) > _RUN_ID_MAX_LEN:
        raise ValueError(f'run-id exceeds max length ({_RUN_ID_MAX_LEN})')

    if '/' in run_id:
        raise ValueError("run-id contains '/' which can escape artifact paths")
    if '\\' in run_id:
        raise ValueError("run-id contains '\\' which can escape artifact paths")
    if '..' in run_id:
        raise ValueError("run-id contains '..' path traversal segment")

    if _RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError('run-id must match only [A-Za-z0-9_-]')

    return run_id


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'missing artifact: {path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'artifact must be JSON object: {path}')
    return data


def _artifact_status(path: Path) -> dict[str, Any]:
    return {
        'path': str(path),
        'exists': path.exists(),
        'mtime': datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        .isoformat()
        .replace('+00:00', 'Z')
        if path.exists()
        else None,
    }


def build_report(
    *,
    run_id: str,
    benchmark_path: Path,
    utility_path: Path,
    lane_hygiene_path: Path,
    pr_a_sha: str,
    overlay_manifest_ref: str,
) -> str:
    run_id = _validate_run_id(run_id)

    benchmark = _load_json(benchmark_path)
    utility = _load_json(utility_path)
    lane_hygiene = _load_json(lane_hygiene_path)

    state_dir = benchmark_path.parent
    g2_xml = state_dir / f'g2_lane_routing_{run_id}.xml'
    g3_xml = state_dir / f'g3_adapter_isolation_{run_id}.xml'
    g4_xml = state_dir / f'g4_candidate_bridge_{run_id}.xml'

    utility_agg = utility.get('aggregate', {}) if isinstance(utility, dict) else {}
    benchmark_agg = benchmark.get('bicameral_aggregate', {}) if isinstance(benchmark, dict) else {}

    artifacts = [benchmark_path, utility_path, lane_hygiene_path, g2_xml, g3_xml, g4_xml]

    lines = [
        f'# OM Closeout Report ({run_id})',
        '',
        f'- generated_at: {_utc_now()}',
        f'- PR-A SHA: {pr_a_sha}',
        f'- overlay-manifest: {overlay_manifest_ref}',
        '',
        '## Artifact Manifest',
        '',
    ]

    for artifact in artifacts:
        status = _artifact_status(artifact)
        lines.append(
            f"- `{status['path']}` exists={status['exists']} mtime={status['mtime']}"
        )

    lines.extend(
        [
            '',
            '## Gate Snapshot',
            '',
            f"- G5 benchmark mean recall: {benchmark_agg.get('mean_combined_recall_at_k')}",
            f"- G6 utility net advantage: {utility_agg.get('net_advantage')}",
            f"- G7 lane hygiene unresolved lanes: {lane_hygiene.get('unresolved_lanes', [])}",
            f'- G2 junitxml present: {g2_xml.exists()}',
            f'- G3 junitxml present: {g3_xml.exists()}',
            f'- G4 junitxml present: {g4_xml.exists()}',
            '',
            '## Done',
            '',
            '- Implemented PR-A benchmark/contract/lane-hygiene utility scripts and test gates.',
            '- Generated corrected benchmark and utility evaluation artifacts.',
            '- Audited lane hygiene with explicit keep/deprecate decisions for legacy s1_* lanes.',
            '',
            '## Changed',
            '',
            f'- Harness contract check: enabled fail-closed `--contract-check-only` mode (run_id={run_id}).',
            '- Retrieval validation now expects junitxml artifacts for G2/G3/G4.',
            '- Added utility-vs-QMD scorecards and worksheet output for calibration.',
            '',
            '## Intentionally Not Done',
            '',
            '- No runtime repository PRs were created (runtime remains rebuild-only).',
            '- No destructive lane cleanup was executed; this report covers read-only audit decisions.',
            '',
            '## Rollback',
            '',
            '- Revert PR-A commit(s) in public repo and re-run owned-path + contract preflight.',
            '- Restore previous benchmark harness behavior by removing `--contract-check-only` usage.',
            '- Rebuild runtime from last known-good public/private SHAs and re-verify overlay-manifest.',
            '',
            '## Risks',
            '',
            '- QMD command availability still required for G6 runs.',
            '- Lane hygiene decisions are policy-level declarations and may need operator sign-off for enforcement.',
            '',
        ]
    )

    return '\n'.join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description='Build OM closeout markdown report')
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--benchmark', required=True, help='Benchmark artifact JSON path')
    ap.add_argument('--utility', required=True, help='Utility eval artifact JSON path')
    ap.add_argument('--lane-hygiene', required=True, help='Lane hygiene artifact JSON path')
    ap.add_argument('--out', required=True, help='Output markdown path')
    ap.add_argument('--pr-a-sha', default=os.environ.get('PR_A_SHA', 'UNKNOWN'))
    ap.add_argument('--overlay-manifest-ref', default='overlay-manifest.json')
    args = ap.parse_args()

    try:
        run_id = _validate_run_id(args.run_id)
    except ValueError as exc:
        print(f'Invalid --run-id: {exc}', file=sys.stderr)
        return 1

    try:
        report = build_report(
            run_id=run_id,
            benchmark_path=Path(args.benchmark),
            utility_path=Path(args.utility),
            lane_hygiene_path=Path(args.lane_hygiene),
            pr_a_sha=args.pr_a_sha,
            overlay_manifest_ref=args.overlay_manifest_ref,
        )
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding='utf-8')
    print(f'closeout report: {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
