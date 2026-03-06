#!/usr/bin/env python3
"""Owned-path boundary preflight for PR-A public repo scope.

Validates that changed files are a strict subset of the PRD-declared owned paths.
Emits: state/owned_paths_preflight_<run_id>.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

OWNED_PATHS: set[str] = {
    'docs/runbooks/om-operations.md',
    'docs/runbooks/sessions-ingestion.md',
    'mcp_server/src/graphiti_mcp_server.py',
    'mcp_server/src/services/queue_service.py',
    'scripts/build_om_closeout_report.py',
    'scripts/lane_hygiene_audit.py',
    'scripts/owned_paths_preflight.py',
    'scripts/run_retrieval_benchmark.py',
    'scripts/utility_eval_vs_qmd.py',
    'tests/fixtures/retrieval_benchmark_queries.json',
    'tests/helpers_mcp_import.py',
    'tests/test_falkordb_all_lanes_bypass_om_adapter.py',
    'tests/test_lane_alias_resolution_to_group_ids_is_explicit.py',
    'tests/test_lane_aliases.py',
    'tests/test_mcp_tool_schema_contract_matches_harness_args.py',
    'tests/test_mixed_lane_query_returns_fused_results_with_lane_provenance.py',
    'tests/test_om_candidate_bridge_emits_candidate_rows.py',
    'tests/test_om_only_query_returns_om_evidence.py',
    'tests/test_retrieval_benchmark.py',
    'tests/test_run_id_sanitization_scripts.py',
}

_RUN_ID_RE = re.compile(r'^[A-Za-z0-9_-]+$')
_RUN_ID_MAX_LEN = 128


def _validate_run_id(run_id: str) -> str:
    if len(run_id) > _RUN_ID_MAX_LEN:
        raise ValueError(f'run-id exceeds max length ({_RUN_ID_MAX_LEN})')

    if '/' in run_id:
        raise ValueError("run-id contains '/' which can escape state output directory")
    if '\\' in run_id:
        raise ValueError("run-id contains '\\' which can escape state output directory")
    if '..' in run_id:
        raise ValueError("run-id contains '..' path traversal segment")

    if _RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError('run-id must match only [A-Za-z0-9_-]')

    return run_id


def _run(cmd: list[str], cwd: Path) -> str:
    return subprocess.check_output(cmd, cwd=str(cwd), text=True).strip()


def _repo_root(start: Path) -> Path:
    return Path(_run(['git', 'rev-parse', '--show-toplevel'], cwd=start))


def _changed_from_ref(repo_root: Path, base_ref: str) -> set[str]:
    changed = _run(['git', 'diff', '--name-only', f'{base_ref}...HEAD'], cwd=repo_root)
    return {line.strip() for line in changed.splitlines() if line.strip()}


def _changed_worktree(repo_root: Path) -> set[str]:
    raw_status = subprocess.check_output(
        ['git', 'status', '--porcelain'], cwd=str(repo_root), text=True
    )
    paths: set[str] = set()
    for raw in raw_status.splitlines():
        if not raw:
            continue
        # porcelain format: XY <path> (or XY <old> -> <new> for renames)
        # Keep leading spaces intact (do not global-strip the full output),
        # otherwise first-line paths can lose their first character.
        if len(raw) < 4:
            continue
        path_part = raw[3:]
        if ' -> ' in path_part:
            _, new_path = path_part.split(' -> ', 1)
            paths.add(new_path.strip())
        else:
            paths.add(path_part.strip())
    return {p for p in paths if p}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def main() -> int:
    ap = argparse.ArgumentParser(description='Owned-path boundary preflight')
    ap.add_argument('--run-id', required=True, help='Run identifier for artifact naming')
    ap.add_argument('--base-ref', default='origin/main', help='Baseline git ref (default: origin/main)')
    ap.add_argument('--repo-root', default='.', help='Path inside target git repo')
    args = ap.parse_args()

    try:
        run_id = _validate_run_id(args.run_id)
    except ValueError as exc:
        print(f'Invalid --run-id: {exc}', file=sys.stderr)
        return 1

    repo_root = _repo_root(Path(args.repo_root).resolve())

    changed_paths = sorted(
        _changed_from_ref(repo_root, args.base_ref) | _changed_worktree(repo_root)
    )
    out_of_scope = [p for p in changed_paths if p not in OWNED_PATHS]

    artifact = {
        'run_id': run_id,
        'timestamp': _utc_now(),
        'repo_root': str(repo_root),
        'base_ref': args.base_ref,
        'owned_paths': sorted(OWNED_PATHS),
        'changed_paths': changed_paths,
        'out_of_scope_paths': out_of_scope,
        'ok': len(out_of_scope) == 0,
    }

    out_path = repo_root / 'state' / f'owned_paths_preflight_{run_id}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2) + '\n', encoding='utf-8')

    if out_of_scope:
        print('owned-path preflight FAILED')
        for path in out_of_scope:
            print(f'  out-of-scope: {path}')
        print(f'artifact: {out_path}')
        return 1

    print('owned-path preflight ok')
    print(f'artifact: {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
