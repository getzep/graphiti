#!/usr/bin/env python3
"""Generate migration-candidate reports for public history cutover planning."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from delta_contracts import validate_migration_sync_policy
from migration_sync_lib import dump_json, load_json, now_utc_iso, resolve_repo_root, run_git
from public_boundary_policy import (
    ALLOW,
    AMBIGUOUS,
    BLOCK,
    classify_path,
    collect_git_files,
    read_yaml_list,
    summarize_decisions,
)

DEFAULT_HISTORY_METRICS: dict[str, Any] = {
    'filtered_history': {
        'privacy_risk': {
            'base': 100,
            'block_penalty': 35,
            'ambiguous_penalty': 0.5,
        },
        'simplicity': {
            'base': 100,
            'commit_divisor': 15,
            'commit_cap': 35,
            'ambiguous_penalty': 0.3,
        },
        'merge_conflict_risk': {
            'base': 100,
            'commit_divisor': 20,
            'commit_cap': 30,
            'ambiguous_penalty': 0.2,
        },
        'auditability': {
            'base': 100,
            'block_penalty': 20,
            'ambiguous_penalty': 0.4,
        },
    },
    'clean_foundation': {
        'privacy_risk': {
            'base': 97,
        },
        'simplicity': {
            'base': 90,
            'commit_bonus_divisor': 100,
            'commit_bonus_cap': 6,
        },
        'merge_conflict_risk': {
            'base': 92,
        },
        'auditability': {
            'base': 90,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate migration candidate report for public history cutover.')
    parser.add_argument('--repo', type=Path, default=Path('.'), help='Repository root or subdirectory')
    parser.add_argument('--mode', choices=('filtered-history', 'clean-foundation'), required=True)
    parser.add_argument(
        '--manifest',
        type=Path,
        default=Path('config/public_export_allowlist.yaml'),
        help='Allowlist policy path',
    )
    parser.add_argument(
        '--denylist',
        type=Path,
        default=Path('config/public_export_denylist.yaml'),
        help='Denylist policy path',
    )
    parser.add_argument(
        '--policy',
        type=Path,
        default=Path('config/migration_sync_policy.json'),
        help='Optional migration/sync policy JSON with history metric coefficients',
    )
    parser.add_argument('--report', type=Path, required=True, help='Markdown output path')
    parser.add_argument('--summary-json', type=Path, help='Optional JSON summary output path')
    parser.add_argument('--dry-run', action='store_true', help='Emit planning report only (no git rewrites)')
    return parser.parse_args()


def _git_count(repo_root: Path, *args: str) -> int:
    result = run_git(repo_root, *args, check=False)
    if result.returncode != 0:
        return 0
    output = result.stdout.strip()
    if output.isdigit():
        return int(output)
    lines = [line for line in output.splitlines() if line.strip()]
    return len(lines)


def _number(value: object, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) else float(default)


def _metric_cfg(history_cfg: dict[str, Any], candidate: str, metric: str) -> dict[str, Any]:
    candidate_cfg = history_cfg.get(candidate, {})
    if not isinstance(candidate_cfg, dict):
        return {}
    metric_cfg = candidate_cfg.get(metric, {})
    return metric_cfg if isinstance(metric_cfg, dict) else {}


def _clamp_metric(value: float) -> int:
    return int(max(0, min(100, round(value))))


def _calc_filtered_metrics(
    commit_count: int,
    block_count: int,
    ambiguous_count: int,
    history_cfg: dict[str, Any],
) -> dict[str, int]:
    privacy_cfg = _metric_cfg(history_cfg, 'filtered_history', 'privacy_risk')
    simplicity_cfg = _metric_cfg(history_cfg, 'filtered_history', 'simplicity')
    merge_cfg = _metric_cfg(history_cfg, 'filtered_history', 'merge_conflict_risk')
    audit_cfg = _metric_cfg(history_cfg, 'filtered_history', 'auditability')

    privacy = _number(privacy_cfg.get('base'), 100) - (
        block_count * _number(privacy_cfg.get('block_penalty'), 35)
    ) - (ambiguous_count * _number(privacy_cfg.get('ambiguous_penalty'), 0.5))

    simplicity_divisor = _number(simplicity_cfg.get('commit_divisor'), 15)
    simplicity_cap = _number(simplicity_cfg.get('commit_cap'), 35)
    simplicity_commit_penalty = min(
        int(commit_count / simplicity_divisor) if simplicity_divisor > 0 else 0,
        simplicity_cap,
    )
    simplicity = _number(simplicity_cfg.get('base'), 100) - simplicity_commit_penalty - (
        ambiguous_count * _number(simplicity_cfg.get('ambiguous_penalty'), 0.3)
    )

    merge_divisor = _number(merge_cfg.get('commit_divisor'), 20)
    merge_cap = _number(merge_cfg.get('commit_cap'), 30)
    merge_commit_penalty = min(
        int(commit_count / merge_divisor) if merge_divisor > 0 else 0,
        merge_cap,
    )
    merge_conflict = _number(merge_cfg.get('base'), 100) - merge_commit_penalty - (
        ambiguous_count * _number(merge_cfg.get('ambiguous_penalty'), 0.2)
    )

    auditability = _number(audit_cfg.get('base'), 100) - (
        block_count * _number(audit_cfg.get('block_penalty'), 20)
    ) - (ambiguous_count * _number(audit_cfg.get('ambiguous_penalty'), 0.4))

    return {
        'privacy_risk': _clamp_metric(privacy),
        'simplicity': _clamp_metric(simplicity),
        'merge_conflict_risk': _clamp_metric(merge_conflict),
        'auditability': _clamp_metric(auditability),
    }


def _calc_clean_metrics(commit_count: int, history_cfg: dict[str, Any]) -> dict[str, int]:
    privacy_cfg = _metric_cfg(history_cfg, 'clean_foundation', 'privacy_risk')
    simplicity_cfg = _metric_cfg(history_cfg, 'clean_foundation', 'simplicity')
    merge_cfg = _metric_cfg(history_cfg, 'clean_foundation', 'merge_conflict_risk')
    audit_cfg = _metric_cfg(history_cfg, 'clean_foundation', 'auditability')

    simplicity_base = _number(simplicity_cfg.get('base'), 90)
    bonus_divisor = _number(simplicity_cfg.get('commit_bonus_divisor'), 100)
    bonus_cap = _number(simplicity_cfg.get('commit_bonus_cap'), 6)
    simplicity_bonus = min(int(commit_count / bonus_divisor) if bonus_divisor > 0 else 0, bonus_cap)

    return {
        'privacy_risk': _clamp_metric(_number(privacy_cfg.get('base'), 97)),
        'simplicity': _clamp_metric(simplicity_base + simplicity_bonus),
        'merge_conflict_risk': _clamp_metric(_number(merge_cfg.get('base'), 92)),
        'auditability': _clamp_metric(_number(audit_cfg.get('base'), 90)),
    }


def _load_history_metrics(policy_path: Path) -> tuple[dict[str, Any], bool]:
    if not policy_path.exists():
        return DEFAULT_HISTORY_METRICS, False

    policy_payload = validate_migration_sync_policy(load_json(policy_path), context=str(policy_path))
    history_cfg = policy_payload.get('history_metrics')
    if not isinstance(history_cfg, dict):
        return DEFAULT_HISTORY_METRICS, True

    merged: dict[str, Any] = {
        'filtered_history': dict(DEFAULT_HISTORY_METRICS['filtered_history']),
        'clean_foundation': dict(DEFAULT_HISTORY_METRICS['clean_foundation']),
    }

    for candidate in ('filtered_history', 'clean_foundation'):
        candidate_cfg = history_cfg.get(candidate, {})
        if not isinstance(candidate_cfg, dict):
            continue
        for metric, defaults in DEFAULT_HISTORY_METRICS[candidate].items():
            metric_cfg = candidate_cfg.get(metric, {})
            if isinstance(metric_cfg, dict):
                merged[candidate][metric] = {**defaults, **metric_cfg}
            else:
                merged[candidate][metric] = dict(defaults)

    return merged, True


def main() -> int:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo.resolve())

    manifest_path = args.manifest if args.manifest.is_absolute() else (repo_root / args.manifest).resolve()
    denylist_path = args.denylist if args.denylist.is_absolute() else (repo_root / args.denylist).resolve()
    policy_path = args.policy if args.policy.is_absolute() else (repo_root / args.policy).resolve()
    report_path = args.report if args.report.is_absolute() else (Path.cwd() / args.report).resolve()

    allowlist = read_yaml_list(manifest_path, 'allowlist')
    denylist = read_yaml_list(denylist_path, 'denylist')
    history_metrics, policy_loaded = _load_history_metrics(policy_path)

    files = collect_git_files(repo_root, include_untracked=False)
    decisions = [classify_path(path, allowlist=allowlist, denylist=denylist) for path in files]
    counts, blocked, ambiguous = summarize_decisions(decisions)

    commit_count = _git_count(repo_root, 'rev-list', '--count', 'HEAD')
    author_count = _git_count(repo_root, 'shortlog', '-sn', 'HEAD')

    if args.mode == 'filtered-history':
        metrics = _calc_filtered_metrics(commit_count, len(blocked), len(ambiguous), history_metrics)
        unresolved_high = len(blocked) > 0
        rationale = (
            'Filtered-history keeps more provenance but inherits policy ambiguity and privacy review burden.'
        )
    else:
        metrics = _calc_clean_metrics(commit_count, history_metrics)
        unresolved_high = False
        rationale = 'Clean-foundation minimizes carry-over complexity and should reduce long-term merge friction.'

    summary = {
        'mode': args.mode,
        'generated_at': now_utc_iso(),
        'dry_run': bool(args.dry_run),
        'repo_root': str(repo_root),
        'policy': {
            'allowlist': str(manifest_path),
            'denylist': str(denylist_path),
            'history_metrics': str(policy_path) if policy_loaded else 'DEFAULTS',
        },
        'git': {
            'commit_count': commit_count,
            'author_count': author_count,
        },
        'boundary_counts': {
            ALLOW: counts[ALLOW],
            BLOCK: counts[BLOCK],
            AMBIGUOUS: counts[AMBIGUOUS],
        },
        'metrics': metrics,
        'risk_flags': {
            'unresolved_high': unresolved_high,
        },
        'rationale': rationale,
    }

    report_lines = [
        f'# Public History Candidate Report — {args.mode}',
        '',
        f'- Generated: `{summary["generated_at"]}`',
        f'- Repo: `{repo_root}`',
        f'- Dry run: `{args.dry_run}`',
        '',
        '## Inputs',
        '',
        f'- Allowlist: `{manifest_path}`',
        f'- Denylist: `{denylist_path}`',
        f"- History metric policy: `{summary['policy']['history_metrics']}`",
        '',
        '## Repository baseline',
        '',
        f'- Commits: `{commit_count}`',
        f'- Authors: `{author_count}`',
        f'- Paths scanned: `{len(files)}`',
        '',
        '## Boundary counts',
        '',
        f'- ALLOW: `{counts[ALLOW]}`',
        f'- BLOCK: `{counts[BLOCK]}`',
        f'- AMBIGUOUS: `{counts[AMBIGUOUS]}`',
        '',
        '## Candidate metrics (0-100, higher is better)',
        '',
        f'- privacy_risk: `{metrics["privacy_risk"]}`',
        f'- simplicity: `{metrics["simplicity"]}`',
        f'- merge_conflict_risk: `{metrics["merge_conflict_risk"]}`',
        f'- auditability: `{metrics["auditability"]}`',
        '',
        '## Rationale',
        '',
        f'- {rationale}',
        f'- unresolved_high: `{unresolved_high}`',
        '',
        '## Next step',
        '',
        '- Feed this summary JSON into `scripts/public_history_scorecard.py` for winner selection.',
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(report_lines).rstrip() + '\n', encoding='utf-8')

    if args.summary_json:
        summary_path = args.summary_json if args.summary_json.is_absolute() else (Path.cwd() / args.summary_json).resolve()
        dump_json(summary_path, summary)
        print(f'Summary JSON written: {summary_path}')

    print(f'Report written: {report_path}')
    print(
        'Metrics: '
        f"privacy={metrics['privacy_risk']} simplicity={metrics['simplicity']} "
        f"merge_conflict={metrics['merge_conflict_risk']} auditability={metrics['auditability']}",
    )
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
