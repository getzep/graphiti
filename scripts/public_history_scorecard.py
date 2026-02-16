#!/usr/bin/env python3
"""Compare migration candidates and select cutover recommendation."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from delta_contracts import validate_migration_sync_policy
from migration_sync_lib import dump_json, load_json, now_utc_iso

METRIC_KEYS = ('privacy_risk', 'simplicity', 'merge_conflict_risk', 'auditability')
DEFAULT_THRESHOLD = 80.0
DEFAULT_WEIGHTS = {
    'privacy_risk': 0.35,
    'simplicity': 0.35,
    'merge_conflict_risk': 0.2,
    'auditability': 0.1,
}
DEFAULT_BRANCHES = {
    'filtered-history': 'cutover/filtered-history',
    'clean-foundation': 'cutover/clean-foundation',
}

MODE_PATTERN = re.compile(r'^# Public History Candidate Report — (?P<mode>.+)$')
METRIC_PATTERN = re.compile(
    r'^- (?P<metric>privacy_risk|simplicity|merge_conflict_risk|auditability): `(?P<value>\d+)`$',
)
HIGH_PATTERN = re.compile(r'^- unresolved_high: `(?P<value>True|False)`$')
BRANCH_PATTERN = re.compile(r'^- Candidate branch \(planned\): `(?P<branch>[^`]+)`$')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compute public history migration scorecard.')
    parser.add_argument('--filtered-report', type=Path, help='Filtered-history markdown report path')
    parser.add_argument('--clean-report', type=Path, help='Clean-foundation markdown report path')

    # Legacy args kept for compatibility with existing tests/callers.
    parser.add_argument('--filtered-summary', type=Path, help='Filtered-history JSON summary path')
    parser.add_argument('--clean-summary', type=Path, help='Clean-foundation JSON summary path')

    parser.add_argument(
        '--policy',
        type=Path,
        default=Path('config/migration_sync_policy.json'),
        help='Policy JSON with score thresholds/weights',
    )
    parser.add_argument('--out', type=Path, required=True, help='Markdown scorecard output path')
    parser.add_argument('--summary-json', type=Path, help='Optional machine-readable output')
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _normalize_weights(raw_weights: dict[str, Any]) -> dict[str, float]:
    missing = [key for key in METRIC_KEYS if key not in raw_weights]
    if missing:
        raise ValueError(f'Missing score weights for: {", ".join(missing)}')

    normalized = {key: float(raw_weights[key]) for key in METRIC_KEYS}
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError('Score weights must sum to > 0')

    return {key: normalized[key] / total for key in METRIC_KEYS}


def _weighted_total(metrics: dict[str, float], weights: dict[str, float]) -> float:
    return round(sum(float(metrics[key]) * float(weights[key]) for key in METRIC_KEYS), 2)


def _parse_report(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        raise FileNotFoundError(f'Report not found: {report_path}')

    mode: str | None = None
    branch: str | None = None
    unresolved_high = False
    metrics: dict[str, int] = {}

    for line in report_path.read_text(encoding='utf-8').splitlines():
        mode_match = MODE_PATTERN.match(line)
        if mode_match:
            mode = mode_match.group('mode').strip()
            continue

        metric_match = METRIC_PATTERN.match(line)
        if metric_match:
            metrics[metric_match.group('metric')] = int(metric_match.group('value'))
            continue

        high_match = HIGH_PATTERN.match(line)
        if high_match:
            unresolved_high = high_match.group('value') == 'True'
            continue

        branch_match = BRANCH_PATTERN.match(line)
        if branch_match:
            branch = branch_match.group('branch').strip()

    missing_metrics = [metric for metric in METRIC_KEYS if metric not in metrics]
    if missing_metrics:
        raise ValueError(
            f'Report {report_path} is missing metrics: {", ".join(missing_metrics)}',
        )

    candidate_mode = mode or report_path.stem

    return {
        'mode': candidate_mode,
        'metrics': metrics,
        'risk_flags': {'unresolved_high': unresolved_high},
        'candidate_branch': branch or DEFAULT_BRANCHES.get(candidate_mode, f'cutover/{candidate_mode}'),
    }


def _resolve_candidate_payload(
    *,
    label: str,
    report_arg: Path | None,
    summary_arg: Path | None,
) -> dict[str, Any]:
    if summary_arg is not None:
        return load_json(_resolve(summary_arg))

    if report_arg is None:
        raise ValueError(
            f'Missing candidate input for {label}: provide --{label}-report or --{label}-summary',
        )

    report_path = _resolve(report_arg)
    sibling_summary = report_path.with_suffix('.json')
    if sibling_summary.exists():
        return load_json(sibling_summary)

    return _parse_report(report_path)


def _coerce_metrics(payload: dict[str, Any], *, context: str) -> dict[str, float]:
    metrics = payload.get('metrics')
    if not isinstance(metrics, dict):
        raise ValueError(f'{context} summary must contain a metrics object')

    coerced: dict[str, float] = {}
    for key in METRIC_KEYS:
        if key not in metrics:
            raise ValueError(f'{context} summary missing metric: {key}')
        coerced[key] = float(metrics[key])

    return coerced


def _candidate_branch(payload: dict[str, Any], fallback_mode: str) -> str:
    raw = payload.get('candidate_branch')
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return DEFAULT_BRANCHES.get(fallback_mode, f'cutover/{fallback_mode}')


def _winner_cutover_commands(decision: str, branch: str) -> list[str]:
    if decision == 'clean-foundation':
        return [
            'git fetch origin',
            f'git checkout --orphan {branch} origin/main',
            'git add -A',
            'git commit -m "chore(public): clean-foundation cutover snapshot"',
            f'git push --force-with-lease origin {branch}',
            f'# after review approval: git push --force-with-lease origin {branch}:main',
        ]

    return [
        'git fetch origin',
        f'git checkout -B {branch} origin/main',
        'python3 scripts/public_history_export.py --mode filtered-history --dry-run',
        '# apply approved filtered-history rewrite plan, then:',
        f'git push --force-with-lease origin {branch}',
        f'# after review approval: git push --force-with-lease origin {branch}:main',
    ]


def _rollback_commands(pre_cutover_ref: str) -> list[str]:
    return [
        f'git tag pre-cutover-main {pre_cutover_ref}',
        '# if rollback is required:',
        f'git push --force-with-lease origin {pre_cutover_ref}:main',
    ]


def _format_bash_block(commands: list[str]) -> list[str]:
    return ['```bash', *commands, '```']


def main() -> int:
    args = parse_args()

    filtered_payload = _resolve_candidate_payload(
        label='filtered',
        report_arg=args.filtered_report,
        summary_arg=args.filtered_summary,
    )
    clean_payload = _resolve_candidate_payload(
        label='clean',
        report_arg=args.clean_report,
        summary_arg=args.clean_summary,
    )

    policy_path = _resolve(args.policy)
    policy = validate_migration_sync_policy(load_json(policy_path), context=str(policy_path))

    score_cfg = policy.get('scorecard', {})
    threshold = float(score_cfg.get('clean_foundation_threshold', DEFAULT_THRESHOLD))
    weights_raw = score_cfg.get('weights', DEFAULT_WEIGHTS)
    if not isinstance(weights_raw, dict):
        raise ValueError('scorecard.weights must be an object')
    weights = _normalize_weights(weights_raw)

    filtered_metrics = _coerce_metrics(filtered_payload, context='filtered')
    clean_metrics = _coerce_metrics(clean_payload, context='clean')

    filtered_total = _weighted_total(filtered_metrics, weights)
    clean_total = _weighted_total(clean_metrics, weights)

    filtered_unresolved_high = bool(filtered_payload.get('risk_flags', {}).get('unresolved_high', False))

    filtered_branch = _candidate_branch(filtered_payload, 'filtered-history')
    clean_branch = _candidate_branch(clean_payload, 'clean-foundation')

    decision_reason: str
    if filtered_unresolved_high:
        decision = 'clean-foundation'
        decision_reason = 'Filtered-history has unresolved HIGH risk after one remediation pass.'
    elif filtered_total < threshold:
        decision = 'clean-foundation'
        decision_reason = (
            f'Filtered-history score {filtered_total} is below threshold {threshold}. '
            'Clean-foundation fallback is mandatory.'
        )
    else:
        decision = 'filtered-history' if filtered_total >= clean_total else 'clean-foundation'
        decision_reason = f'Winner by weighted score comparison ({filtered_total} vs {clean_total}).'

    winner_branch = filtered_branch if decision == 'filtered-history' else clean_branch

    pre_cutover_ref = 'origin/main'
    for payload in (clean_payload, filtered_payload):
        git_meta = payload.get('git')
        if isinstance(git_meta, dict):
            head_sha = git_meta.get('head_sha')
            if isinstance(head_sha, str) and head_sha.strip():
                pre_cutover_ref = head_sha.strip()
                break

    winner_commands = _winner_cutover_commands(decision, winner_branch)
    rollback_commands = _rollback_commands(pre_cutover_ref)

    report = {
        'generated_at': now_utc_iso(),
        'decision': decision,
        'decision_reason': decision_reason,
        'threshold': threshold,
        'weights': weights,
        'scores': {
            'filtered-history': filtered_total,
            'clean-foundation': clean_total,
        },
        'branches': {
            'filtered-history': filtered_branch,
            'clean-foundation': clean_branch,
            'winner': winner_branch,
        },
        'filtered_unresolved_high': filtered_unresolved_high,
        'winner_cutover_commands': winner_commands,
        'rollback_commands': rollback_commands,
    }

    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        '# Public History Migration Scorecard',
        '',
        f"- Generated: `{report['generated_at']}`",
        f"- Threshold (clean fallback): `{threshold}`",
        '- Rule: choose clean-foundation if filtered-history score is below threshold or '
        'an unresolved HIGH finding remains after one remediation pass.',
        '',
        '## Weighted scores',
        '',
        '| Candidate | Score |',
        '| --- | --- |',
        f'| filtered-history | {filtered_total} |',
        f'| clean-foundation | {clean_total} |',
        '',
        '## Weights',
        '',
        '| Metric | Weight |',
        '| --- | --- |',
        *[f'| {metric} | {weights[metric]:.2f} |' for metric in METRIC_KEYS],
        '',
        '## Candidate branches',
        '',
        f"- filtered-history: `{filtered_branch}`",
        f"- clean-foundation: `{clean_branch}`",
        '',
        '## Decision',
        '',
        f"- Winner: `{decision}`",
        f"- Winner branch: `{winner_branch}`",
        f'- Reason: {decision_reason}',
        f"- filtered unresolved HIGH: `{filtered_unresolved_high}`",
        '',
        '## Cutover commands (winner)',
        '',
        *_format_bash_block(winner_commands),
        '',
        '## Rollback plan',
        '',
        *_format_bash_block(rollback_commands),
    ]

    out_path.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')

    if args.summary_json:
        dump_json(_resolve(args.summary_json), report)

    print(f'Scorecard written: {out_path}')
    print(f'Decision: {decision} ({decision_reason})')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
