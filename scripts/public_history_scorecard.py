#!/usr/bin/env python3
"""Compare migration candidates and select cutover recommendation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from delta_contracts import validate_migration_sync_policy
from migration_sync_lib import dump_json, load_json, now_utc_iso

METRIC_KEYS = ('privacy_risk', 'simplicity', 'merge_conflict_risk', 'auditability')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compute public history migration scorecard.')
    parser.add_argument('--filtered-summary', type=Path, required=True)
    parser.add_argument('--clean-summary', type=Path, required=True)
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


def _weighted_total(metrics: dict, weights: dict[str, float]) -> float:
    return round(sum(float(metrics[key]) * float(weights[key]) for key in METRIC_KEYS), 2)


def _normalize_weights(raw_weights: dict) -> dict[str, float]:
    missing = [key for key in METRIC_KEYS if key not in raw_weights]
    if missing:
        raise ValueError(f'Missing score weights for: {", ".join(missing)}')
    normalized = {key: float(raw_weights[key]) for key in METRIC_KEYS}
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError('Score weights must sum to > 0')
    return {key: normalized[key] / total for key in METRIC_KEYS}


def main() -> int:
    args = parse_args()
    filtered_summary = load_json(_resolve(args.filtered_summary))
    clean_summary = load_json(_resolve(args.clean_summary))
    policy_path = _resolve(args.policy)
    policy = validate_migration_sync_policy(load_json(policy_path), context=str(policy_path))

    score_cfg = policy.get('scorecard', {})
    threshold = float(score_cfg.get('clean_foundation_threshold', 80))
    weights = _normalize_weights(score_cfg.get('weights', {}))

    filtered_metrics = filtered_summary.get('metrics', {})
    clean_metrics = clean_summary.get('metrics', {})
    if not isinstance(filtered_metrics, dict) or not isinstance(clean_metrics, dict):
        raise ValueError('Summary files must contain metrics objects')

    filtered_total = _weighted_total(filtered_metrics, weights)
    clean_total = _weighted_total(clean_metrics, weights)

    filtered_unresolved_high = bool(filtered_summary.get('risk_flags', {}).get('unresolved_high', False))

    decision_reason: str
    if filtered_unresolved_high:
        decision = 'clean-foundation'
        decision_reason = 'Filtered-history has unresolved HIGH risk flag.'
    elif filtered_total < threshold:
        decision = 'clean-foundation'
        decision_reason = (
            f'Filtered-history score {filtered_total} is below threshold {threshold}. '
            'Use clean-foundation fallback per policy.'
        )
    else:
        decision = 'filtered-history' if filtered_total >= clean_total else 'clean-foundation'
        decision_reason = f'Winner by weighted score comparison ({filtered_total} vs {clean_total}).'

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
        'filtered_unresolved_high': filtered_unresolved_high,
    }

    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        '# Public History Migration Scorecard',
        '',
        f'- Generated: `{report["generated_at"]}`',
        f'- Threshold (clean fallback): `{threshold}`',
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
        '## Decision',
        '',
        f'- Winner: `{decision}`',
        f'- Reason: {decision_reason}',
        f'- filtered unresolved HIGH: `{filtered_unresolved_high}`',
        '',
        '## Policy rule',
        '',
        '- Choose clean-foundation automatically if filtered-history score is below threshold '
        'or unresolved HIGH remains after one remediation pass.',
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
