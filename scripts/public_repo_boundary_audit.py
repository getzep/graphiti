#!/usr/bin/env python3
"""Audit repository paths against allowlist/denylist boundary policy."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from public_boundary_policy import (
    STATUS_ORDER,
    build_markdown_report,
    classify_path,
    collect_git_files,
    read_yaml_list,
    resolve_input_path,
    resolve_repo_root,
    summarize_decisions,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description='Run Public Foundation Boundary audit using allowlist and denylist rules.',
    )
    parser.add_argument('--manifest', required=True, type=Path, help='Path to allowlist YAML file')
    parser.add_argument('--denylist', required=True, type=Path, help='Path to denylist YAML file')
    parser.add_argument('--report', type=Path, required=True, help='Output markdown report path')
    parser.add_argument(
        '--summary-json',
        type=Path,
        help='Optional JSON summary output path for CI/reporting automation',
    )
    parser.add_argument(
        '--include-untracked',
        action='store_true',
        help='Include untracked files (git ls-files --others --exclude-standard)',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Exit with non-zero when BLOCK or AMBIGUOUS paths are detected',
    )
    return parser.parse_args()


def _write_summary_json(
    output_path: Path,
    status_counts: dict[str, int],
    blocked_paths: list[str],
    ambiguous_paths: list[str],
) -> None:
    """Write machine-readable summary output for CI/report parsers."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'status_counts': {status: status_counts[status] for status in STATUS_ORDER},
        'blocked_paths': blocked_paths,
        'ambiguous_paths': ambiguous_paths,
    }
    output_path.write_text(f'{json.dumps(payload, indent=2, sort_keys=True)}\n', encoding='utf-8')


def main() -> int:
    """Run the audit and write the markdown report."""

    args = parse_args()
    repo_root = resolve_repo_root(Path.cwd())

    manifest = resolve_input_path(args.manifest, repo_root)
    denylist = resolve_input_path(args.denylist, repo_root)
    report = args.report if args.report.is_absolute() else (Path.cwd() / args.report).resolve()

    allowlist_rules = read_yaml_list(manifest, 'allowlist')
    denylist_rules = read_yaml_list(denylist, 'denylist')

    files = collect_git_files(repo_root, include_untracked=args.include_untracked)
    decisions = [classify_path(path, allowlist=allowlist_rules, denylist=denylist_rules) for path in files]
    status_counts, block_list, ambiguous_list = summarize_decisions(decisions)

    report_text = build_markdown_report(
        decisions=decisions,
        manifest_path=manifest,
        denylist_path=denylist,
        include_untracked=args.include_untracked,
        status_counts=status_counts,
        block_list=block_list,
        ambiguous_list=ambiguous_list,
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(report_text, encoding='utf-8')

    if args.summary_json:
        summary_json = args.summary_json if args.summary_json.is_absolute() else (Path.cwd() / args.summary_json).resolve()
        _write_summary_json(
            output_path=summary_json,
            status_counts=status_counts,
            blocked_paths=[decision.path for decision in block_list],
            ambiguous_paths=[decision.path for decision in ambiguous_list],
        )

    summary = ' | '.join(f'{status}: {status_counts[status]}' for status in STATUS_ORDER)
    print(f'Report written to: {report}')
    if args.summary_json:
        print(f'Summary JSON written to: {summary_json}')
    print(f'Total: {len(decisions)} | {summary}')

    if args.strict and (block_list or ambiguous_list):
        return 1
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
