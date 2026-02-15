#!/usr/bin/env python3
"""Audit repository paths against allowlist/denylist boundary policy."""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


Status = str
ALLOW = 'ALLOW'
BLOCK = 'BLOCK'
AMBIGUOUS = 'AMBIGUOUS'


@dataclass(frozen=True)
class RuleDecision:
    path: str
    status: Status
    reason_code: str
    matched_rule: str | None


def _read_yaml_list(file_path: Path, section_name: str) -> list[str]:
    """Load a simple YAML list under a named top-level section.

    Supports only a small subset used by the policy files:
      section_name:
        - pattern
    """

    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    lines = file_path.read_text(encoding='utf-8').splitlines()
    rules: list[str] = []
    current_section: str | None = None
    section_matcher = re.compile(r'^([A-Za-z0-9_]+):')

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue

        section_match = section_matcher.match(line)
        if section_match:
            current_section = section_match.group(1)
            continue

        if current_section == section_name and raw_line.lstrip().startswith('-'):
            value = raw_line.lstrip()[1:].strip()
            if not value:
                continue
            value = value.strip("\"'")
            rules.append(value.lstrip('./'))
    return rules


def _collect_git_files(repo_root: Path, include_untracked: bool) -> list[str]:
    """Collect tracked files from git and optionally append untracked files."""

    result = subprocess.run(
        ['git', '-C', str(repo_root), 'ls-files'],
        capture_output=True,
        text=True,
        check=True,
    )
    files = {line.strip() for line in result.stdout.splitlines() if line.strip()}

    if include_untracked:
        untracked_result = subprocess.run(
            ['git', '-C', str(repo_root), 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in untracked_result.stdout.splitlines():
            if not line.startswith('?? '):
                continue
            path = line[3:].strip()
            files.add(path)

    return sorted(files)


def _classify(path: str, allowlist: list[str], denylist: list[str]) -> RuleDecision:
    """Classify a repo path as ALLOW/BLOCK/AMBIGUOUS."""

    for deny_rule in denylist:
        if fnmatch.fnmatch(path, deny_rule):
            return RuleDecision(path, BLOCK, 'DENYLIST_MATCH', deny_rule)

    for allow_rule in allowlist:
        if fnmatch.fnmatch(path, allow_rule):
            return RuleDecision(path, ALLOW, 'ALLOWLIST_MATCH', allow_rule)

    return RuleDecision(path, AMBIGUOUS, 'NO_MATCH', None)


def _build_report(
    decisions: list[RuleDecision],
    manifest_path: Path,
    denylist_path: Path,
    include_untracked: bool,
) -> str:
    """Build a markdown report with summary counts and remediation hints."""

    status_counts = {
        ALLOW: 0,
        BLOCK: 0,
        AMBIGUOUS: 0,
    }
    for decision in decisions:
        status_counts[decision.status] += 1

    block_list = [d for d in decisions if d.status == BLOCK]
    ambiguous_list = [d for d in decisions if d.status == AMBIGUOUS]

    lines = [
        '# Public Foundation Boundary Audit',
        '',
        f'- Manifest: `{manifest_path}`',
        f'- Denylist: `{denylist_path}`',
        f'- Include untracked: `{include_untracked}`',
        f'- Total paths evaluated: `{len(decisions)}`',
        '',
        '## Summary',
        '',
        '| Status | Count |',
        '| --- | --- |',
        f'| {ALLOW} | {status_counts[ALLOW]} |',
        f'| {BLOCK} | {status_counts[BLOCK]} |',
        f'| {AMBIGUOUS} | {status_counts[AMBIGUOUS]} |',
        '',
        '## Offending paths',
        '',
        f'- **BLOCK**: {len(block_list)}',
        f'- **AMBIGUOUS**: {len(ambiguous_list)}',
        '',
    ]

    if block_list:
        lines.extend([
            '### BLOCK',
            '',
            '| Path | Rule matched | Reason code | Remediation hint |',
            '| --- | --- | --- | --- |',
        ])
        for decision in block_list:
            path = decision.path
            rule = decision.matched_rule or ''
            lines.append(
                f'| `{path}` | `{rule}` | {decision.reason_code} | '
                'Move sensitive artifacts out of export scope or explicitly harden denylist. |',
            )
        lines.append('')

    if ambiguous_list:
        lines.extend([
            '### AMBIGUOUS',
            '',
            '| Path | Reason code | Remediation hint |',
            '| --- | --- | --- |',
        ])
        for decision in ambiguous_list:
            lines.append(
                f'| `{decision.path}` | {decision.reason_code} | '
                'Add explicit allowlist entry if file should be public; otherwise keep out of export. |',
            )
        lines.append('')

    return '\n'.join(lines).rstrip() + '\n'


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description='Run Public Foundation Boundary audit using allowlist and denylist rules.',
    )
    parser.add_argument('--manifest', required=True, type=Path, help='Path to allowlist YAML file')
    parser.add_argument('--denylist', required=True, type=Path, help='Path to denylist YAML file')
    parser.add_argument('--report', type=Path, required=True, help='Output markdown report path')
    parser.add_argument(
        '--include-untracked',
        action='store_true',
        help='Include untracked files from git status',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Exit with non-zero when BLOCK or AMBIGUOUS paths are detected',
    )
    return parser.parse_args()


def main() -> int:
    """Run the audit and write the markdown report."""

    args = parse_args()
    manifest = args.manifest
    denylist = args.denylist
    report = args.report

    allowlist_rules = _read_yaml_list(manifest, 'allowlist')
    denylist_rules = _read_yaml_list(denylist, 'denylist')

    files = _collect_git_files(Path.cwd(), include_untracked=args.include_untracked)
    decisions = [
        _classify(path, allowlist=allowlist_rules, denylist=denylist_rules)
        for path in files
    ]

    report_text = _build_report(
        decisions=decisions,
        manifest_path=manifest,
        denylist_path=denylist,
        include_untracked=args.include_untracked,
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(report_text, encoding='utf-8')

    print(f'Report written to: {report}')
    print(f'Total: {len(decisions)} | ALLOW: {sum(d.status == ALLOW for d in decisions)} | '
          f'BLOCK: {sum(d.status == BLOCK for d in decisions)} | '
          f'AMBIGUOUS: {sum(d.status == AMBIGUOUS for d in decisions)}')

    if args.strict and any(d.status != ALLOW for d in decisions):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
