#!/usr/bin/env python3
"""Audit repository paths against allowlist/denylist boundary policy."""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ALLOW = 'ALLOW'
BLOCK = 'BLOCK'
AMBIGUOUS = 'AMBIGUOUS'
STATUS_ORDER = (ALLOW, BLOCK, AMBIGUOUS)


@dataclass(frozen=True)
class RuleDecision:
    path: str
    status: str
    reason_code: str
    matched_rule: str | None


def _read_yaml_list(file_path: Path, section_name: str) -> list[str]:
    """Load a simple YAML list under a named top-level section.

    Supports only a small subset used by the policy files:
      section_name:
        - pattern
    """

    if not file_path.exists():
        raise FileNotFoundError(f'Missing file: {file_path}')

    rules: list[str] = []
    in_target_section = False
    found_section = False

    for raw_line in file_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue

        if line.endswith(':') and not line.startswith('-'):
            section = line[:-1].strip()
            in_target_section = section == section_name
            found_section = found_section or in_target_section
            continue

        if not in_target_section or not line.startswith('-'):
            continue

        value = line[1:].split('#', 1)[0].strip().strip("\"'")
        if value:
            rules.append(value.removeprefix('./'))

    if not found_section:
        raise ValueError(f"Manifest missing required section '{section_name}': {file_path}")

    return rules


def _run_git_lines(repo_root: Path, *args: str) -> set[str]:
    """Run a git command and return non-empty output lines as a set."""

    result = subprocess.run(
        ['git', '-C', str(repo_root), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _resolve_repo_root(cwd: Path) -> Path:
    """Resolve git repository root for the given working directory."""

    result = subprocess.run(
        ['git', '-C', str(cwd), 'rev-parse', '--show-toplevel'],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def _resolve_input_path(path: Path, repo_root: Path) -> Path:
    """Resolve input paths from cwd first, then repo root."""

    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (repo_root / path).resolve()


def _collect_git_files(repo_root: Path, include_untracked: bool) -> list[str]:
    """Collect tracked files from git and optionally append untracked files."""

    files = _run_git_lines(repo_root, 'ls-files')

    if include_untracked:
        files |= _run_git_lines(repo_root, 'ls-files', '--others', '--exclude-standard')

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


def _summarize_decisions(
    decisions: list[RuleDecision],
) -> tuple[dict[str, int], list[RuleDecision], list[RuleDecision]]:
    """Return status counts plus blocked and ambiguous path decisions."""

    counts = {status: 0 for status in STATUS_ORDER}
    blocked: list[RuleDecision] = []
    ambiguous: list[RuleDecision] = []

    for decision in decisions:
        counts[decision.status] += 1
        if decision.status == BLOCK:
            blocked.append(decision)
        elif decision.status == AMBIGUOUS:
            ambiguous.append(decision)

    return counts, blocked, ambiguous


def _build_report(
    decisions: list[RuleDecision],
    manifest_path: Path,
    denylist_path: Path,
    include_untracked: bool,
    status_counts: dict[str, int],
    block_list: list[RuleDecision],
    ambiguous_list: list[RuleDecision],
) -> str:
    """Build a markdown report with summary counts and remediation hints."""

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
        *[f'| {status} | {status_counts[status]} |' for status in STATUS_ORDER],
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
        help='Include untracked files (git ls-files --others --exclude-standard)',
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
    repo_root = _resolve_repo_root(Path.cwd())

    manifest = _resolve_input_path(args.manifest, repo_root)
    denylist = _resolve_input_path(args.denylist, repo_root)
    report = args.report if args.report.is_absolute() else (Path.cwd() / args.report).resolve()

    allowlist_rules = _read_yaml_list(manifest, 'allowlist')
    denylist_rules = _read_yaml_list(denylist, 'denylist')

    files = _collect_git_files(repo_root, include_untracked=args.include_untracked)
    decisions = [_classify(path, allowlist=allowlist_rules, denylist=denylist_rules) for path in files]
    status_counts, block_list, ambiguous_list = _summarize_decisions(decisions)

    report_text = _build_report(
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

    summary = ' | '.join(f'{status}: {status_counts[status]}' for status in STATUS_ORDER)
    print(f'Report written to: {report}')
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
