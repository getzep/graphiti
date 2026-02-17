#!/usr/bin/env python3
"""Shared policy/audit helpers for public boundary enforcement."""

from __future__ import annotations

import fnmatch
import subprocess
from dataclasses import dataclass
from pathlib import Path

from migration_sync_lib import resolve_repo_root as _resolve_repo_root

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


def read_yaml_list(file_path: Path, section_name: str) -> list[str]:
    """Load a simple YAML list under a named top-level section.

    Supports only a small subset used by the boundary policy files:
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


def run_git_lines(repo_root: Path, *args: str) -> set[str]:
    """Run a git command and return non-empty output lines as a set."""

    result = subprocess.run(
        ['git', '-C', str(repo_root), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def resolve_repo_root(cwd: Path) -> Path:
    """Resolve git repository root for the given working directory."""

    return _resolve_repo_root(cwd)


def resolve_input_path(path: Path, repo_root: Path) -> Path:
    """Resolve input paths from cwd first, then repo root."""

    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (repo_root / path).resolve()


def collect_git_files(repo_root: Path, include_untracked: bool) -> list[str]:
    """Collect tracked files from git and optionally append untracked files."""

    files = run_git_lines(repo_root, 'ls-files')

    if include_untracked:
        files |= run_git_lines(repo_root, 'ls-files', '--others', '--exclude-standard')

    return sorted(files)


def classify_path(path: str, allowlist: list[str], denylist: list[str]) -> RuleDecision:
    """Classify a repo path as ALLOW/BLOCK/AMBIGUOUS.

    Precedence: explicit allowlist match overrides denylist (so safe files
    like ``.env.example`` can be allowed despite a broad ``.env*`` deny).
    If neither list matches, the path is AMBIGUOUS.
    """
    # Check allowlist first — explicit allow overrides general deny.
    for allow_rule in allowlist:
        if fnmatch.fnmatch(path, allow_rule):
            return RuleDecision(path, ALLOW, 'ALLOWLIST_MATCH', allow_rule)

    for deny_rule in denylist:
        if fnmatch.fnmatch(path, deny_rule):
            return RuleDecision(path, BLOCK, 'DENYLIST_MATCH', deny_rule)

    return RuleDecision(path, AMBIGUOUS, 'NO_MATCH', None)


def summarize_decisions(
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


def build_markdown_report(
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
