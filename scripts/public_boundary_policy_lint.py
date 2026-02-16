#!/usr/bin/env python3
"""Lint allowlist/denylist policy files for maintainability hazards."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from public_boundary_policy import read_yaml_list, resolve_input_path, resolve_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Lint public boundary policy files for duplicates and contradictory rules.',
    )
    parser.add_argument('--manifest', required=True, type=Path, help='Path to allowlist YAML file')
    parser.add_argument('--denylist', required=True, type=Path, help='Path to denylist YAML file')
    parser.add_argument(
        '--warn-only',
        action='store_true',
        help='Print issues but always exit zero',
    )
    return parser.parse_args()


def _find_duplicates(rules: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for rule in rules:
        if rule in seen and rule not in duplicates:
            duplicates.append(rule)
        seen.add(rule)
    return duplicates


def main() -> int:
    args = parse_args()
    repo_root = resolve_repo_root(Path.cwd())

    manifest = resolve_input_path(args.manifest, repo_root)
    denylist = resolve_input_path(args.denylist, repo_root)

    allowlist_rules = read_yaml_list(manifest, 'allowlist')
    denylist_rules = read_yaml_list(denylist, 'denylist')

    allow_dupes = _find_duplicates(allowlist_rules)
    deny_dupes = _find_duplicates(denylist_rules)
    contradictory = sorted(set(allowlist_rules) & set(denylist_rules))

    issues: list[str] = []
    if allow_dupes:
        issues.append(
            f'allowlist duplicates ({len(allow_dupes)}): ' + ', '.join(f"`{rule}`" for rule in allow_dupes),
        )
    if deny_dupes:
        issues.append(
            f'denylist duplicates ({len(deny_dupes)}): ' + ', '.join(f"`{rule}`" for rule in deny_dupes),
        )
    if contradictory:
        issues.append(
            'rules present in both allowlist and denylist '
            f'({len(contradictory)}): ' + ', '.join(f"`{rule}`" for rule in contradictory),
        )

    if not issues:
        print('Boundary policy lint: OK (no duplicates or contradictory rules).')
        return 0

    print('Boundary policy lint: issues found:', file=sys.stderr)
    for issue in issues:
        print(f'- {issue}', file=sys.stderr)

    return 0 if args.warn_only else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
