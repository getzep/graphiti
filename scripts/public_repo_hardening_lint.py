#!/usr/bin/env python3
"""Hardening lint for public repo safety rails.

Checks:
1) No legacy hard-path dependencies (`projects/graphiti`, `clawd-graphiti`) in
   active runtime/config/docs/workflow surfaces.
2) No private workflow identifiers accidentally committed to public files.
3) No private workflow pack files by legacy naming convention (`workflows/vc_*.pack.yaml`).
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path


FORBIDDEN_SUBSTRINGS = (
    'projects/graphiti',
    'projects/clawd-graphiti',
    'clawd-graphiti',
)

PRIVATE_TOKENS = (
    'vc_memo_drafting',
    'vc_deal_brief',
    'vc_diligence_questions',
    'vc_ic_prep',
)

SCAN_GLOBS = (
    'scripts/**/*.py',
    'config/**/*.json',
    'config/**/*.yaml',
    'config/**/*.yml',
    'docs/**/*.md',
    'workflows/**/*.pack.yaml',
    '.github/workflows/*.yml',
    '.github/workflows/*.yaml',
)

# Historical records may mention legacy repo names; do not gate on them.
EXCLUDE_GLOBS = (
    'reports/**',
    'prd/**',
)


def git_ls_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ['git', '-C', str(repo_root), 'ls-files'],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def should_exclude(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in EXCLUDE_GLOBS)


def should_scan(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in SCAN_GLOBS)


def main() -> int:
    parser = argparse.ArgumentParser(description='Lint public repo hardening rails.')
    parser.add_argument('--repo', type=Path, default=Path('.'))
    args = parser.parse_args()

    repo_root = args.repo.resolve()
    files = [p for p in git_ls_files(repo_root) if should_scan(p) and not should_exclude(p)]

    issues: list[str] = []

    # path-based checks first
    forbidden_pack_paths = [p for p in files if fnmatch.fnmatch(p, 'workflows/vc_*.pack.yaml')]
    if forbidden_pack_paths:
        issues.append(
            'forbidden private workflow pack paths: ' + ', '.join(f'`{p}`' for p in forbidden_pack_paths)
        )

    # content checks
    for rel in files:
        full = repo_root / rel
        try:
            text = full.read_text(encoding='utf-8', errors='ignore')
        except OSError as exc:
            issues.append(f'cannot read `{rel}`: {exc}')
            continue

        for token in FORBIDDEN_SUBSTRINGS:
            if token in text:
                issues.append(f'legacy path token `{token}` found in `{rel}`')

        for token in PRIVATE_TOKENS:
            if token in text:
                issues.append(f'private token `{token}` found in `{rel}`')

    if issues:
        print('Public repo hardening lint: FAIL', file=sys.stderr)
        for issue in issues:
            print(f'- {issue}', file=sys.stderr)
        return 1

    print('Public repo hardening lint: OK')
    print(f'Scanned files: {len(files)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
