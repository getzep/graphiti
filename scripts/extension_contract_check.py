#!/usr/bin/env python3
"""Validate extension manifests for public delta-layer tooling."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from delta_contracts import inspect_extensions
from migration_sync_lib import resolve_repo_root


def _resolve_repo(candidate: Path) -> Path:
    try:
        return resolve_repo_root(candidate)
    except subprocess.CalledProcessError:
        return candidate.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Check extension manifest contracts.')
    parser.add_argument('--repo', type=Path, default=Path('.'), help='Repository root or subdirectory')
    parser.add_argument('--extensions-dir', type=Path, default=Path('extensions'))
    parser.add_argument('--strict', action='store_true', help='Exit non-zero when issues are found')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = _resolve_repo(args.repo.resolve())
    extensions_dir = args.extensions_dir if args.extensions_dir.is_absolute() else (repo_root / args.extensions_dir).resolve()

    report = inspect_extensions(repo_root=repo_root, extensions_dir=extensions_dir)

    if report.issues:
        print('Extension contract issues found:', file=sys.stderr)
        for issue in report.issues:
            print(f'- {issue}', file=sys.stderr)
        return 1 if args.strict else 0

    print(
        'Extension contract check OK '
        f'({len(report.names)} extension(s)): {", ".join(report.names)} '
        f'| extension commands: {len(report.command_registry)}',
    )
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
