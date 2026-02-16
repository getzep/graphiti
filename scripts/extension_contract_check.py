#!/usr/bin/env python3
"""Validate extension manifests against the public extension contract."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from migration_sync_lib import resolve_repo_root

LoadExtensionsFn = Callable[..., Any]


def _import_load_extensions(script_repo_root: Path) -> LoadExtensionsFn:
    """Import ``extensions.loader.load_extensions`` from this repository.

    The checker can run from arbitrary working directories (for example tests
    that execute against temporary repos). Resolve imports from the script's
    repository root instead of relying on cwd.
    """

    root = str(script_repo_root)
    if root not in sys.path:
        sys.path.insert(0, root)

    try:
        from extensions.loader import load_extensions
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f'Unable to import `extensions.loader` from repo root: {script_repo_root}',
        ) from exc

    return load_extensions


def _resolve_repo(candidate: Path) -> Path:
    try:
        return resolve_repo_root(candidate)
    except subprocess.CalledProcessError:
        return candidate.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Check extension contract compatibility.')
    parser.add_argument('--repo', type=Path, default=Path('.'), help='Repository root or subdirectory')
    parser.add_argument('--extensions-dir', type=Path, default=Path('extensions'))
    parser.add_argument('--strict', action='store_true', help='Exit non-zero when incompatibilities are found')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_repo_root = Path(__file__).resolve().parents[1]
    load_extensions = _import_load_extensions(script_repo_root)

    repo_root = _resolve_repo(args.repo.resolve())
    extensions_dir = args.extensions_dir if args.extensions_dir.is_absolute() else (repo_root / args.extensions_dir)

    report = load_extensions(repo_root=repo_root, extensions_dir=extensions_dir)

    if report.diagnostics:
        print('Extension contract diagnostics:', file=sys.stderr)
        for diagnostic in report.diagnostics:
            print(f'- {diagnostic.render()}', file=sys.stderr)

    if report.errors:
        if args.strict:
            print(
                'Extension contract check failed in strict mode. '
                'Fix incompatibilities listed above.',
                file=sys.stderr,
            )
            return 1

        print(
            'Extension contract check completed with incompatibilities '
            '(strict mode disabled).',
            file=sys.stderr,
        )
        return 0

    extension_names = report.extension_names
    print(
        'Extension contract check OK '
        f'({len(extension_names)} extension(s)): '
        f'{", ".join(extension_names) if extension_names else "<none>"} '
        f'| extension commands: {len(report.command_registry)}',
    )
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
