#!/usr/bin/env python3
"""Validate migration package integrity and compatibility."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from delta_contracts import validate_package_manifest
from migration_sync_lib import evaluate_payload_entries, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validate migration package structure and checksums.')
    parser.add_argument('--package', type=Path, required=True, help='Package directory path')
    parser.add_argument('--dry-run', action='store_true', help='Validate manifest/schema only (skip payload checks)')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_root = args.package if args.package.is_absolute() else (Path.cwd() / args.package).resolve()
    manifest_path = package_root / 'package_manifest.json'

    manifest = validate_package_manifest(load_json(manifest_path), context=str(manifest_path))
    entries = manifest.get('entries', [])
    if not isinstance(entries, list):
        raise ValueError('Manifest field `entries` must be a list')

    errors: list[str] = []
    dry_run_preview = bool(manifest.get('dry_run_preview'))
    payload_root = package_root / 'payload'

    should_verify_payload = not args.dry_run and not dry_run_preview
    if should_verify_payload:
        _, missing_payload, integrity_errors = evaluate_payload_entries(
            entries=entries,
            payload_root=payload_root,
            context='state migration payload entry',
        )

        errors.extend(f'Missing payload file: {rel}' for rel in missing_payload)
        errors.extend(integrity_errors)

    if errors:
        print('Migration package check FAILED:', file=sys.stderr)
        for issue in errors:
            print(f'- {issue}', file=sys.stderr)
        return 1

    mode = 'manifest-only' if args.dry_run or dry_run_preview else 'full payload'
    print(f'Migration package check OK ({mode}).')
    print(f'Package: {package_root}')
    print(f'Entries: {len(entries)}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
