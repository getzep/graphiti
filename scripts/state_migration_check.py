#!/usr/bin/env python3
"""Validate migration package integrity and compatibility."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from delta_contracts import validate_package_manifest, validate_state_migration_manifest
from migration_sync_lib import evaluate_payload_entries, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validate migration package structure and checksums.')
    parser.add_argument('--package', type=Path, required=True, help='Package directory path')
    parser.add_argument('--dry-run', action='store_true', help='Validate manifest/schema only (skip payload checks)')
    parser.add_argument(
        '--target',
        type=Path,
        default=None,
        help='Optional target repo to validate manifest compatibility against',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_root = args.package if args.package.is_absolute() else (Path.cwd() / args.package).resolve()
    target_root = (
        args.target.resolve()
        if isinstance(args.target, Path) and args.target.is_absolute()
        else (Path.cwd() / args.target).resolve() if args.target else None
    )
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

    if target_root is not None:
        target_manifest_path = target_root / 'config' / 'state_migration_manifest.json'
        if not target_manifest_path.exists():
            errors.append(f'Compatibility check failed: missing target manifest at {target_manifest_path}')
        else:
            target_manifest = validate_state_migration_manifest(
                load_json(target_manifest_path),
                context=str(target_manifest_path),
            )

            source_manifest = manifest.get('source_manifest')
            if source_manifest is not None:
                if source_manifest.get('version') != target_manifest.get('version'):
                    errors.append(
                        'Compatibility check failed: manifest version mismatch '
                        f'(source={source_manifest.get("version")} target={target_manifest.get("version")})',
                    )

                source_required = set(source_manifest.get('required_files', []))
                target_required = set(target_manifest.get('required_files', []))
                missing_required = sorted(source_required - target_required)
                if missing_required:
                    errors.append(
                        'Compatibility check failed: target manifest is missing required files: '
                        + ', '.join(missing_required),
                    )

                source_optional = sorted(set(source_manifest.get('optional_globs', [])))
                target_optional = sorted(set(target_manifest.get('optional_globs', [])))
                if source_optional != target_optional:
                    errors.append(
                        'Compatibility check failed: optional_globs mismatch '
                        f'(source={source_optional} target={target_optional})',
                    )

                source_exclude = sorted(set(source_manifest.get('exclude_globs', [])))
                target_exclude = sorted(set(target_manifest.get('exclude_globs', [])))
                if source_exclude != target_exclude:
                    errors.append(
                        'Compatibility check failed: exclude_globs mismatch '
                        f'(source={source_exclude} target={target_exclude})',
                    )

            source_package_name = manifest.get('package_name')
            target_package_name = target_manifest.get('package_name')
            if source_package_name != target_package_name:
                errors.append(
                    'Compatibility check failed: package name mismatch '
                    f'(source={source_package_name} target={target_package_name})',
                )

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
