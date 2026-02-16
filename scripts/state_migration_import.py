#!/usr/bin/env python3
"""Import migration package payload into a target repository tree."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from delta_contracts import validate_package_manifest
from migration_sync_lib import (
    evaluate_payload_entries,
    file_has_expected_content,
    load_json,
    resolve_safe_child,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Import migration package payload into target directory.')
    parser.add_argument('--in', dest='package', type=Path, required=True, help='Package directory path')
    parser.add_argument('--target', type=Path, default=Path('.'), help='Target repository root')
    parser.add_argument('--dry-run', action='store_true', help='Show planned writes without mutating target')
    parser.add_argument('--allow-overwrite', action='store_true', help='Allow overwriting existing files')
    parser.add_argument(
        '--atomic',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Apply writes with rollback on failure (default: true)',
    )
    return parser.parse_args()


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _apply_import_atomic(planned_writes: list[tuple[str, Path, Path]], target_root: Path) -> None:
    rollback_root = Path(
        tempfile.mkdtemp(
            prefix='.delta-import-rollback-',
            dir=str(target_root),
        ),
    )

    backup_pairs: list[tuple[str, Path]] = []
    created_paths: list[Path] = []

    try:
        for rel, src, dst in planned_writes:
            if dst.exists():
                backup_path = resolve_safe_child(
                    rollback_root,
                    rel,
                    context='atomic rollback backup path',
                )
                _copy_file(dst, backup_path)
                backup_pairs.append((rel, dst))
            else:
                created_paths.append(dst)

            _copy_file(src, dst)
    except Exception as import_exc:
        for created in reversed(created_paths):
            if created.exists():
                created.unlink()

        for rel, dst in backup_pairs:
            restore_src = resolve_safe_child(
                rollback_root,
                rel,
                context='atomic rollback restore source',
            )
            if not restore_src.exists() or not restore_src.is_file():
                raise ValueError(
                    f'Atomic rollback backup missing or invalid for {rel}: {restore_src}',
                ) from import_exc
            _copy_file(restore_src, dst)

        raise
    finally:
        shutil.rmtree(rollback_root, ignore_errors=True)


def _apply_import_non_atomic(planned_writes: list[tuple[str, Path, Path]]) -> None:
    for _, src, dst in planned_writes:
        _copy_file(src, dst)


def main() -> int:
    args = parse_args()
    package_root = args.package if args.package.is_absolute() else (Path.cwd() / args.package).resolve()
    target_root = args.target if args.target.is_absolute() else (Path.cwd() / args.target).resolve()

    if target_root.exists() and not target_root.is_dir():
        raise ValueError(f'Import target must be a directory: {target_root}')
    target_root.mkdir(parents=True, exist_ok=True)

    manifest = validate_package_manifest(
        load_json(package_root / 'package_manifest.json'),
        context=str(package_root / 'package_manifest.json'),
    )

    entries = manifest.get('entries', [])
    if not isinstance(entries, list):
        raise ValueError('Manifest field `entries` must be a list')

    payload_root = package_root / 'payload'
    dry_run_preview = bool(manifest.get('dry_run_preview'))

    planned_entries, missing_payload, integrity_errors = evaluate_payload_entries(
        entries=entries,
        payload_root=payload_root,
        context='migration payload entry',
    )

    planned_writes: list[tuple[str, Path, Path]] = []
    skipped_matches: list[tuple[str, Path, Path]] = []
    destination_conflicts: list[str] = []
    destination_path_conflicts: list[str] = []

    for rel, src, expected_hash, expected_size in planned_entries:
        dst = resolve_safe_child(target_root, rel, context='migration import target entry')
        if dst.exists() and not dst.is_file():
            destination_path_conflicts.append(
                f'{rel}: destination is not a file: {dst}',
            )
            continue

        if dst.parent.exists() and not dst.parent.is_dir():
            destination_path_conflicts.append(
                f'{rel}: destination parent is not a directory: {dst.parent}',
            )
            continue

        if dst.exists() and file_has_expected_content(
            dst,
            expected_sha256=expected_hash,
            expected_size_bytes=expected_size,
        ):
            skipped_matches.append((rel, src, dst))
            continue

        if dst.exists() and not args.allow_overwrite:
            destination_conflicts.append(rel)
            continue

        planned_writes.append((rel, src, dst))

    if destination_path_conflicts and not args.dry_run:
        print('Import blocked: invalid destination entries.', file=sys.stderr)
        for detail in destination_path_conflicts:
            print(f'- {detail}', file=sys.stderr)
        return 1

    if destination_conflicts and not args.dry_run:
        print('Import blocked: existing files would be overwritten.', file=sys.stderr)
        for rel in destination_conflicts:
            print(f'- {rel}', file=sys.stderr)
        print('Use --allow-overwrite to apply import anyway.', file=sys.stderr)
        return 1

    if missing_payload and not args.dry_run and not dry_run_preview:
        print('Import blocked: package payload is incomplete.', file=sys.stderr)
        for rel in missing_payload:
            print(f'- {rel}', file=sys.stderr)
        return 1

    if integrity_errors and not args.dry_run:
        print('Import blocked: payload integrity check failed.', file=sys.stderr)
        for issue in integrity_errors:
            print(f'- {issue}', file=sys.stderr)
        return 1

    if args.dry_run:
        print(
            f'DRY RUN import plan: write/overwrite={len(planned_writes)}, '
            f'skip-same={len(skipped_matches)}, '
            f'blocked-conflicts={len(destination_conflicts)}, '
            f'blocked-paths={len(destination_path_conflicts)}'
        )
        for rel, _, _ in skipped_matches:
            print(f'- skip identical: {rel}')
        for rel, src, dst in planned_writes:
            note = ' (payload missing in dry-run preview)' if not src.exists() else ''
            print(f'- write: {rel} -> {dst}{note}')

        has_blocking_conflicts = bool(destination_conflicts or destination_path_conflicts)
        if destination_conflicts:
            for rel in destination_conflicts:
                print(f'- would overwrite: {rel}', file=sys.stderr)
        if destination_path_conflicts:
            for detail in destination_path_conflicts:
                print(f'- blocked path: {detail}', file=sys.stderr)

        if integrity_errors:
            print('Dry-run integrity warnings:')
            for issue in integrity_errors:
                print(f'- {issue}')

        if has_blocking_conflicts:
            print(
                'Dry-run detected blocking destination conflicts. '
                'Resolve conflicts or use --allow-overwrite before import.',
                file=sys.stderr,
            )
            return 1
        return 0

    if dry_run_preview:
        raise ValueError(
            'Cannot execute non-dry-run import from dry-run preview package. '
            'Re-export without --dry-run to include payload files.',
        )

    if not planned_writes:
        print(
            f'No changes: all {len(skipped_matches)} entries already match target state.',
        )
        return 0

    if args.atomic:
        _apply_import_atomic(planned_writes, target_root)
    else:
        _apply_import_non_atomic(planned_writes)

    print(
        f'Imported {len(planned_writes)} files into {target_root} '
        f'(atomic={args.atomic})',
    )
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
