#!/usr/bin/env python3
"""Validate that a snapshot artifact can be decrypted, extracted, and verified."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from migration_sync_lib import ensure_within_root, now_utc_iso, resolve_repo_root, resolve_safe_child, sha256_file


def _resolve_repo_path(repo_root: Path, candidate: Path, *, context: str) -> Path:
    absolute = candidate if candidate.is_absolute() else (repo_root / candidate)
    return ensure_within_root(absolute, repo_root, context=context)


def _assert_path_within(parent: Path, child: Path, *, context: str) -> Path:
    return ensure_within_root(child, parent, context=context)


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, mode='r:gz') as tar:
        members = tar.getmembers()
        for member in members:
            member_path = destination / member.name
            _assert_path_within(destination, member_path, context='tar extraction path')

            if member.issym() or member.islnk():
                raise ValueError(f'Unsupported link entry in snapshot archive: {member.name}')
            if member.isdev() or member.isfifo():
                raise ValueError(f'Unsupported special file entry in snapshot archive: {member.name}')

        for member in members:
            tar.extract(member, path=destination)


def _decrypt_archive(encrypted_archive: Path, plain_archive: Path, passphrase_env: str) -> None:
    if not os.environ.get(passphrase_env):
        raise ValueError(
            f'Missing required env var `{passphrase_env}` for encrypted snapshot restore test.',
        )

    subprocess.run(
        [
            'openssl',
            'enc',
            '-d',
            '-aes-256-cbc',
            '-pbkdf2',
            '-in',
            str(encrypted_archive),
            '-out',
            str(plain_archive),
            '-pass',
            f'env:{passphrase_env}',
        ],
        check=True,
    )


def _validate_manifest(payload: dict[str, object], manifest_path: Path) -> tuple[str, bool, str, list[dict[str, object]]]:
    archive_path_value = payload.get('archive_path')
    archive_sha256_value = payload.get('archive_sha256')
    entries_value = payload.get('entries')

    if not isinstance(archive_path_value, str) or not archive_path_value.strip():
        raise ValueError(f'Manifest missing `archive_path`: {manifest_path}')
    if not isinstance(archive_sha256_value, str) or not archive_sha256_value.strip():
        raise ValueError(f'Manifest missing `archive_sha256`: {manifest_path}')
    if not isinstance(entries_value, list):
        raise ValueError(f'Manifest missing `entries` list: {manifest_path}')

    validated_entries: list[dict[str, object]] = []
    for entry in entries_value:
        if not isinstance(entry, dict):
            raise ValueError(f'Manifest entry must be an object: {entry}')
        if not isinstance(entry.get('path'), str):
            raise ValueError(f'Manifest entry missing string path: {entry}')
        if not isinstance(entry.get('sha256'), str):
            raise ValueError(f'Manifest entry missing string sha256: {entry}')
        if not isinstance(entry.get('size_bytes'), int):
            raise ValueError(f'Manifest entry missing integer size_bytes: {entry}')
        validated_entries.append(entry)

    return archive_path_value, bool(payload.get('archive_encrypted')), archive_sha256_value, validated_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Restore-test a snapshot archive using manifest checks.')
    parser.add_argument('--repo', type=Path, default=Path('.'), help='Repository root or subdirectory')
    parser.add_argument(
        '--snapshot-dir',
        type=Path,
        default=Path('backup/snapshots'),
        help='Directory where snapshot archives live',
    )
    parser.add_argument(
        '--manifest',
        type=Path,
        default=Path('backup/snapshots/latest-manifest.json'),
        help='Snapshot manifest path to validate and restore',
    )
    parser.add_argument(
        '--restore-root',
        type=Path,
        default=Path('backup/restore-test'),
        help='Directory for temporary restore-test extraction',
    )
    parser.add_argument(
        '--passphrase-env',
        default='GRAPHITI_SNAPSHOT_PASSPHRASE',
        help='Env var containing snapshot encryption passphrase',
    )
    parser.add_argument('--keep-temp', action='store_true', help='Keep restore-test extraction output')
    parser.add_argument('--dry-run', action='store_true', help='Validate paths + manifest only (no extraction)')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo.resolve())

    snapshot_dir = _resolve_repo_path(repo_root, args.snapshot_dir, context='snapshot directory')
    manifest_path = _resolve_repo_path(repo_root, args.manifest, context='snapshot manifest path')
    restore_root = _resolve_repo_path(repo_root, args.restore_root, context='snapshot restore-test root')

    # Boundary checks above run before any I/O against snapshot_dir.
    if not snapshot_dir.exists() or not snapshot_dir.is_dir():
        raise ValueError(f'Snapshot directory does not exist or is not a directory: {snapshot_dir}')

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict):
        raise ValueError(f'Snapshot manifest must be an object: {manifest_path}')

    archive_rel, archive_encrypted, archive_sha256, entries = _validate_manifest(manifest, manifest_path)
    archive_path = resolve_safe_child(repo_root, archive_rel, context='snapshot archive path from manifest')
    _assert_path_within(snapshot_dir, archive_path, context='snapshot archive path must stay under snapshot_dir')

    actual_archive_sha256 = sha256_file(archive_path)
    if actual_archive_sha256 != archive_sha256:
        raise ValueError(
            f'Archive checksum mismatch: expected {archive_sha256}, got {actual_archive_sha256}',
        )

    if args.dry_run:
        print('DRY RUN restore-test plan:')
        print(f'- repo root: {repo_root}')
        print(f'- snapshot dir: {snapshot_dir}')
        print(f'- manifest: {manifest_path}')
        print(f'- archive: {archive_path}')
        print(f'- archive sha256: {archive_sha256}')
        print(f'- encrypted: {archive_encrypted}')
        print(f'- entries: {len(entries)}')
        return 0

    restore_root.mkdir(parents=True, exist_ok=True)

    temp_restore_dir = Path(
        tempfile.mkdtemp(
            prefix=f'restore-test-{now_utc_iso().replace(":", "").replace("-", "")}-',
            dir=str(restore_root),
        ),
    )

    should_cleanup = not args.keep_temp
    decrypted_archive: Path | None = None

    try:
        archive_for_extract = archive_path
        if archive_encrypted:
            decrypted_archive = temp_restore_dir / 'snapshot.tar.gz'
            _decrypt_archive(archive_path, decrypted_archive, args.passphrase_env)
            archive_for_extract = decrypted_archive

        payload_root = temp_restore_dir / 'payload'
        payload_root.mkdir(parents=True, exist_ok=False)
        _safe_extract_tar(archive_for_extract, payload_root)

        mismatches: list[str] = []
        missing: list[str] = []

        for entry in entries:
            rel = str(entry['path'])
            expected_hash = str(entry['sha256'])
            expected_size = int(entry['size_bytes'])
            extracted = resolve_safe_child(payload_root, rel, context='restore-test extracted file path')

            if not extracted.exists() or not extracted.is_file():
                missing.append(rel)
                continue

            actual_size = extracted.stat().st_size
            if actual_size != expected_size:
                mismatches.append(
                    f'{rel}: size mismatch (expected {expected_size}, got {actual_size})',
                )

            actual_hash = sha256_file(extracted)
            if actual_hash != expected_hash:
                mismatches.append(f'{rel}: checksum mismatch')

        if missing or mismatches:
            print('Snapshot restore-test FAILED:', file=sys.stderr)
            for rel in missing:
                print(f'- missing extracted file: {rel}', file=sys.stderr)
            for issue in mismatches:
                print(f'- {issue}', file=sys.stderr)
            return 1

        print('Snapshot restore-test OK.')
        print(f'Manifest: {manifest_path}')
        print(f'Archive: {archive_path}')
        print(f'Entries validated: {len(entries)}')
        if args.keep_temp:
            print(f'Restored payload kept at: {payload_root}')
        return 0
    finally:
        if should_cleanup and temp_restore_dir.exists():
            shutil.rmtree(temp_restore_dir, ignore_errors=True)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
