#!/usr/bin/env python3
"""Create encrypted repository snapshots for pre-cutover backup gates."""

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

from migration_sync_lib import ensure_safe_relative, ensure_within_root, now_utc_iso, repo_relative, resolve_repo_root, sha256_file

DEFAULT_INCLUDE_PATHS = [
    'state',
    'exports',
    'reports/private',
    'logs',
]


def _resolve_repo_path(repo_root: Path, candidate: Path, *, context: str) -> Path:
    absolute = candidate if candidate.is_absolute() else (repo_root / candidate)
    return ensure_within_root(absolute, repo_root, context=context)


def _collect_files(repo_root: Path, include_paths: list[str]) -> list[Path]:
    files: list[Path] = []
    missing: list[str] = []

    for raw in include_paths:
        safe_rel = ensure_safe_relative(raw)
        candidate = ensure_within_root(
            repo_root / safe_rel,
            repo_root,
            context=f'snapshot include path `{raw}`',
        )

        if not candidate.exists():
            missing.append(raw)
            continue

        if candidate.is_file():
            files.append(candidate)
            continue

        if candidate.is_dir():
            for child in sorted(candidate.rglob('*')):
                if not child.is_file():
                    continue
                files.append(
                    ensure_within_root(
                        child,
                        repo_root,
                        context=f'snapshot file under `{raw}`',
                    ),
                )
            continue

        missing.append(raw)

    if missing:
        joined = ', '.join(sorted(set(missing)))
        raise ValueError(f'Snapshot include paths are missing: {joined}')

    deduped = sorted({path.resolve() for path in files})
    if not deduped:
        raise ValueError('Snapshot selection is empty after scanning include paths.')

    return deduped


def _normalize_tarinfo(tar_info: tarfile.TarInfo) -> tarfile.TarInfo:
    tar_info.uid = 0
    tar_info.gid = 0
    tar_info.uname = 'root'
    tar_info.gname = 'root'
    tar_info.mtime = 0
    return tar_info


def _write_archive(repo_root: Path, files: list[Path], archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, mode='w:gz', format=tarfile.PAX_FORMAT, compresslevel=9) as tar:
        for file_path in files:
            rel = repo_relative(file_path, repo_root)
            tar.add(file_path, arcname=rel, recursive=False, filter=_normalize_tarinfo)


def _encrypt_archive(plain_archive: Path, encrypted_archive: Path, passphrase_env: str) -> None:
    if not os.environ.get(passphrase_env):
        raise ValueError(
            f'Missing required env var `{passphrase_env}` for encrypted snapshot output.',
        )

    subprocess.run(
        [
            'openssl',
            'enc',
            '-aes-256-cbc',
            '-pbkdf2',
            '-salt',
            '-in',
            str(plain_archive),
            '-out',
            str(encrypted_archive),
            '-pass',
            f'env:{passphrase_env}',
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create deterministic encrypted snapshot artifact + manifest.')
    parser.add_argument('--repo', type=Path, default=Path('.'), help='Repository root or subdirectory')
    parser.add_argument(
        '--snapshot-dir',
        type=Path,
        default=Path('backup/snapshots'),
        help='Directory for snapshot archives',
    )
    parser.add_argument(
        '--manifest',
        type=Path,
        default=Path('backup/snapshots/latest-manifest.json'),
        help='Manifest output path',
    )
    parser.add_argument(
        '--name',
        default='',
        help='Snapshot name (default: timestamped)',
    )
    parser.add_argument(
        '--include',
        action='append',
        default=[],
        help='Relative path under repo root to include (repeatable)',
    )
    parser.add_argument(
        '--encrypt',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Encrypt archive with openssl AES-256-CBC (default: true)',
    )
    parser.add_argument(
        '--passphrase-env',
        default='GRAPHITI_SNAPSHOT_PASSPHRASE',
        help='Env var containing snapshot encryption passphrase',
    )
    parser.add_argument('--dry-run', action='store_true', help='Print selected files/paths without writing output')
    parser.add_argument('--force', action='store_true', help='Overwrite existing archive/manifest outputs')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo.resolve())

    snapshot_dir = _resolve_repo_path(repo_root, args.snapshot_dir, context='snapshot directory')
    manifest_path = _resolve_repo_path(repo_root, args.manifest, context='snapshot manifest path')

    include_paths = args.include or list(DEFAULT_INCLUDE_PATHS)
    files = _collect_files(repo_root, include_paths)

    snapshot_name = args.name.strip() or f'precutover-{now_utc_iso().replace(":", "").replace("-", "")}'
    suffix = '.tar.gz.enc' if args.encrypt else '.tar.gz'
    archive_path = ensure_within_root(
        snapshot_dir / f'{snapshot_name}{suffix}',
        repo_root,
        context='snapshot archive path',
    )

    if (archive_path.exists() or manifest_path.exists()) and not args.force:
        raise ValueError(
            'Snapshot outputs already exist. Use --force to overwrite:\n'
            f'- archive: {archive_path}\n'
            f'- manifest: {manifest_path}',
        )

    entries = [
        {
            'path': repo_relative(path, repo_root),
            'size_bytes': path.stat().st_size,
            'sha256': sha256_file(path),
        }
        for path in files
    ]

    manifest = {
        'snapshot_version': 1,
        'created_at': now_utc_iso(),
        'snapshot_name': snapshot_name,
        'archive_path': repo_relative(archive_path, repo_root),
        'archive_encrypted': bool(args.encrypt),
        'entry_count': len(entries),
        'entries': entries,
    }

    if args.dry_run:
        print('DRY RUN snapshot plan:')
        print(f'- repo root: {repo_root}')
        print(f'- snapshot dir: {snapshot_dir}')
        print(f'- manifest path: {manifest_path}')
        print(f'- archive path: {archive_path}')
        print(f'- encrypted: {args.encrypt}')
        print(f'- selected files: {len(entries)}')
        for entry in entries:
            print(f"  - {entry['path']}")
        return 0

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='snapshot-create-', dir=str(snapshot_dir)) as temp_dir:
        temp_archive = Path(temp_dir) / f'{snapshot_name}.tar.gz'
        _write_archive(repo_root, files, temp_archive)

        if args.encrypt:
            _encrypt_archive(temp_archive, archive_path, args.passphrase_env)
        else:
            shutil.copy2(temp_archive, archive_path)

    manifest['archive_sha256'] = sha256_file(archive_path)
    manifest_path.write_text(f'{json.dumps(manifest, indent=2, sort_keys=True)}\n', encoding='utf-8')

    print(f'Snapshot archive written: {archive_path}')
    print(f'Snapshot manifest written: {manifest_path}')
    print(f'Included files: {len(entries)}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
