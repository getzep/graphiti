#!/usr/bin/env python3
"""Shared helpers for public migration/sync tooling."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import shutil
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_utc_iso() -> str:
    """Return a compact UTC timestamp for manifests/reports."""

    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def resolve_repo_root(cwd: Path) -> Path:
    """Resolve git repository root for a working directory."""

    result = subprocess.run(
        ['git', '-C', str(cwd), 'rev-parse', '--show-toplevel'],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def run_git(repo_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a git command in the repository root."""

    return subprocess.run(
        ['git', '-C', str(repo_root), *args],
        capture_output=True,
        text=True,
        check=check,
    )


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON file with strict object expectation."""

    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Expected JSON object in {path}')
    return payload


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    """Write pretty JSON to disk with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'{json.dumps(payload, indent=2, sort_keys=True)}\n', encoding='utf-8')


def sha256_file(path: Path) -> str:
    """Compute SHA-256 digest for a file path."""

    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_within_root(path: Path, root: Path, *, context: str) -> Path:
    """Resolve path and ensure it remains within root."""

    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f'{context} escapes root `{resolved_root}`: {resolved_path}',
        ) from exc
    return resolved_path


def repo_relative(path: Path, repo_root: Path) -> str:
    """Return a POSIX-style relative path from repo root."""

    safe_path = ensure_within_root(path, repo_root, context='repo-relative path')
    return safe_path.relative_to(repo_root.resolve()).as_posix()


def ensure_safe_relative(rel_path: str) -> Path:
    """Validate migration entry paths are relative and traversal-safe."""

    if not rel_path.strip():
        raise ValueError('Unsafe package entry path: empty path')

    path = Path(rel_path)
    if path.is_absolute() or '..' in path.parts or '.' in path.parts:
        raise ValueError(f'Unsafe package entry path: {rel_path}')
    return path


def resolve_safe_child(root: Path, rel_path: str, *, context: str) -> Path:
    """Resolve a relative child path and assert it stays under root."""

    safe_rel = ensure_safe_relative(rel_path)
    return ensure_within_root(root / safe_rel, root, context=context)


def _matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def collect_manifest_files(
    repo_root: Path,
    required_files: list[str],
    optional_globs: list[str],
    exclude_globs: list[str],
) -> list[Path]:
    """Collect files declared by migration manifest rules."""

    selected: dict[str, Path] = {}

    for rel in required_files:
        candidate = resolve_safe_child(repo_root, rel, context='required manifest file')
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f'Required manifest file missing: {rel}')
        selected[repo_relative(candidate, repo_root)] = candidate

    for pattern in optional_globs:
        for candidate in repo_root.glob(pattern):
            if not candidate.is_file():
                continue
            safe_candidate = ensure_within_root(
                candidate,
                repo_root,
                context=f'optional glob match for `{pattern}`',
            )
            rel = repo_relative(safe_candidate, repo_root)
            selected[rel] = safe_candidate

    filtered = [
        candidate
        for rel, candidate in selected.items()
        if not _matches_any(rel, exclude_globs)
    ]

    return sorted(filtered, key=lambda path: repo_relative(path, repo_root))


def collect_file_entries(files: list[Path], repo_root: Path) -> list[dict[str, Any]]:
    """Build normalized metadata entries for a list of files."""

    entries: list[dict[str, Any]] = []
    for path in files:
        safe_path = ensure_within_root(path, repo_root, context='manifest-selected file')
        rel = repo_relative(safe_path, repo_root)
        entries.append(
            {
                'path': rel,
                'sha256': sha256_file(safe_path),
                'size_bytes': safe_path.stat().st_size,
            },
        )
    return entries


def evaluate_payload_entries(
    entries: list[dict[str, Any]],
    payload_root: Path,
    *,
    context: str,
) -> tuple[list[tuple[str, Path]], list[str], list[str]]:
    """Evaluate payload entries for presence and integrity.

    Returns:
      - planned entries as ``(relative_path, payload_source_path)``
      - missing payload relative paths
      - integrity issues (size/checksum mismatches)
    """

    planned_entries: list[tuple[str, Path]] = []
    missing_payload: list[str] = []
    integrity_errors: list[str] = []

    for entry in entries:
        rel = str(entry['path'])
        expected_hash = str(entry['sha256'])
        expected_size = int(entry['size_bytes'])

        src = resolve_safe_child(payload_root, rel, context=context)
        planned_entries.append((rel, src))

        if not src.exists() or not src.is_file():
            missing_payload.append(rel)
            continue

        actual_size = src.stat().st_size
        if actual_size != expected_size:
            integrity_errors.append(
                f'{rel}: size mismatch (expected {expected_size}, got {actual_size})',
            )

        actual_hash = sha256_file(src)
        if actual_hash != expected_hash:
            integrity_errors.append(f'{rel}: checksum mismatch')

    return planned_entries, missing_payload, integrity_errors


def copy_entry(src_root: Path, dst_root: Path, rel_path: str) -> None:
    """Copy a relative file path from src_root to dst_root preserving structure."""

    src = resolve_safe_child(src_root, rel_path, context='copy source')
    dst = resolve_safe_child(dst_root, rel_path, context='copy destination')
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f'Missing source file for copy: {rel_path}')

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
