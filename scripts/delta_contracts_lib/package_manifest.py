from __future__ import annotations

from typing import Any

from migration_sync_lib import ensure_safe_relative

from .common import expect_bool, expect_dict, expect_int, expect_non_empty_str
from .state_manifest import validate_state_migration_manifest


def validate_package_manifest(
    payload: object,
    *,
    context: str = 'state_package_manifest',
) -> dict[str, Any]:
    """Validate exported package manifest schema."""

    manifest = expect_dict(payload, context=context)

    required_keys = {
        'package_version',
        'manifest_version',
        'package_name',
        'created_at',
        'source_repo',
        'source_commit',
        'dry_run_preview',
        'entry_count',
        'entries',
    }
    missing = sorted(required_keys - set(manifest))
    if missing:
        raise ValueError(f'{context} missing required keys: {", ".join(missing)}')

    expect_int(manifest.get('package_version'), context=f'{context}.package_version', min_value=1)
    expect_int(manifest.get('manifest_version'), context=f'{context}.manifest_version', min_value=1)
    expect_non_empty_str(manifest.get('package_name'), context=f'{context}.package_name')
    expect_non_empty_str(manifest.get('created_at'), context=f'{context}.created_at')
    expect_non_empty_str(manifest.get('source_repo'), context=f'{context}.source_repo')

    source_commit = manifest.get('source_commit')
    if not isinstance(source_commit, str):
        raise ValueError(f'{context}.source_commit must be a string')

    expect_bool(manifest.get('dry_run_preview'), context=f'{context}.dry_run_preview')

    entries = manifest.get('entries')
    if not isinstance(entries, list):
        raise ValueError(f'{context}.entries must be a list')

    expected_entry_count = expect_int(
        manifest.get('entry_count'),
        context=f'{context}.entry_count',
        min_value=0,
    )

    source_manifest = manifest.get('source_manifest')
    if source_manifest is not None:
        if not isinstance(source_manifest, dict):
            raise ValueError(f'{context}.source_manifest must be an object')
        validate_state_migration_manifest(
            source_manifest,
            context=f'{context}.source_manifest',
        )

    if expected_entry_count != len(entries):
        raise ValueError(
            f'{context}.entry_count mismatch: expected {expected_entry_count}, found {len(entries)}',
        )

    seen_paths: set[str] = set()

    for index, entry in enumerate(entries):
        entry_context = f'{context}.entries[{index}]'
        entry_dict = expect_dict(entry, context=entry_context)

        rel_path = expect_non_empty_str(entry_dict.get('path'), context=f'{entry_context}.path')
        try:
            ensure_safe_relative(rel_path)
        except ValueError as exc:
            raise ValueError(f'{entry_context}.path invalid: {exc}') from exc

        digest = expect_non_empty_str(entry_dict.get('sha256'), context=f'{entry_context}.sha256')
        if len(digest) != 64 or any(ch not in '0123456789abcdef' for ch in digest.lower()):
            raise ValueError(f'{entry_context}.sha256 must be a 64-char hex string')

        expect_int(entry_dict.get('size_bytes'), context=f'{entry_context}.size_bytes', min_value=0)
        if rel_path in seen_paths:
            raise ValueError(f'{entry_context}.path duplicate: {rel_path}')
        seen_paths.add(rel_path)

    return manifest
