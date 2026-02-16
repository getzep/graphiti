from __future__ import annotations

from typing import Any

from migration_sync_lib import ensure_safe_relative

from .common import (
    expect_dict,
    expect_int,
    expect_non_empty_str,
    expect_string_list,
    validate_glob_patterns,
)


def validate_state_migration_manifest(
    payload: object,
    *,
    context: str = 'state_migration_manifest',
) -> dict[str, Any]:
    """Validate state migration manifest schema."""

    manifest = expect_dict(payload, context=context)
    expect_int(manifest.get('version'), context=f'{context}.version', min_value=1)
    expect_non_empty_str(manifest.get('package_name'), context=f'{context}.package_name')

    required_files = expect_string_list(
        manifest.get('required_files'),
        context=f'{context}.required_files',
        allow_empty=False,
        unique=True,
    )
    for index, rel in enumerate(required_files):
        try:
            ensure_safe_relative(rel)
        except ValueError as exc:
            raise ValueError(f'{context}.required_files[{index}] invalid: {exc}') from exc

    optional_globs = expect_string_list(
        manifest.get('optional_globs'),
        context=f'{context}.optional_globs',
        allow_empty=True,
        unique=True,
    )
    validate_glob_patterns(optional_globs, context=f'{context}.optional_globs')

    exclude_globs = expect_string_list(
        manifest.get('exclude_globs'),
        context=f'{context}.exclude_globs',
        allow_empty=True,
        unique=True,
    )
    validate_glob_patterns(exclude_globs, context=f'{context}.exclude_globs')

    return manifest
