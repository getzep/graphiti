from __future__ import annotations

from typing import Any

from migration_sync_lib import ensure_safe_relative

from .common import (
    expect_dict,
    expect_int,
    expect_non_empty_str,
    expect_string_list,
    normalize_slug,
    validate_command_part,
)


def _validate_entrypoints(manifest: dict[str, Any], *, context: str) -> None:
    entrypoints = expect_dict(manifest.get('entrypoints'), context=f'{context}.entrypoints')
    if not entrypoints:
        raise ValueError(f'{context}.entrypoints must not be empty')

    for key, value in entrypoints.items():
        expect_non_empty_str(key, context=f'{context}.entrypoints key')
        rel_path = expect_non_empty_str(value, context=f'{context}.entrypoints.{key}')
        try:
            ensure_safe_relative(rel_path)
        except ValueError as exc:
            raise ValueError(f'{context}.entrypoints.{key} invalid: {exc}') from exc


def _validate_command_registry(manifest: dict[str, Any], *, context: str) -> None:
    commands = manifest.get('commands')
    if commands is None:
        return

    commands_dict = expect_dict(commands, context=f'{context}.commands')
    if not commands_dict:
        raise ValueError(f'{context}.commands must not be empty when provided')

    command_contract = expect_dict(
        manifest.get('command_contract'),
        context=f'{context}.command_contract',
    )
    version = expect_int(command_contract.get('version'), context=f'{context}.command_contract.version', min_value=1)
    if version != 1:
        raise ValueError(
            f'{context}.command_contract.version `{version}` is unsupported; expected `1`',
        )

    namespace = validate_command_part(
        command_contract.get('namespace'),
        context=f'{context}.command_contract.namespace',
    )

    expected_namespace = normalize_slug(expect_non_empty_str(manifest.get('name'), context=f'{context}.name'))
    if namespace != expected_namespace:
        raise ValueError(
            f'{context}.command_contract.namespace `{namespace}` must equal '
            f'normalized extension name `{expected_namespace}`',
        )

    prefix = f'{namespace}/'
    for command_name, rel_path in commands_dict.items():
        command_key = expect_non_empty_str(command_name, context=f'{context}.commands key')
        if not command_key.startswith(prefix):
            raise ValueError(
                f'{context}.commands key `{command_key}` must start with namespace prefix `{prefix}`',
            )

        suffix = command_key[len(prefix) :]
        validate_command_part(suffix, context=f'{context}.commands key suffix')

        rel = expect_non_empty_str(rel_path, context=f'{context}.commands.{command_key}')
        try:
            ensure_safe_relative(rel)
        except ValueError as exc:
            raise ValueError(f'{context}.commands.{command_key} invalid: {exc}') from exc


def validate_extension_manifest(payload: object, *, context: str = 'extension_manifest') -> dict[str, Any]:
    """Validate extension manifest schema."""

    manifest = expect_dict(payload, context=context)

    expect_non_empty_str(manifest.get('name'), context=f'{context}.name')
    expect_non_empty_str(manifest.get('version'), context=f'{context}.version')

    if 'description' in manifest and manifest.get('description') is not None:
        expect_non_empty_str(manifest.get('description'), context=f'{context}.description')

    expect_string_list(
        manifest.get('capabilities'),
        context=f'{context}.capabilities',
        allow_empty=False,
        unique=True,
    )

    _validate_entrypoints(manifest, context=context)
    _validate_command_registry(manifest, context=context)

    return manifest
