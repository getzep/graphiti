"""Versioned extension contract surface for external packs.

The contract is intentionally small:
- explicit ``api_version`` semantics,
- declarative manifest-only registration,
- path-safe entrypoint and command declarations.

This module is runtime-agnostic and can be imported by tooling or runtime
loaders without requiring private hooks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXTENSION_API_VERSION = 1
SUPPORTED_API_VERSIONS = frozenset({EXTENSION_API_VERSION})

_COMMAND_SEGMENT_RE = re.compile(r'^[a-z0-9][a-z0-9-]*$')


@dataclass(slots=True)
class ExtensionCommandContract:
    """Namespaced extension command contract metadata."""

    version: int
    namespace: str


@dataclass(slots=True)
class ExtensionManifest:
    """Validated extension manifest model."""

    name: str
    version: str
    api_version: int
    capabilities: list[str]
    entrypoints: dict[str, str]
    description: str | None = None
    command_contract: ExtensionCommandContract | None = None
    commands: dict[str, str] | None = None

    @property
    def normalized_name(self) -> str:
        return normalize_extension_name(self.name)


def normalize_extension_name(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r'[^a-z0-9]+', '-', lowered)
    slug = re.sub(r'-+', '-', slug).strip('-')
    return slug


def ensure_safe_relative_path(path_value: str) -> Path:
    """Validate pack-relative paths and block traversal.

    Error text intentionally includes ``Unsafe package entry path`` so existing
    callers/tests keep actionable, recognizable diagnostics.
    """

    candidate = path_value.strip()
    if not candidate:
        raise ValueError('Unsafe package entry path: empty path')

    path = Path(candidate)
    if path.is_absolute() or '..' in path.parts or '.' in path.parts:
        raise ValueError(f'Unsafe package entry path: {candidate}')

    return path


def _expect_dict(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f'{context} must be an object')
    return value


def _expect_non_empty_str(value: object, *, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f'{context} must be a non-empty string')
    text = value.strip()
    if not text:
        raise ValueError(f'{context} must be a non-empty string')
    return text


def _expect_int(value: object, *, context: str, min_value: int = 1) -> int:
    if not isinstance(value, int):
        raise ValueError(f'{context} must be an integer')
    if value < min_value:
        raise ValueError(f'{context} must be >= {min_value}')
    return value


def _expect_non_empty_unique_string_list(value: object, *, context: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f'{context} must be a list of strings')

    parsed = [_expect_non_empty_str(item, context=f'{context}[{index}]') for index, item in enumerate(value)]

    if not parsed:
        raise ValueError(f'{context} must not be empty')

    if len(set(parsed)) != len(parsed):
        raise ValueError(f'{context} must not contain duplicates')

    return parsed


def _validate_command_segment(value: object, *, context: str) -> str:
    segment = _expect_non_empty_str(value, context=context)
    if not _COMMAND_SEGMENT_RE.fullmatch(segment):
        raise ValueError(f'{context} must match pattern `{_COMMAND_SEGMENT_RE.pattern}`')
    return segment


def _resolve_api_version(payload: dict[str, Any], *, context: str) -> tuple[int, list[str]]:
    raw_api_version = payload.get('api_version')

    warnings: list[str] = []
    if raw_api_version is None:
        api_version = EXTENSION_API_VERSION
        warnings.append(
            f'{context}.api_version missing; defaulting to `{api_version}` for backward compatibility. '
            'Set api_version explicitly.',
        )
    else:
        api_version = _expect_int(raw_api_version, context=f'{context}.api_version', min_value=1)

    if api_version not in SUPPORTED_API_VERSIONS:
        supported = ', '.join(str(item) for item in sorted(SUPPORTED_API_VERSIONS))
        raise ValueError(
            f'{context}.api_version `{api_version}` is unsupported; supported versions: {supported}',
        )

    return api_version, warnings


def _validate_entrypoints(payload: dict[str, Any], *, context: str) -> dict[str, str]:
    entrypoints_raw = _expect_dict(payload.get('entrypoints'), context=f'{context}.entrypoints')
    if not entrypoints_raw:
        raise ValueError(f'{context}.entrypoints must not be empty')

    entrypoints: dict[str, str] = {}
    for key, value in entrypoints_raw.items():
        entry_name = _expect_non_empty_str(key, context=f'{context}.entrypoints key')
        rel_path = _expect_non_empty_str(value, context=f'{context}.entrypoints.{entry_name}')
        try:
            ensure_safe_relative_path(rel_path)
        except ValueError as exc:
            raise ValueError(f'{context}.entrypoints.{entry_name} invalid: {exc}') from exc

        entrypoints[entry_name] = rel_path

    return entrypoints


def _validate_command_contract(
    payload: dict[str, Any],
    *,
    context: str,
    expected_namespace: str,
    api_version: int,
) -> ExtensionCommandContract | None:
    command_contract_raw = payload.get('command_contract')
    if command_contract_raw is None:
        return None

    command_contract = _expect_dict(command_contract_raw, context=f'{context}.command_contract')
    version = _expect_int(
        command_contract.get('version'),
        context=f'{context}.command_contract.version',
        min_value=1,
    )
    namespace = _validate_command_segment(
        command_contract.get('namespace'),
        context=f'{context}.command_contract.namespace',
    )

    if namespace != expected_namespace:
        raise ValueError(
            f'{context}.command_contract.namespace `{namespace}` must equal '
            f'normalized extension name `{expected_namespace}`',
        )

    if version != api_version:
        raise ValueError(
            f'{context}.command_contract.version `{version}` must match '
            f'{context}.api_version `{api_version}`',
        )

    return ExtensionCommandContract(version=version, namespace=namespace)


def _validate_commands(
    payload: dict[str, Any],
    *,
    context: str,
    command_contract: ExtensionCommandContract | None,
) -> dict[str, str] | None:
    commands_raw = payload.get('commands')
    if commands_raw is None:
        return None

    commands = _expect_dict(commands_raw, context=f'{context}.commands')
    if not commands:
        raise ValueError(f'{context}.commands must not be empty when provided')

    if command_contract is None:
        raise ValueError(
            f'{context}.commands requires {context}.command_contract metadata',
        )

    prefix = f'{command_contract.namespace}/'
    validated: dict[str, str] = {}

    for key, value in commands.items():
        command_name = _expect_non_empty_str(key, context=f'{context}.commands key')
        if not command_name.startswith(prefix):
            raise ValueError(
                f'{context}.commands key `{command_name}` must start with namespace prefix `{prefix}`',
            )

        suffix = command_name[len(prefix) :]
        _validate_command_segment(suffix, context=f'{context}.commands key suffix')

        rel_path = _expect_non_empty_str(value, context=f'{context}.commands.{command_name}')
        try:
            ensure_safe_relative_path(rel_path)
        except ValueError as exc:
            raise ValueError(f'{context}.commands.{command_name} invalid: {exc}') from exc

        validated[command_name] = rel_path

    return validated


def parse_extension_manifest(
    payload: object,
    *,
    context: str = 'extension_manifest',
) -> tuple[ExtensionManifest, list[str]]:
    """Validate and normalize extension manifest payload.

    Returns:
      (manifest, warnings)
    """

    manifest_raw = _expect_dict(payload, context=context)

    name = _expect_non_empty_str(manifest_raw.get('name'), context=f'{context}.name')
    normalized_name = normalize_extension_name(name)
    if not normalized_name:
        raise ValueError(
            f'{context}.name `{name}` must include letters or digits after normalization',
        )

    version = _expect_non_empty_str(manifest_raw.get('version'), context=f'{context}.version')

    description: str | None = None
    if 'description' in manifest_raw and manifest_raw.get('description') is not None:
        description = _expect_non_empty_str(
            manifest_raw.get('description'),
            context=f'{context}.description',
        )

    capabilities = _expect_non_empty_unique_string_list(
        manifest_raw.get('capabilities'),
        context=f'{context}.capabilities',
    )

    api_version, warnings = _resolve_api_version(manifest_raw, context=context)
    entrypoints = _validate_entrypoints(manifest_raw, context=context)

    command_contract = _validate_command_contract(
        manifest_raw,
        context=context,
        expected_namespace=normalized_name,
        api_version=api_version,
    )
    commands = _validate_commands(
        manifest_raw,
        context=context,
        command_contract=command_contract,
    )

    return (
        ExtensionManifest(
            name=name,
            version=version,
            api_version=api_version,
            capabilities=capabilities,
            entrypoints=entrypoints,
            description=description,
            command_contract=command_contract,
            commands=commands,
        ),
        warnings,
    )


__all__ = [
    'EXTENSION_API_VERSION',
    'SUPPORTED_API_VERSIONS',
    'ExtensionCommandContract',
    'ExtensionManifest',
    'ensure_safe_relative_path',
    'normalize_extension_name',
    'parse_extension_manifest',
]
