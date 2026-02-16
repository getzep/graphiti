#!/usr/bin/env python3
"""Migrate delta contract artifacts to the current schema policy."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from delta_contracts import (
    validate_delta_contract_policy,
    validate_extension_manifest,
    validate_migration_sync_policy,
    validate_state_migration_manifest,
)
from delta_contracts_lib.common import normalize_slug
from migration_sync_lib import load_json, resolve_repo_root, resolve_safe_child


@dataclass(frozen=True)
class TargetResult:
    target_name: str
    changed_count: int
    inspected_count: int
    changed_paths: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Migrate delta contract files to current schema policy.')
    parser.add_argument('--repo', type=Path, default=Path('.'), help='Repository root or subdirectory')
    parser.add_argument('--extensions-dir', type=Path, default=Path('extensions'))
    parser.add_argument(
        '--policy',
        type=Path,
        default=Path('config/migration_sync_policy.json'),
        help='Migration/sync policy JSON file',
    )
    parser.add_argument(
        '--state-manifest',
        type=Path,
        default=Path('config/state_migration_manifest.json'),
        help='State migration manifest JSON file',
    )
    parser.add_argument(
        '--contract-policy',
        type=Path,
        default=Path('config/delta_contract_policy.json'),
        help='Delta contract policy JSON file',
    )
    parser.add_argument('--write', action='store_true', help='Apply migrations in place')
    return parser.parse_args()


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _stable_command_key(namespace: str, raw_key: str, seen: set[str]) -> str:
    candidate_suffix = normalize_slug(raw_key)
    if not candidate_suffix:
        candidate_suffix = 'command'

    counter = 1
    while True:
        suffix = candidate_suffix if counter == 1 else f'{candidate_suffix}-{counter}'
        command_key = f'{namespace}/{suffix}'
        if command_key not in seen:
            seen.add(command_key)
            return command_key
        counter += 1


def _migrate_extension_manifest(manifest: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    commands = manifest.get('commands')
    if not isinstance(commands, dict) or not commands:
        return manifest, False

    extension_name = str(manifest.get('name', '')).strip()
    namespace = normalize_slug(extension_name)
    if not namespace:
        raise ValueError('Cannot derive namespace from extension name')

    existing_contract = manifest.get('command_contract')
    contract_valid = (
        isinstance(existing_contract, dict)
        and existing_contract.get('version') == 1
        and str(existing_contract.get('namespace', '')).strip() == namespace
    )

    seen_keys: set[str] = set()
    migrated_commands: dict[str, str] = {}
    changed = False

    for key, rel_path in commands.items():
        if not isinstance(key, str) or not isinstance(rel_path, str):
            raise ValueError('Extension commands must map string keys to string paths')

        normalized_key = key.strip()
        if normalized_key.startswith(f'{namespace}/'):
            command_key = _stable_command_key(namespace, normalized_key.split('/', 1)[1], seen_keys)
            if command_key != normalized_key:
                changed = True
        else:
            command_key = _stable_command_key(namespace, normalized_key, seen_keys)
            changed = True

        migrated_commands[command_key] = rel_path

    if changed or not contract_valid:
        manifest['command_contract'] = {
            'version': 1,
            'namespace': namespace,
        }
        manifest['commands'] = migrated_commands
        changed = True

    validate_extension_manifest(manifest, context='migrated extension manifest')
    return manifest, changed


def _assert_supported_target_version(target_name: str, version: int) -> None:
    if version != 1:
        raise ValueError(
            f'Unsupported target version `{version}` for `{target_name}` migration',
        )


def _validate_migration_script_reference(
    *,
    repo_root: Path,
    target_name: str,
    target_cfg: dict[str, Any],
) -> None:
    raw_path = target_cfg.get('migration_script')
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f'Target `{target_name}` missing migration_script path')

    migration_script_path = resolve_safe_child(
        repo_root,
        raw_path,
        context=f'contract migration script for target `{target_name}`',
    )

    if not migration_script_path.exists() or not migration_script_path.is_file():
        raise FileNotFoundError(
            f'Target `{target_name}` migration_script missing: {raw_path}',
        )


def _migrate_target_migration_sync_policy(
    *,
    policy_path: Path,
) -> TargetResult:
    validate_migration_sync_policy(load_json(policy_path), context=str(policy_path), strict=True)
    return TargetResult(
        target_name='migration_sync_policy',
        changed_count=0,
        inspected_count=1,
        changed_paths=(),
    )


def _migrate_target_state_manifest(
    *,
    state_manifest_path: Path,
) -> TargetResult:
    validate_state_migration_manifest(load_json(state_manifest_path), context=str(state_manifest_path))
    return TargetResult(
        target_name='state_migration_manifest',
        changed_count=0,
        inspected_count=1,
        changed_paths=(),
    )


def _migrate_target_extension_commands(
    *,
    repo_root: Path,
    extensions_dir: Path,
    write: bool,
) -> TargetResult:
    if not extensions_dir.exists() or not extensions_dir.is_dir():
        raise FileNotFoundError(f'Extensions directory missing: {extensions_dir}')

    changed_files: list[Path] = []
    inspected = 0

    for extension_dir in sorted(path for path in extensions_dir.iterdir() if path.is_dir()):
        manifest_path = extension_dir / 'manifest.json'
        if not manifest_path.exists():
            continue

        inspected += 1
        manifest_payload = load_json(manifest_path)
        migrated_payload, changed = _migrate_extension_manifest(dict(manifest_payload))
        if not changed:
            continue

        changed_files.append(manifest_path)
        if write:
            manifest_path.write_text(f'{json.dumps(migrated_payload, indent=2)}\n', encoding='utf-8')

    return TargetResult(
        target_name='extension_command_contract',
        changed_count=len(changed_files),
        inspected_count=inspected,
        changed_paths=tuple(changed_files),
    )


def main() -> int:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo.resolve())

    policy_path = _resolve(repo_root, args.policy)
    state_manifest_path = _resolve(repo_root, args.state_manifest)
    extensions_dir = _resolve(repo_root, args.extensions_dir)
    contract_policy_path = _resolve(repo_root, args.contract_policy)

    contract_policy = validate_delta_contract_policy(load_json(contract_policy_path), context=str(contract_policy_path))
    targets = contract_policy.get('targets', {})
    if not isinstance(targets, dict):
        raise ValueError('Contract policy targets must be an object')

    target_handlers = {
        'migration_sync_policy': lambda: _migrate_target_migration_sync_policy(policy_path=policy_path),
        'state_migration_manifest': lambda: _migrate_target_state_manifest(state_manifest_path=state_manifest_path),
        'extension_command_contract': lambda: _migrate_target_extension_commands(
            repo_root=repo_root,
            extensions_dir=extensions_dir,
            write=args.write,
        ),
    }

    results: list[TargetResult] = []

    for target_name, target_cfg_raw in targets.items():
        if target_name not in target_handlers:
            raise ValueError(f'Unsupported contract policy target `{target_name}`')

        if not isinstance(target_cfg_raw, dict):
            raise ValueError(f'Contract policy target `{target_name}` must be an object')

        _validate_migration_script_reference(
            repo_root=repo_root,
            target_name=target_name,
            target_cfg=target_cfg_raw,
        )

        version_value = target_cfg_raw.get('current_version')
        if not isinstance(version_value, int):
            raise ValueError(f'Contract policy target `{target_name}` current_version must be an integer')

        _assert_supported_target_version(target_name, version_value)
        results.append(target_handlers[target_name]())

    mode = 'WRITE' if args.write else 'DRY RUN'
    print(f'Delta contract migrate ({mode})')
    print(f'Repo: {repo_root}')
    print(f'Policy: {contract_policy_path}')

    total_changed = 0
    for result in results:
        total_changed += result.changed_count
        print(
            f'- {result.target_name}: inspected={result.inspected_count}, changed={result.changed_count}',
        )
        for path in result.changed_paths:
            print(f'  * {path}')

    print(f'Total changed artifacts: {total_changed}')

    if not args.write:
        print('No files were modified. Re-run with --write to apply migrations.')

    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
