#!/usr/bin/env python3
"""Validate delta-layer config and extension contracts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from delta_contracts import (
    inspect_extensions,
    validate_delta_contract_policy,
    validate_extension_manifest,
    validate_migration_sync_policy,
    validate_state_migration_manifest,
)
from migration_sync_lib import load_json, repo_relative, resolve_repo_root, resolve_safe_child

EXPECTED_CONTRACT_TARGETS = {
    'migration_sync_policy',
    'state_migration_manifest',
    'extension_command_contract',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validate delta-layer contract files.')
    parser.add_argument('--repo', type=Path, default=Path('.'), help='Repository root or subdirectory')
    parser.add_argument(
        '--policy',
        type=Path,
        default=Path('config/migration_sync_policy.json'),
        help='Migration/sync policy JSON path',
    )
    parser.add_argument(
        '--state-manifest',
        type=Path,
        default=Path('config/state_migration_manifest.json'),
        help='State migration manifest JSON path',
    )
    parser.add_argument('--extensions-dir', type=Path, default=Path('extensions'))
    parser.add_argument(
        '--contract-policy',
        type=Path,
        default=Path('config/delta_contract_policy.json'),
        help='Delta contract policy JSON path',
    )
    parser.add_argument('--strict', action='store_true', help='Exit non-zero when issues are found')
    return parser.parse_args()


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _collect_extension_version_issues(
    *,
    repo_root: Path,
    extensions_dir: Path,
    max_version: int,
) -> list[str]:
    issues: list[str] = []

    if not extensions_dir.exists() or not extensions_dir.is_dir():
        return issues

    for extension_dir in sorted(path for path in extensions_dir.iterdir() if path.is_dir()):
        manifest_path = extension_dir / 'manifest.json'
        if not manifest_path.exists():
            continue

        try:
            manifest = validate_extension_manifest(load_json(manifest_path), context=str(manifest_path))
        except (FileNotFoundError, ValueError):
            # Base extension validation is already handled via inspect_extensions.
            continue

        commands = manifest.get('commands')
        if not isinstance(commands, dict) or not commands:
            continue

        command_contract = manifest.get('command_contract')
        if not isinstance(command_contract, dict):
            continue

        version_value = command_contract.get('version')
        if not isinstance(version_value, int):
            continue

        if version_value > max_version:
            issues.append(
                f'{manifest_path}: command_contract.version `{version_value}` '
                f'exceeds policy target version `{max_version}`',
            )

    return issues


def _validate_cross_contract_invariants(
    *,
    repo_root: Path,
    policy_path: Path,
    policy_payload: dict[str, Any],
    manifest_path: Path,
    state_manifest_payload: dict[str, Any],
    contract_policy_path: Path,
    contract_policy_payload: dict[str, Any],
    extensions_dir: Path,
) -> list[str]:
    issues: list[str] = []

    required_files_raw = state_manifest_payload.get('required_files', [])
    required_files = {
        item.strip()
        for item in required_files_raw
        if isinstance(item, str) and item.strip()
    }

    required_contract_files = {
        repo_relative(policy_path, repo_root),
        repo_relative(manifest_path, repo_root),
        repo_relative(contract_policy_path, repo_root),
    }

    for missing_path in sorted(required_contract_files - required_files):
        issues.append(
            f'{manifest_path}: required_files missing contract artifact `{missing_path}`',
        )

    targets = contract_policy_payload.get('targets', {})
    if isinstance(targets, dict):
        target_names = set(targets)

        missing_targets = sorted(EXPECTED_CONTRACT_TARGETS - target_names)
        if missing_targets:
            issues.append(
                f'{contract_policy_path}: missing required targets: {", ".join(missing_targets)}',
            )

        unsupported_targets = sorted(target_names - EXPECTED_CONTRACT_TARGETS)
        if unsupported_targets:
            issues.append(
                f'{contract_policy_path}: unsupported targets declared: {", ".join(unsupported_targets)}',
            )

        for target_name, raw_cfg in targets.items():
            if not isinstance(raw_cfg, dict):
                continue

            migration_script_value = raw_cfg.get('migration_script')
            if not isinstance(migration_script_value, str) or not migration_script_value.strip():
                continue

            try:
                migration_script_path = resolve_safe_child(
                    repo_root,
                    migration_script_value,
                    context=f'contract policy migration script for target `{target_name}`',
                )
            except ValueError as exc:
                issues.append(str(exc))
                continue

            if not migration_script_path.exists() or not migration_script_path.is_file():
                issues.append(
                    f'{contract_policy_path}: migration_script for target `{target_name}` '
                    f'missing `{migration_script_value}`',
                )

        extension_target = targets.get('extension_command_contract')
        if isinstance(extension_target, dict):
            version_value = extension_target.get('current_version')
            if isinstance(version_value, int):
                issues.extend(
                    _collect_extension_version_issues(
                        repo_root=repo_root,
                        extensions_dir=extensions_dir,
                        max_version=version_value,
                    ),
                )

    return issues


def main() -> int:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo.resolve())

    policy_path = _resolve(repo_root, args.policy)
    manifest_path = _resolve(repo_root, args.state_manifest)
    extensions_dir = _resolve(repo_root, args.extensions_dir)
    contract_policy_path = _resolve(repo_root, args.contract_policy)

    issues: list[str] = []

    policy_payload: dict[str, Any] | None = None
    manifest_payload: dict[str, Any] | None = None
    contract_policy_payload: dict[str, Any] | None = None

    try:
        policy_payload = validate_migration_sync_policy(
            load_json(policy_path),
            context=str(policy_path),
            strict=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        issues.append(str(exc))

    try:
        manifest_payload = validate_state_migration_manifest(load_json(manifest_path), context=str(manifest_path))
    except (FileNotFoundError, ValueError) as exc:
        issues.append(str(exc))

    try:
        contract_policy_payload = validate_delta_contract_policy(load_json(contract_policy_path), context=str(contract_policy_path))
    except (FileNotFoundError, ValueError) as exc:
        issues.append(str(exc))

    extension_report = inspect_extensions(repo_root=repo_root, extensions_dir=extensions_dir)
    issues.extend(extension_report.issues)

    if policy_payload is not None and manifest_payload is not None and contract_policy_payload is not None:
        issues.extend(
            _validate_cross_contract_invariants(
                repo_root=repo_root,
                policy_path=policy_path,
                policy_payload=policy_payload,
                manifest_path=manifest_path,
                state_manifest_payload=manifest_payload,
                contract_policy_path=contract_policy_path,
                contract_policy_payload=contract_policy_payload,
                extensions_dir=extensions_dir,
            ),
        )

    if issues:
        print('Delta contract check: issues found', file=sys.stderr)
        for issue in issues:
            print(f'- {issue}', file=sys.stderr)
        return 1 if args.strict else 0

    print(
        'Delta contract check OK '
        f'(policy={policy_path.name}, state_manifest={manifest_path.name}, '
        f'contract_policy={contract_policy_path.name}, '
        f'extensions={len(extension_report.names)}, extension_commands={len(extension_report.command_registry)})',
    )
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
