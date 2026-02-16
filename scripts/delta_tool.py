#!/usr/bin/env python3
"""Single entrypoint for delta-layer migration/sync tooling."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from delta_contracts import inspect_extensions
from migration_sync_lib import resolve_repo_root

CORE_COMMAND_TO_SCRIPT = {
    'boundary-audit': 'public_repo_boundary_audit.py',
    'boundary-lint': 'public_boundary_policy_lint.py',
    'contracts-check': 'delta_contract_check.py',
    'contracts-migrate': 'delta_contract_migrate.py',
    'extension-check': 'extension_contract_check.py',
    'sync-doctor': 'upstream_sync_doctor.py',
    'history-export': 'public_history_export.py',
    'history-scorecard': 'public_history_scorecard.py',
    'state-export': 'state_migration_export.py',
    'state-check': 'state_migration_check.py',
    'state-import': 'state_migration_import.py',
    'list-commands': '',
}


def _resolve_repo(candidate: Path) -> Path:
    try:
        return resolve_repo_root(candidate)
    except subprocess.CalledProcessError:
        return candidate.resolve()


def _build_command_registry(
    scripts_dir: Path,
    repo_root: Path,
) -> tuple[dict[str, Path], list[str]]:
    registry: dict[str, Path] = {}

    for command, script_name in CORE_COMMAND_TO_SCRIPT.items():
        if not script_name:
            continue
        registry[command] = (scripts_dir / script_name).resolve()

    extension_report = inspect_extensions(
        repo_root=repo_root,
        extensions_dir=(repo_root / 'extensions').resolve(),
    )
    warnings = list(extension_report.issues)

    for command_name, script_path in extension_report.command_registry.items():
        if command_name in registry:
            warnings.append(
                f'Extension command `{command_name}` collides with core command and was ignored.',
            )
            continue
        registry[command_name] = script_path

    return registry, warnings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run delta-layer tools via a single stable command surface.',
    )
    parser.add_argument(
        '--repo',
        type=Path,
        default=Path('.'),
        help='Repository root used for extension command discovery (default: cwd)',
    )
    parser.add_argument(
        '--allow-registry-warnings',
        action='store_true',
        help='Allow command execution even when extension registry inspection reports warnings.',
    )
    parser.add_argument(
        'command',
        help='Tool command (use `list-commands` to print available commands)',
    )
    parser.add_argument(
        'tool_args',
        nargs=argparse.REMAINDER,
        help='Arguments forwarded to the selected tool. Prefix with `--` when needed.',
    )
    return parser.parse_args(argv)


def _print_registry_warnings(warnings: list[str]) -> None:
    if not warnings:
        return
    print('delta_tool: extension registry warnings:', file=sys.stderr)
    for warning in warnings:
        print(f'- {warning}', file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    forwarded = list(args.tool_args)
    if forwarded and forwarded[0] == '--':
        forwarded = forwarded[1:]

    scripts_dir = Path(__file__).resolve().parent
    repo_root = _resolve_repo(args.repo.resolve())
    command_registry, warnings = _build_command_registry(scripts_dir, repo_root)

    if args.command == 'list-commands':
        print('Available delta_tool commands:')
        for command in sorted(['list-commands', *command_registry]):
            print(f'- {command}')
        _print_registry_warnings(warnings)
        return 0

    if warnings and not args.allow_registry_warnings:
        _print_registry_warnings(warnings)
        print(
            'delta_tool: refusing to execute commands while extension registry warnings exist. '
            'Use --allow-registry-warnings to override.',
            file=sys.stderr,
        )
        return 1

    script_path = command_registry.get(args.command)
    if script_path is None:
        print(f'Unknown delta_tool command: {args.command}', file=sys.stderr)
        print('Use `list-commands` to inspect available commands.', file=sys.stderr)
        _print_registry_warnings(warnings)
        return 2

    result = subprocess.run([sys.executable, str(script_path), *forwarded], check=False)
    return result.returncode


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
