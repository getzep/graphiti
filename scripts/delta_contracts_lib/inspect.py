from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from migration_sync_lib import load_json, resolve_safe_child

from .extension import validate_extension_manifest


@dataclass(frozen=True)
class ExtensionInspection:
    """Result of inspecting extension contracts under an extension directory."""

    names: list[str]
    command_registry: dict[str, Path]
    issues: list[str]


def inspect_extensions(repo_root: Path, extensions_dir: Path) -> ExtensionInspection:
    """Inspect extension manifests, return discovered names, commands, and issues."""

    issues: list[str] = []
    names: list[str] = []
    command_registry: dict[str, Path] = {}
    seen_names: set[str] = set()

    if not extensions_dir.exists() or not extensions_dir.is_dir():
        issues.append(f'Extensions directory missing: {extensions_dir}')
        return ExtensionInspection(names=names, command_registry=command_registry, issues=issues)

    for extension_dir in sorted(path for path in extensions_dir.iterdir() if path.is_dir()):
        manifest_path = extension_dir / 'manifest.json'
        if not manifest_path.exists():
            issues.append(f'{extension_dir.name}: missing manifest.json')
            continue

        try:
            manifest = validate_extension_manifest(load_json(manifest_path), context=str(manifest_path))
        except (FileNotFoundError, ValueError) as exc:
            issues.append(str(exc))
            continue

        extension_name = str(manifest['name']).strip()
        names.append(extension_name)
        if extension_name in seen_names:
            issues.append(f'{extension_dir.name}: duplicate extension name `{extension_name}`')
        seen_names.add(extension_name)

        entrypoints = manifest.get('entrypoints', {})
        if isinstance(entrypoints, dict):
            for entry_name, rel_path in entrypoints.items():
                if not isinstance(entry_name, str) or not isinstance(rel_path, str):
                    continue
                try:
                    resolved = resolve_safe_child(
                        repo_root,
                        rel_path,
                        context=f'extension `{extension_name}` entrypoint `{entry_name}`',
                    )
                except ValueError as exc:
                    issues.append(str(exc))
                    continue

                if not resolved.exists() or not resolved.is_file():
                    issues.append(
                        f'{manifest_path}: entrypoint path missing `{rel_path}`',
                    )

        commands = manifest.get('commands')
        if isinstance(commands, dict):
            for command_name, rel_path in commands.items():
                if not isinstance(command_name, str) or not isinstance(rel_path, str):
                    continue

                normalized_command = command_name.strip()
                if normalized_command in command_registry:
                    issues.append(
                        f'{manifest_path}: duplicate command `{normalized_command}` across extensions',
                    )
                    continue

                try:
                    resolved_command = resolve_safe_child(
                        repo_root,
                        rel_path,
                        context=f'extension `{extension_name}` command `{normalized_command}`',
                    )
                except ValueError as exc:
                    issues.append(str(exc))
                    continue

                if not resolved_command.exists() or not resolved_command.is_file():
                    issues.append(
                        f'{manifest_path}: command path missing `{rel_path}`',
                    )
                    continue

                command_registry[normalized_command] = resolved_command

    return ExtensionInspection(names=names, command_registry=command_registry, issues=issues)
