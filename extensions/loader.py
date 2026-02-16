"""Fail-safe extension loader with isolated diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .contracts import ExtensionManifest, ensure_safe_relative_path, parse_extension_manifest

DiagnosticLevel = Literal['warning', 'error']
MANIFEST_FILENAME = 'manifest.json'


@dataclass(slots=True)
class ExtensionDiagnostic:
    """Actionable extension diagnostic."""

    level: DiagnosticLevel
    location: str
    message: str
    hint: str | None = None

    def render(self) -> str:
        text = f'[{self.level.upper()}] {self.location}: {self.message}'
        if self.hint:
            text += f' (hint: {self.hint})'
        return text


@dataclass(slots=True)
class LoadedExtension:
    """Loaded extension and resolved command/entrypoint files."""

    extension_dir: Path
    manifest_path: Path
    manifest: ExtensionManifest
    entrypoints: dict[str, Path]
    commands: dict[str, Path]


@dataclass(slots=True)
class ExtensionLoadReport:
    """Result of extension discovery + validation."""

    loaded_extensions: list[LoadedExtension] = field(default_factory=list)
    diagnostics: list[ExtensionDiagnostic] = field(default_factory=list)

    @property
    def extension_names(self) -> list[str]:
        return [item.manifest.name for item in self.loaded_extensions]

    @property
    def command_registry(self) -> dict[str, Path]:
        registry: dict[str, Path] = {}
        for extension in self.loaded_extensions:
            registry.update(extension.commands)
        return registry

    @property
    def errors(self) -> list[ExtensionDiagnostic]:
        return [item for item in self.diagnostics if item.level == 'error']

    @property
    def warnings(self) -> list[ExtensionDiagnostic]:
        return [item for item in self.diagnostics if item.level == 'warning']


def _is_extension_dir(candidate: Path) -> bool:
    name = candidate.name
    if name.startswith('.') or name.startswith('__'):
        return False
    return candidate.is_dir()


def _iter_extension_dirs(extensions_dir: Path) -> list[Path]:
    return sorted(candidate for candidate in extensions_dir.iterdir() if _is_extension_dir(candidate))


def discover_extension_manifests(extensions_dir: Path) -> list[Path]:
    """Return discovered manifest paths for direct extension folders."""

    if not extensions_dir.exists() or not extensions_dir.is_dir():
        return []

    manifests: list[Path] = []
    for extension_dir in _iter_extension_dirs(extensions_dir):
        manifest = extension_dir / MANIFEST_FILENAME
        if manifest.exists():
            manifests.append(manifest)
    return manifests


def _resolve_extensions_dir(repo_root: Path, extensions_dir: Path | None) -> Path:
    if extensions_dir is None:
        return (repo_root / 'extensions').resolve()
    if extensions_dir.is_absolute():
        return extensions_dir.resolve()
    return (repo_root / extensions_dir).resolve()


def _resolve_repo_file(repo_root: Path, rel_path: str, *, context: str) -> Path:
    safe_rel = ensure_safe_relative_path(rel_path)
    resolved_repo = repo_root.resolve()
    target = (resolved_repo / safe_rel).resolve()

    try:
        target.relative_to(resolved_repo)
    except ValueError as exc:
        raise ValueError(f'{context} invalid: Unsafe package entry path: {rel_path}') from exc

    return target


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Expected JSON object in {path}')
    return payload


def _resolve_declared_file(
    *,
    repo_root: Path,
    manifest_path: Path,
    rel_path: str,
    missing_label: str,
    context: str,
    report: ExtensionLoadReport,
) -> Path | None:
    try:
        resolved = _resolve_repo_file(repo_root, rel_path, context=context)
    except ValueError as exc:
        report.diagnostics.append(
            ExtensionDiagnostic(
                level='error',
                location=str(manifest_path),
                message=str(exc),
                hint='Use repo-relative paths and avoid `..` traversal segments.',
            ),
        )
        return None

    if not resolved.exists() or not resolved.is_file():
        report.diagnostics.append(
            ExtensionDiagnostic(
                level='error',
                location=str(manifest_path),
                message=f'{missing_label} `{rel_path}`',
                hint='Create the file or update the manifest entry.',
            ),
        )
        return None

    return resolved


def _load_single_extension(
    *,
    repo_root: Path,
    extension_dir: Path,
    report: ExtensionLoadReport,
    seen_names: dict[str, str],
    seen_commands: set[str],
) -> None:
    manifest_path = extension_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        report.diagnostics.append(
            ExtensionDiagnostic(
                level='warning',
                location=str(extension_dir),
                message='missing manifest.json; skipping directory',
                hint='Add manifest.json or remove this directory from extensions/.',
            ),
        )
        return

    try:
        payload = _read_json_object(manifest_path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        report.diagnostics.append(
            ExtensionDiagnostic(
                level='error',
                location=str(manifest_path),
                message=str(exc),
                hint='Fix manifest JSON syntax and object shape.',
            ),
        )
        return

    try:
        manifest, warnings = parse_extension_manifest(payload, context=str(manifest_path))
    except ValueError as exc:
        report.diagnostics.append(
            ExtensionDiagnostic(
                level='error',
                location=str(manifest_path),
                message=str(exc),
                hint='See docs/public/EXTENSION-INTERFACE.md for the required contract.',
            ),
        )
        return

    for warning in warnings:
        hint: str | None = None
        if '.api_version missing' in warning:
            hint = 'Add explicit `api_version` to keep compatibility intent unambiguous.'
        report.diagnostics.append(
            ExtensionDiagnostic(
                level='warning',
                location=str(manifest_path),
                message=warning,
                hint=hint,
            ),
        )

    normalized_name = manifest.normalized_name
    existing_name = seen_names.get(normalized_name)
    if existing_name is not None:
        report.diagnostics.append(
            ExtensionDiagnostic(
                level='error',
                location=str(manifest_path),
                message=(
                    f'duplicate extension name `{manifest.name}` conflicts with `{existing_name}` '
                    f'(normalized: `{normalized_name}`)'
                ),
                hint='Rename the extension or remove duplicate manifests.',
            ),
        )
        return

    extension_failed = False
    resolved_entrypoints: dict[str, Path] = {}
    for entry_name, rel_path in manifest.entrypoints.items():
        resolved = _resolve_declared_file(
            repo_root=repo_root,
            manifest_path=manifest_path,
            rel_path=rel_path,
            missing_label='entrypoint path missing',
            context=f'extension `{manifest.name}` entrypoint `{entry_name}`',
            report=report,
        )
        if resolved is None:
            extension_failed = True
            continue
        resolved_entrypoints[entry_name] = resolved

    resolved_commands: dict[str, Path] = {}
    for command_name, rel_path in (manifest.commands or {}).items():
        if command_name in seen_commands:
            report.diagnostics.append(
                ExtensionDiagnostic(
                    level='error',
                    location=str(manifest_path),
                    message=f'duplicate command `{command_name}` across extensions',
                    hint='Rename the command or move it to only one extension.',
                ),
            )
            extension_failed = True
            continue

        resolved = _resolve_declared_file(
            repo_root=repo_root,
            manifest_path=manifest_path,
            rel_path=rel_path,
            missing_label='command path missing',
            context=f'extension `{manifest.name}` command `{command_name}`',
            report=report,
        )
        if resolved is None:
            extension_failed = True
            continue
        resolved_commands[command_name] = resolved

    if extension_failed:
        return

    seen_names[normalized_name] = manifest.name
    seen_commands.update(resolved_commands)
    report.loaded_extensions.append(
        LoadedExtension(
            extension_dir=extension_dir,
            manifest_path=manifest_path,
            manifest=manifest,
            entrypoints=resolved_entrypoints,
            commands=resolved_commands,
        ),
    )


def load_extensions(
    *,
    repo_root: Path,
    extensions_dir: Path | None = None,
) -> ExtensionLoadReport:
    """Load extension manifests without hard failures.

    Extensions are optional: missing directory is a warning and core stays
    operational with zero loaded extensions.
    """

    resolved_repo = repo_root.resolve()
    resolved_extensions_dir = _resolve_extensions_dir(resolved_repo, extensions_dir)

    report = ExtensionLoadReport()
    if not resolved_extensions_dir.exists():
        report.diagnostics.append(
            ExtensionDiagnostic(
                level='warning',
                location=str(resolved_extensions_dir),
                message='extensions directory missing; continuing with zero extensions',
                hint='Create extensions/<name>/manifest.json to add an extension.',
            ),
        )
        return report

    if not resolved_extensions_dir.is_dir():
        report.diagnostics.append(
            ExtensionDiagnostic(
                level='error',
                location=str(resolved_extensions_dir),
                message='extensions path exists but is not a directory',
                hint='Point --extensions-dir to a directory containing extension manifests.',
            ),
        )
        return report

    seen_names: dict[str, str] = {}
    seen_commands: set[str] = set()

    for extension_dir in _iter_extension_dirs(resolved_extensions_dir):
        _load_single_extension(
            repo_root=resolved_repo,
            extension_dir=extension_dir,
            report=report,
            seen_names=seen_names,
            seen_commands=seen_commands,
        )

    return report


__all__ = [
    'ExtensionDiagnostic',
    'ExtensionLoadReport',
    'LoadedExtension',
    'discover_extension_manifests',
    'load_extensions',
]
