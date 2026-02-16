"""Public extension contract + loader APIs."""

from .contracts import (
    EXTENSION_API_VERSION,
    SUPPORTED_API_VERSIONS,
    ExtensionCommandContract,
    ExtensionManifest,
    ensure_safe_relative_path,
    normalize_extension_name,
    parse_extension_manifest,
)
from .loader import (
    ExtensionDiagnostic,
    ExtensionLoadReport,
    LoadedExtension,
    discover_extension_manifests,
    load_extensions,
)

__all__ = [
    'EXTENSION_API_VERSION',
    'SUPPORTED_API_VERSIONS',
    'ExtensionCommandContract',
    'ExtensionDiagnostic',
    'ExtensionLoadReport',
    'ExtensionManifest',
    'LoadedExtension',
    'discover_extension_manifests',
    'ensure_safe_relative_path',
    'load_extensions',
    'normalize_extension_name',
    'parse_extension_manifest',
]
