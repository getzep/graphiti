#!/usr/bin/env python3
"""Compatibility facade for delta contract validators.

The canonical implementations live under ``scripts/delta_contracts_lib/``.
This module remains as a stable import path for existing tooling.
"""

from __future__ import annotations

from delta_contracts_lib import (
    METRIC_KEYS,
    ExtensionInspection,
    inspect_extensions,
    validate_delta_contract_policy,
    validate_extension_manifest,
    validate_migration_sync_policy,
    validate_package_manifest,
    validate_state_migration_manifest,
)

__all__ = [
    'ExtensionInspection',
    'METRIC_KEYS',
    'inspect_extensions',
    'validate_delta_contract_policy',
    'validate_extension_manifest',
    'validate_migration_sync_policy',
    'validate_package_manifest',
    'validate_state_migration_manifest',
]
