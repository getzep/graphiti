# migration-sync extension

This extension defines the contract surface for delta-layer migration + upstream sync tooling.

## Why this exists

- keep migration/sync entrypoints explicit and discoverable,
- provide a stable contract checked by `scripts/extension_contract_check.py`,
- avoid ad-hoc script discovery from CI/runbooks.

## Entrypoints

See `manifest.json` for canonical command paths.
