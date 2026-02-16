#!/usr/bin/env bash
set -euo pipefail

uvx ruff check --output-format=github
python3 scripts/delta_tool.py boundary-lint -- \
  --manifest config/public_export_allowlist.yaml \
  --denylist config/public_export_denylist.yaml
python3 scripts/delta_tool.py contracts-check -- \
  --policy config/migration_sync_policy.json \
  --state-manifest config/state_migration_manifest.json \
  --contract-policy config/delta_contract_policy.json \
  --extensions-dir extensions \
  --strict
