#!/usr/bin/env bash
set -euo pipefail

python3 scripts/delta_tool.py contracts-check -- \
  --policy config/migration_sync_policy.json \
  --state-manifest config/state_migration_manifest.json \
  --contract-policy config/delta_contract_policy.json \
  --extensions-dir extensions \
  --strict

# PR-safe: --allow-missing-upstream + --dry-run ensures CI does not hard-fail
# when the upstream remote/refs are absent (e.g. shallow clones, forks without
# upstream configured).  Sync-button-safety degrades to a warning in this mode.
python3 scripts/delta_tool.py sync-doctor -- \
  --repo . \
  --policy config/migration_sync_policy.json \
  --dry-run \
  --allow-missing-upstream \
  --allow-dirty \
  --check-sync-button-safety \
  --output-json /tmp/upstream-sync-doctor.json

python3 scripts/delta_tool.py history-export -- \
  --repo . \
  --mode filtered-history \
  --dry-run \
  --report /tmp/filtered-history.md \
  --summary-json /tmp/filtered-history.json

python3 scripts/delta_tool.py history-export -- \
  --repo . \
  --mode clean-foundation \
  --dry-run \
  --report /tmp/clean-foundation.md \
  --summary-json /tmp/clean-foundation.json

python3 scripts/delta_tool.py history-scorecard -- \
  --filtered-summary /tmp/filtered-history.json \
  --clean-summary /tmp/clean-foundation.json \
  --policy config/migration_sync_policy.json \
  --out /tmp/history-scorecard.md \
  --summary-json /tmp/history-scorecard.json

python3 scripts/delta_tool.py state-export -- \
  --manifest config/state_migration_manifest.json \
  --out /tmp/graphiti-state-export \
  --dry-run \
  --force

python3 scripts/delta_tool.py state-check -- \
  --package /tmp/graphiti-state-export \
  --dry-run

python3 scripts/delta_tool.py state-import -- \
  --in /tmp/graphiti-state-export \
  --dry-run
