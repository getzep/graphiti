# Public Migration + Upstream Sync Toolkit

This repository ships a **delta-layer migration/sync toolkit** designed for:

- repeatable upstream sync preflight checks,
- deterministic history candidate comparison,
- state package export/check/import,
- extension contract validation.

`graphiti_core/**` is intentionally out of scope for this toolkit.

## Architecture (delta layer)

The toolkit is organized as layered contracts + CLIs:

- `scripts/migration_sync_lib.py` — shared path/hash/git + payload integrity helpers
- `scripts/delta_contracts.py` — compatibility facade
- `scripts/delta_contracts_lib/` — canonical contract modules
  - `policy.py`, `state_manifest.py`, `package_manifest.py`, `extension.py`, `inspect.py`
- `scripts/delta_tool.py` — single command surface for all delta tooling
- focused CLIs (`state_migration_*`, `public_history_*`, `upstream_sync_doctor.py`, `extension_contract_check.py`)

Use `delta_tool.py` as the preferred entrypoint.

To inspect all available commands (core + extension-provided):

```bash
python3 scripts/delta_tool.py list-commands
```

By default, `delta_tool` refuses non-list command execution when extension registry warnings exist.
Use `--allow-registry-warnings` only for local debugging while fixing the registry.

## Policy config

- `config/migration_sync_policy.json`
  - upstream/origin remotes + branch defaults,
  - sync-button safety policy,
  - history export metric coefficients,
  - history scorecard threshold + weights,
  - weekly cadence metadata.
- `config/state_migration_manifest.json`
  - required files,
  - optional globs,
  - exclusion patterns for migration package generation.
- `config/delta_contract_policy.json`
  - canonical contract-version metadata,
  - migration script references,
  - command-contract governance notes.

## 0) Validate contracts first

```bash
python3 scripts/delta_tool.py contracts-check -- \
  --policy config/migration_sync_policy.json \
  --state-manifest config/state_migration_manifest.json \
  --contract-policy config/delta_contract_policy.json \
  --extensions-dir extensions \
  --strict
```

This validates schema shape + key invariants for policy/config/extension contracts.

## 0.5) Contract migration path (v1 → future)

```bash
python3 scripts/delta_tool.py contracts-migrate -- \
  --repo . \
  --extensions-dir extensions \
  --contract-policy config/delta_contract_policy.json

# apply in place
python3 scripts/delta_tool.py contracts-migrate -- \
  --repo . \
  --extensions-dir extensions \
  --contract-policy config/delta_contract_policy.json \
  --write
```

Current migration policy executes target-aware handlers declared in `delta_contract_policy.json`:
- `migration_sync_policy` (schema validation + version gate),
- `state_migration_manifest` (schema validation + version gate),
- `extension_command_contract` (namespaced command normalization to v1).

## 1) Upstream sync doctor

```bash
python3 scripts/delta_tool.py sync-doctor -- \
  --repo . \
  --policy config/migration_sync_policy.json \
  --dry-run \
  --check-sync-button-safety
```

Use `--ensure-upstream` to add missing upstream remote from policy (when not in dry-run).
Use `--allow-dirty` only for local diagnostics when you intentionally run checks on a dirty working tree.

## 2) History candidate reports + scorecard

```bash
python3 scripts/delta_tool.py history-export -- \
  --repo . \
  --mode filtered-history \
  --dry-run \
  --report reports/publicization/filtered-history.md \
  --summary-json reports/publicization/filtered-history.json

python3 scripts/delta_tool.py history-export -- \
  --repo . \
  --mode clean-foundation \
  --dry-run \
  --report reports/publicization/clean-foundation.md \
  --summary-json reports/publicization/clean-foundation.json

python3 scripts/delta_tool.py history-scorecard -- \
  --filtered-summary reports/publicization/filtered-history.json \
  --clean-summary reports/publicization/clean-foundation.json \
  --policy config/migration_sync_policy.json \
  --out reports/publicization/history-scorecard.md
```

Policy fallback rule is encoded in scorecard output:
- choose clean-foundation automatically if filtered-history score is below threshold,
- or if unresolved HIGH risk remains.

## 3) State migration kit

```bash
python3 scripts/delta_tool.py state-export -- \
  --manifest config/state_migration_manifest.json \
  --out /tmp/graphiti-state-export \
  --dry-run

python3 scripts/delta_tool.py state-check -- \
  --package /tmp/graphiti-state-export \
  --dry-run

python3 scripts/delta_tool.py state-import -- \
  --in /tmp/graphiti-state-export \
  --dry-run

# non-dry-run import with atomic rollback (default)
python3 scripts/delta_tool.py state-import -- \
  --in /tmp/graphiti-state-export \
  --target .

# explicitly disable rollback semantics (debug-only)
python3 scripts/delta_tool.py state-import -- \
  --in /tmp/graphiti-state-export \
  --target . \
  --no-atomic
```

Notes:
- dry-run export writes package manifest preview (no payload files copied),
- non-dry-run export writes payload files and checksums for deterministic imports,
- import/check share the same payload integrity evaluation path,
- import defaults to atomic apply with rollback on write failure.

## 4) Extension contract check

```bash
python3 scripts/delta_tool.py extension-check -- --strict
```

Checks `extensions/*/manifest.json` for:
- required fields (`name`, `version`, `capabilities`, `entrypoints`),
- optional `command_contract` metadata,
- optional extension command registry (`commands`) with **namespaced keys**:
  - `<namespace>/<command>`
  - `namespace` must equal normalized extension name,
- duplicate extension names/capabilities,
- traversal-safe relative entrypoint/command paths,
- missing entrypoint/command files.

## CI policy

Canonical CI workflows:
- `.github/workflows/ci.yml`
- `.github/workflows/migration-sync-tooling.yml`

Legacy one-off workflows are archived under `.github/workflows-archive/` and are not part of canonical PR gating.

The full migration/sync CI pipeline command is:

```bash
bash scripts/ci/run_migration_sync_toolkit.sh
```
