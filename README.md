# Graphiti OpenClaw Fork

This repository tracks upstream Graphiti while adding a **delta/control layer** for publicization, migration, upstream-sync safety, and extension governance.

If you want the full upstream Graphiti framework docs (core runtime, drivers, examples), start here:
- Upstream repo: <https://github.com/getzep/graphiti>
- Upstream docs: <https://help.getzep.com/graphiti>

---

## Current status (publicization + hardening)

- Publicization execution lanes completed: adapter wiring, backup wiring, cron cutover.
- Integration gate: **GO** (`reports/publicization/integration-report.md`).
- Boundary policy: **ALLOW=370 / BLOCK=0 / AMBIGUOUS=0**.
- Sync doctor safety decision: **ALLOW** under policy threshold.

Runtime reliability rails (overlay manifest, drift guard, deterministic rebuild, migration integrity, cron canary) live in the private overlay companion repo: `yhl999/graphiti-openclaw-private`.

---

## What this fork adds (delta layer)

The fork introduces a policy-first operator layer that sits alongside upstream Graphiti core.

### 1) Boundary + contract enforcement
- Public boundary policy + audit tooling
- Contract validation for migration/sync artifacts
- Cross-contract invariant checks (not only per-file schema checks)

Key files:
- `scripts/public_boundary_policy.py`
- `scripts/public_repo_boundary_audit.py`
- `scripts/delta_contract_check.py`
- `config/public_export_allowlist.yaml`
- `config/delta_contract_policy.json`

### 2) Unified command surface
A single command bus for delta operations:

```bash
python3 scripts/delta_tool.py list-commands
```

By default, command execution fails fast when extension registry warnings exist.
Use `--allow-registry-warnings` only for local debugging.

### 3) Migration/sync toolkit
- Upstream sync doctor
- History export + scorecard decisioning
- State export/check/import with payload integrity validation
- Contract migration tooling for versioned contract artifacts

Key files:
- `scripts/upstream_sync_doctor.py`
- `scripts/public_history_export.py`
- `scripts/public_history_scorecard.py`
- `scripts/state_migration_export.py`
- `scripts/state_migration_check.py`
- `scripts/state_migration_import.py`
- `scripts/delta_contract_migrate.py`

### 4) Extension command governance
- Versioned `command_contract`
- Mandatory namespace alignment with extension name
- Namespaced command keys (`<namespace>/<command>`)
- Centralized extension inspection/registry build

Key files:
- `scripts/delta_contracts_lib/extension.py`
- `scripts/delta_contracts_lib/inspect.py`
- `extensions/migration_sync/manifest.json`

---

## Quick start (delta tooling)

From repo root:

```bash
# lint + boundary + contracts
bash scripts/ci/run_ruff_lint.sh

# full migration/sync delta pipeline (dry-run checks)
bash scripts/ci/run_migration_sync_toolkit.sh
```

Core contract validation:

```bash
python3 scripts/delta_tool.py contracts-check -- \
  --policy config/migration_sync_policy.json \
  --state-manifest config/state_migration_manifest.json \
  --contract-policy config/delta_contract_policy.json \
  --extensions-dir extensions \
  --strict
```

Contract migration (target-aware):

```bash
python3 scripts/delta_tool.py contracts-migrate -- \
  --repo . \
  --policy config/migration_sync_policy.json \
  --state-manifest config/state_migration_manifest.json \
  --contract-policy config/delta_contract_policy.json

python3 scripts/delta_tool.py contracts-migrate -- \
  --repo . \
  --policy config/migration_sync_policy.json \
  --state-manifest config/state_migration_manifest.json \
  --contract-policy config/delta_contract_policy.json \
  --write
```

State import uses atomic rollback by default:

```bash
python3 scripts/delta_tool.py state-import -- \
  --in /tmp/graphiti-state-export \
  --target .

# debug only
python3 scripts/delta_tool.py state-import -- \
  --in /tmp/graphiti-state-export \
  --target . \
  --no-atomic
```

---

## Architecture map

- Upstream Graphiti core: `graphiti_core/**` (intentionally preserved)
- Delta contracts: `scripts/delta_contracts_lib/**`
- Compatibility facade: `scripts/delta_contracts.py`
- Unified CLI: `scripts/delta_tool.py`
- Policy/config surface: `config/**`
- Public operator docs: `docs/public/**`

---

## Docs index (fork-specific)

- `docs/public/BOUNDARY-CONTRACT.md`
- `docs/public/MIGRATION-SYNC-TOOLKIT.md`
- `docs/public/RELEASE-CHECKLIST.md`
- `docs/runbooks/runtime-pack-overlay.md`
- `docs/runbooks/publicization-backup-cutover.md`
- `docs/runbooks/state-migration.md`
- `docs/runbooks/upstream-sync-openclaw.md`
- `prd/EPIC-PUBLICIZATION-UPSTREAM-SYNC-SIMPLIFICATION-v1.md`
- `prd/EXEC-PUBLICIZATION-INTEGRATION-E2E-v1.md`

---

## CI policy

Canonical PR gates:
- `.github/workflows/ci.yml`
- `.github/workflows/migration-sync-tooling.yml`

Legacy duplicate one-off workflows are archived under:
- `.github/workflows-archive/`

---

## Notes

This README is intentionally fork-operator focused.
For full upstream API/runtime usage docs, refer to upstream Graphiti docs/repo links at the top.
