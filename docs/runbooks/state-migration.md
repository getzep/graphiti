# State Migration Runbook (graphiti-openclaw)

## Scope
Use this runbook to move runtime state between environments after history/registry cutover with minimal risk.

- Source of truth commands: `scripts/state_migration_export.py`, `scripts/state_migration_check.py`, `scripts/state_migration_import.py`.
- Scope is limited to `config/**` and runtime state artifacts selected in `config/state_migration_manifest.json`.
- Artifact format is deterministic and inspectable by default at:
  - `package_manifest.json`
  - `payload/**`

## Security and leakage prevention

1. Always stage migration artifacts in a temporary path outside the repository.
2. Run checks before every import:
   - `python3 scripts/state_migration_check.py --package /tmp/graphiti-state-export --target . --dry-run`
   - this validates both package integrity and manifest compatibility against target config.
3. Confirm payload contains only expected files:
   - inspect `package_manifest.json` for `entries`.
   - confirm paths are under expected scopes (`config/`, `state/`, docs and scripts per manifest).
4. Do not commit artifacts:
   - `git status --short`
   - verify exports are not tracked: `git status --ignored` should show only standard ignore status.
5. Keep artifacts out of shared/shared-dev scratch locations and IDE workspace caches.

## Runbook steps

### 0) Dry-run discovery

```bash
python3 scripts/state_migration_export.py --repo /path/to/source --manifest config/state_migration_manifest.json --out /tmp/graphiti-state-export --dry-run --force
python3 scripts/state_migration_check.py --package /tmp/graphiti-state-export --dry-run
```

If output is clean, proceed.

### 1) Build migration package

```bash
python3 scripts/state_migration_export.py --repo /path/to/source --manifest config/state_migration_manifest.json --out /tmp/graphiti-state-export --force
```

### 2) Validate target compatibility

```bash
python3 scripts/state_migration_check.py --package /tmp/graphiti-state-export --target /path/to/target --dry-run
```

If compatibility fails, stop and fix manifest/version drift before import.

### 3) Import dry-run

```bash
python3 scripts/state_migration_import.py --in /tmp/graphiti-state-export --target /path/to/target --dry-run
```

Review output:
- `skip identical` entries are no-ops for idempotent re-runs.
- `write` entries will be applied.

### 4) Import into target

```bash
python3 scripts/state_migration_import.py --in /tmp/graphiti-state-export --target /path/to/target
```

Use `--no-atomic` only for manual recovery operations; keep default atomic behavior for production.

### 5) Post-import verification

```bash
python3 scripts/state_migration_check.py --package /tmp/graphiti-state-export --target /path/to/target --dry-run
python3 scripts/state_migration_export.py --repo /path/to/target --manifest config/state_migration_manifest.json --out /tmp/graphiti-state-export-target --dry-run --force
python3 scripts/state_migration_check.py --package /tmp/graphiti-state-export-target --target /path/to/target --dry-run
```

`git status` should remain clean after import unless you expect working-tree state changes.

## Integrity suite (post-cutover hardening)

After cutover, run the private overlay integrity suite to verify migration invariants on active runtime:

```bash
python3 /path/to/graphiti-openclaw-private/scripts/sqlite_migration_integrity.py \
  --source-root /path/to/archive/deprecated-2026-02/graphiti-legacy-<timestamp> \
  --target-root /path/to/graphiti-openclaw-runtime \
  --out /tmp/sqlite-migration-integrity.json \
  --allow-growth
```

Notes:
- `--allow-growth` is expected for live systems where runtime has continued ingest after migration.
- Integrity still requires `PRAGMA integrity_check=ok` and no table loss.

Dry-run rollback drill:

```bash
python3 /path/to/graphiti-openclaw-private/scripts/sqlite_migration_rollback.py \
  --backup-dir /path/to/graphiti-openclaw-runtime/state_migration_backups/<snapshot-ts> \
  --target-root /path/to/graphiti-openclaw-runtime
```

## Rollback and recovery

### If import was done on an uncommitted working tree
- Restore changed files with `git checkout -- <file>` or `git restore <file>` for each touched path.
- Re-run check in dry-run mode and repeat import only when state is clean.

### If import was committed
- Identify import commit `C`.
- Revert the commit:

```bash
git checkout main
git pull --ff-only origin main
git revert C
git push origin main
```

- If `C` is a merge commit, use `git revert -m 1 C`.
- Open a corrective follow-up PR for the desired state transition.

## Artifact deletion

Immediately after successful migration validation:

```bash
rm -rf /tmp/graphiti-state-export /tmp/graphiti-state-export-target
```

If local policy requires secure deletion, replace with platform-appropriate secure erase tooling.

Recommended:

- macOS: `rm -rf /tmp/graphiti-state-export*`
- Linux: `shred -u -v /tmp/graphiti-state-export*/*` where supported, then remove directories.

Confirm deletion:

```bash
test ! -d /tmp/graphiti-state-export
test ! -d /tmp/graphiti-state-export-target
```
