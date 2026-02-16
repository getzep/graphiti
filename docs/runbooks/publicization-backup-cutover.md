# Publicization Backup Cutover Runbook

## Purpose
Run one deterministic pre-cutover safety lane that combines migration dry-run checks with encrypted snapshot + restore-test verification.

## Scope
- Repo: `graphiti-openclaw-integration-e2e`
- Canonical entrypoint: `scripts/backup.sh`
- Required scripts:
  - `scripts/state_migration_export.py`
  - `scripts/state_migration_check.py`
  - `scripts/state_migration_import.py`
  - `scripts/snapshot_create.py`
  - `scripts/snapshot_restore_test.py`

## Required environment
- `GRAPHITI_SNAPSHOT_PASSPHRASE`: passphrase for encrypted snapshot create/restore.
  - Treat as secret material; do not commit or echo in shell history.
  - Prefer ephemeral session export: `read -r -s GRAPHITI_SNAPSHOT_PASSPHRASE`.

## Pre-cutover sequence (must run in order)

1) **Migration package dry-run (clean workspace path)**
```bash
set -euo pipefail
rm -rf /tmp/graphiti-state-export-clean
python3 scripts/state_migration_export.py --dry-run --out /tmp/graphiti-state-export-clean
python3 scripts/state_migration_check.py --package /tmp/graphiti-state-export-clean --dry-run
python3 scripts/state_migration_import.py --dry-run --allow-overwrite --in /tmp/graphiti-state-export-clean
```

2) **Create encrypted snapshot artifact**
```bash
python3 scripts/snapshot_create.py \
  --repo . \
  --snapshot-dir backup/snapshots \
  --manifest backup/snapshots/latest-manifest.json \
  --include state \
  --include exports \
  --include reports/private \
  --include logs \
  --force
```

3) **Restore-test snapshot before cutover GO**
```bash
python3 scripts/snapshot_restore_test.py \
  --repo . \
  --snapshot-dir backup/snapshots \
  --manifest backup/snapshots/latest-manifest.json
```

4) **Optional single-command operator lane**
```bash
scripts/backup.sh precutover
```
This wrapper is rerun-safe for snapshot outputs (`snapshot_create.py` is invoked with `--force` inside `precutover`).

## GO / NO-GO decision points

- **NO-GO** if any migration dry-run command fails.
- **NO-GO** if snapshot create fails (missing include paths, missing passphrase, archive write failure).
- **NO-GO** if restore-test fails checksum/size verification.
- **GO** only when all checks are green and evidence is recorded in `reports/publicization/backup-readiness.md`.

## Rollback guidance

### Decision point A: pre-cutover, no production writes yet
- Stop the cutover.
- Fix the failing migration/snapshot prereq and rerun this runbook from step 1.

### Decision point B: post-cutover issue discovered
- Pause rollout and isolate writes.
- Use `backup/snapshots/latest-manifest.json` + encrypted archive to restore validated state in an isolated target.
- Re-run `scripts/snapshot_restore_test.py` against restored payload before any production rollback action.

### Decision point C: migration import caused drift in target repo
- If not committed: restore via `git restore`/`git checkout --` for affected files.
- If committed: revert the import commit and reopen cutover gate review.

## Post-run hygiene
```bash
rm -rf /tmp/graphiti-state-export-clean
```
Keep encrypted snapshots only in approved local storage; never commit snapshot payloads/manifests with private data.
