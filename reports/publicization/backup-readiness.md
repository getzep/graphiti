# Publicization Backup Readiness Report

- Task: `task-graphiti-publicization-db-backup-wiring`
- Owner: Publicization lane operator
- Report date: 2026-02-16 (ET)

## Backup gate status
- **Migration dry-run gate:** GO (commands defined and validated in PRD validation block)
- **Snapshot CLI gate:** GO (`snapshot_create.py` and `snapshot_restore_test.py` ship deterministic help/compile surface)
- **Runbook gate:** GO (`docs/runbooks/publicization-backup-cutover.md`)
- **Overall backup gate:** **GO for cutover rehearsal**, pending real environment passphrase + runtime data path availability for final production execution

## Latest evidence (repo-local validation)

Executed sequence:
1. `state_migration_export.py --dry-run`
2. `state_migration_check.py --dry-run`
3. `state_migration_import.py --dry-run --allow-overwrite`
4. `snapshot_create.py --help`
5. `snapshot_restore_test.py --help`
6. `python3 -m compileall scripts/snapshot_create.py scripts/snapshot_restore_test.py`

Result summary:
- All required dry-run migration commands exit 0.
- Snapshot tooling compiles and exposes stable CLI usage.
- Backup runbook + readiness artifacts are present in-repo.

## Blockers and remediation

- Blocker: production cutover has no valid snapshot encryption passphrase in CI (expected).
  - Remediation: provide `GRAPHITI_SNAPSHOT_PASSPHRASE` through approved secret channel at execution time.
  - Owner: cutover operator.
- Blocker: runtime directories (`state/`, `exports/`, `reports/private`, `logs/`) may be absent in clean dev clones.
  - Remediation: create/populate runtime paths in target environment before `snapshot_create.py`.
  - Owner: environment owner.

## Archive/deprecation readiness for legacy repo

Archive/deprecation decision for legacy `clawd-graphiti` remains **conditional**.

Archive readiness preconditions:
1. This backup lane passes in production-like environment with encrypted snapshot + restore-test evidence.
2. Migration dry-run remains clean after latest upstream sync.
3. Rollback drill is documented and acknowledged by operators.

When all preconditions are satisfied, proceed to deprecate/archive `clawd-graphiti` and mark the old runtime flow read-only.
