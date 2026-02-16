# PRD: Publicization Runtime Data + Backup Wiring v1

## PRD Metadata
- Type: Execution
- Kanban Task: task-graphiti-publicization-db-backup-wiring
- Parent Epic: task-graphiti-publicization-upstream-sync
- Depends On: task-graphiti-state-migration-kit, task-graphiti-publicization-adapter-wiring
- Preferred Engine: Codex
- Owned Paths:
  - `prd/EXEC-PUBLICIZATION-DATA-BACKUP-WIRING-v1.md`
  - `.gitignore`
  - `scripts/backup.sh`
  - `scripts/snapshot_create.py`
  - `scripts/snapshot_restore_test.py`
  - `docs/runbooks/publicization-backup-cutover.md`
  - `reports/publicization/backup-readiness.md`

## Overview
Wire the existing migration/export and snapshot backup paths into one reproducible pre-cutover data safety lane.

## Goals
- Ensure migration kit + snapshot backup can be executed as one operator playbook.
- Provide explicit backup readiness evidence before production cutover.
- Keep backup flow simple, deterministic, and reversible.
- Ensure private/runtime artifacts are explicitly gitignored in this repo so private data can live here safely without accidental commits.
- Record migration-readiness for deprecating/archiving legacy `clawd-graphiti` once safety gates are complete.

## Definition of Done (DoD)
**DoD checklist:**
- [ ] Backup runbook defines exact pre-cutover sequence (export/check/import-dry-run + snapshot + restore test).
- [ ] Backup readiness report records latest dry-run evidence and blocking conditions.
- [ ] Snapshot scripts support runbook flow without undocumented manual steps.
- [ ] Rollback guidance is explicit and includes decision points.
- [ ] `.gitignore` explicitly covers private/runtime artifacts required for the migrated private surfaces.
- [ ] Backup readiness report includes archive-readiness notes for deprecating old `clawd-graphiti` after cutover gates pass.

**Validation commands (run from repo root):**
```bash
set -euo pipefail
rm -rf /tmp/graphiti-state-export-clean
python3 scripts/state_migration_export.py --dry-run --out /tmp/graphiti-state-export-clean
python3 scripts/state_migration_check.py --package /tmp/graphiti-state-export-clean --dry-run
python3 scripts/state_migration_import.py --dry-run --allow-overwrite --in /tmp/graphiti-state-export-clean

python3 scripts/snapshot_create.py --help >/tmp/snapshot-create-help.txt
python3 scripts/snapshot_restore_test.py --help >/tmp/snapshot-restore-help.txt
python3 -m compileall scripts/snapshot_create.py scripts/snapshot_restore_test.py

test -s /tmp/snapshot-create-help.txt
test -s /tmp/snapshot-restore-help.txt
test -s docs/runbooks/publicization-backup-cutover.md
test -s reports/publicization/backup-readiness.md
rg -n "^backup/|^state/|^exports/|^reports/|^logs/" .gitignore
rg -n "archive|deprecat|clawd-graphiti" reports/publicization/backup-readiness.md
```
**Pass criteria:** all commands exit 0; migration dry-run is clean; backup runbook/readiness artifacts exist and are internally consistent; `.gitignore` covers private/runtime artifacts; archive-readiness notes are present.

## User Stories

### US-001: Cutover data safety
**Description:** As operator, I want one explicit backup play before cutover so rollback is practical, not theoretical.

**Acceptance Criteria:**
- [ ] Pre-cutover sequence has no ambiguous steps.
- [ ] Rollback entry point is documented and tested at procedure level.

### US-002: Evidence-based readiness
**Description:** As reviewer, I want concrete backup evidence in the repo before approving cutover.

**Acceptance Criteria:**
- [ ] Backup readiness report includes latest run timestamps and outcomes.
- [ ] Report explicitly marks GO/NO-GO for backup gate.

## Functional Requirements
- FR-1: Migration dry-run and snapshot lifecycle must be documented as one sequence.
- FR-2: Runbook must include required env vars and secrets handling boundaries.
- FR-3: Readiness report must include blockers, owner, and remediation path.
- FR-4: Backup guidance must stay compatible with existing encrypted snapshot policy.
- FR-5: `.gitignore` policy must explicitly guard private/runtime artifacts that are required to exist locally after migration.
- FR-6: Readiness artifacts must include explicit deprecation/archive preconditions for the legacy `clawd-graphiti` repo.

## Non-Goals (Out of Scope)
- Changing encryption technology or key-management strategy.
- Implementing continuous replication.

## Technical Considerations
- Keep shell entrypoint (`backup.sh`) as canonical operator interface.
- Do not add hidden defaults for recipients/identities.
- Prefer explicit fail-fast errors for missing prerequisites.

## Execution Plan (Serial vs Parallel)
### Critical path (serial)
1. Align runbook with current scripts.
2. Add backup-readiness evidence report.
3. Validate migration dry-run + snapshot CLI surfaces.
4. Final consistency pass across runbook/report.

### Parallel workstreams (if any)
- Script help-text ergonomics and runbook drafting can run in parallel.

### Dependency map
- Depends on state migration kit.
- Feeds cron cutover readiness and final GO gate.

## Open Questions
- Should backup readiness report be regenerated by CI on every PR touching migration/backup paths?
