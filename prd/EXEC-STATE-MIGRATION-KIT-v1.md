# PRD: Runtime State Migration Kit (Graph History + Ingest Registry) v1

## PRD Metadata
- Type: Execution
- Kanban Task: task-graphiti-state-migration-kit
- Parent Epic: task-graphiti-publicization-upstream-sync
- Depends On: task-graphiti-public-history-migration, task-graphiti-pack-adapter-interface
- Preferred Engine: Either
- Owned Paths:
  - `prd/EXEC-STATE-MIGRATION-KIT-v1.md`
  - `scripts/state_migration_export.py` (new)
  - `scripts/state_migration_import.py` (new)
  - `scripts/state_migration_check.py` (new)
  - `docs/runbooks/state-migration.md` (new)

## Overview
Provide a clean migration kit for runtime data so users can move Graphiti state between environments without committing private state into git.

## Mandatory Cross-Repo Baseline Review (to prevent narrow-pass regressions)
Before implementation, the agent must:
1. Review corresponding paths in `projects/graphiti` (private/source baseline) and `projects/graphiti-openclaw` (public target).
2. Produce a short cross-repo inventory in PR notes listing concrete files/directories reviewed in both repos.
3. Identify at least 3 candidate simplifications across the owned-path surface; implement selected items or explicitly defer each candidate with rationale.
4. If the PR touches only one file or one narrow function, include explicit justification for why broader owned-path opportunities were not applicable.

## Goals
- Support deterministic export/import of required runtime state.
- Keep migration flow simple, documented, and reversible.
- Preserve privacy controls and avoid accidental data leakage.

## Definition of Done (DoD)
**DoD checklist:**
- [ ] Export tool generates package for required runtime state (graph history, ingest registry, minimal metadata).
- [ ] Import tool restores package idempotently with validation checks.
- [ ] Migration check script verifies package integrity and target compatibility.
- [ ] Runbook documents migration flow, rollback, and security handling.
- [ ] Refactor-pass simplification loop is applied to all touched code.

**Validation commands (run from repo root):**
```bash
set -euo pipefail
python3 scripts/state_migration_export.py --dry-run --out /tmp/graphiti-state-export
python3 scripts/state_migration_check.py --package /tmp/graphiti-state-export --dry-run
python3 scripts/state_migration_import.py --dry-run --in /tmp/graphiti-state-export
python3 scripts/run_tests.py --target migration
```
**Pass criteria:** all commands exit 0; dry-run reports no schema/compatibility errors.

## User Stories

### US-001: Clean environment migration
**Description:** As operator, I want to move runtime graph state into a new setup cleanly.

**Acceptance Criteria:**
- [ ] Export/import flow works without manual DB surgery.
- [ ] Migration preserves required references and integrity checks pass.

### US-002: Private data safety
**Description:** As maintainer, I want migration data handled explicitly so no private runtime state leaks into git.

**Acceptance Criteria:**
- [ ] Migration artifacts are explicitly excluded from git in docs/checks.
- [ ] Runbook includes secure handling/deletion guidance.

## Functional Requirements
- FR-1: Export package format is versioned and documented.
- FR-2: Import path validates source version and fails safely on incompatibility.
- FR-3: Tools support dry-run mode for preflight.
- FR-4: Migration checker reports missing components and remediation steps.
- FR-5: Runbook includes rollback steps and post-migration verification.
- FR-6: Any code touched in this PRD must follow the mandatory simplification loop from the epic.

## Non-Goals (Out of Scope)
- Real-time replication.
- Automatic cloud backup policy.
- Migrating private workflow/content packs themselves.

## Technical Considerations
- Keep package structure simple and inspectable.
- Prefer deterministic manifests/checksums over ad-hoc copy scripts.
- Avoid over-generalized migration framework.

## Execution Plan (Serial vs Parallel)
### Critical path (serial)
1. Define state package schema + versioning.
2. Implement export script.
3. Implement import script.
4. Implement migration checker.
5. Document runbook and validate dry-runs.

### Parallel workstreams (if any)
- Runbook drafting can run in parallel after schema is frozen.

### Dependency map
- Depends on migration baseline choice and adapter contract.
- Feeds integration E2E cutover gate.

## Open Questions
- Should export support optional redaction mode for sharing minimal reproducible state snapshots?
