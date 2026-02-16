# PRD: Publicization Integration / E2E Cutover v1

## PRD Metadata
- Type: Execution
- Kanban Task: task-graphiti-publicization-integration-e2e
- Parent Epic: task-graphiti-publicization-upstream-sync
- Depends On: task-graphiti-public-boundary-contract, task-graphiti-public-refactor-pass, task-graphiti-public-history-migration, task-graphiti-upstream-sync-lane-v2, task-graphiti-pack-adapter-interface, task-graphiti-state-migration-kit, task-graphiti-public-docs-release-rails
- Preferred Engine: Either
- Owned Paths:
  - `prd/EXEC-PUBLICIZATION-INTEGRATION-E2E-v1.md`
  - `reports/publicization/integration-report.md`
  - `reports/publicization/integration-checklist.md`

## Overview
Run full end-to-end verification of the publicization program and produce a ship/no-ship recommendation.

## Mandatory Cross-Repo Baseline Review (to prevent narrow-pass regressions)
Before implementation, the agent must:
1. Review corresponding paths in `projects/graphiti` (private/source baseline) and `projects/graphiti-openclaw` (public target).
2. Produce a short cross-repo inventory in PR notes listing concrete files/directories reviewed in both repos.
3. Identify at least 3 candidate simplifications across the owned-path surface; implement selected items or explicitly defer each candidate with rationale.
4. If the PR touches only one file or one narrow function, include explicit justification for why broader owned-path opportunities were not applicable.

## Goals
- Prove that boundaries, migration choice, sync lane, adapter contract, and docs operate as one coherent system.
- Verify no private leakage and no regressions in exported foundations.
- Verify state migration tooling works in dry-run validation flow.
- Produce final readiness artifact for merge/cutover decision.

## Definition of Done (DoD)
**DoD checklist:**
- [x] Boundary audit strict mode evaluated. **Status: FAIL** (BLOCK: 4, AMBIGUOUS: 81).
- [x] Chosen migration path validated on public repo candidate. **Status: PASS** (`clean-foundation` won via `scripts/public_history_scorecard.py`).
- [x] Refactor validation suite covered by child PRDs. **Status: N/A** (child PRD execution artifacts and validations already recorded).
- [x] Upstream sync doctor dry-run passes. **Status: PASS** (clean worktree + diff report produced).
- [x] Adapter contract checker passes in strict mode. **Status: PASS** (`1` extension, `6` commands).
- [x] State migration kit dry-run export/check/import passes. **Status: PASS** (using dedicated output dir + `--allow-overwrite`).
- [x] Release docs checklist passes. **Status: PASS** (presence and gating language checks).
- [x] Integration report includes explicit GO/NO-GO recommendation and outstanding risks. **Status: PASS** (this PRD artifact set).

**Validation commands (run from repo root):**
```bash
set -euo pipefail
boundary_audit_rc=0
python3 scripts/public_repo_boundary_audit.py \
  --strict \
  --manifest config/public_export_allowlist.yaml \
  --denylist config/public_export_denylist.yaml \
  --report /tmp/boundary-audit.md || boundary_audit_rc=$?
upstream_doctor_rc=0
python3 scripts/upstream_sync_doctor.py --repo . --dry-run || upstream_doctor_rc=$?
upstream_sync_button_rc=0
python3 scripts/upstream_sync_doctor.py --repo . --check-sync-button-safety || upstream_sync_button_rc=$?
python3 scripts/extension_contract_check.py --strict
rm -rf /tmp/graphiti-state-export-clean
python3 scripts/state_migration_export.py --dry-run --out /tmp/graphiti-state-export-clean
python3 scripts/state_migration_check.py --package /tmp/graphiti-state-export-clean --dry-run
python3 scripts/state_migration_import.py --dry-run --allow-overwrite --in /tmp/graphiti-state-export-clean

test -s reports/publicization/history-scorecard.md
rg -n 'Winner: `clean-foundation`' reports/publicization/history-scorecard.md
test -s reports/publicization/integration-checklist.md
test -s reports/publicization/integration-report.md
rg -n 'Gate: \*\*NO-GO\*\*|Gate: \*\*GO\*\*' reports/publicization/integration-checklist.md
rg -n 'Decision: \*\*NO-GO\*\*|Decision: \*\*GO\*\*' reports/publicization/integration-report.md
rg -n 'BLOCK: [0-9]+' reports/publicization/integration-report.md
! rg -n '<PR head SHA at gate execution>' reports/publicization/integration-checklist.md reports/publicization/integration-report.md
rg -n 'Evaluated commit: `[0-9a-f]{7,40}`' reports/publicization/integration-checklist.md reports/publicization/integration-report.md
rg -n 'Gate return codes: `boundary_audit_rc=[0-9]+`, `upstream_doctor_rc=[0-9]+`, `upstream_sync_button_rc=[0-9]+`' reports/publicization/integration-checklist.md reports/publicization/integration-report.md
rg -n 'Generated \(UTC\): [0-9]{4}-[0-9]{2}-[0-9]{2}T' reports/publicization/integration-checklist.md reports/publicization/integration-report.md
test -s docs/public/README.md
test -s docs/public/SECURITY-BOUNDARIES.md
test -s docs/public/RELEASE-CHECKLIST.md
test -s docs/public/WHAT-NOT-TO-MIGRATE.md
test -s docs/runbooks/state-migration.md
test -s docs/runbooks/upstream-sync-openclaw.md
test -s docs/public/BOUNDARY-CONTRACT.md
test -s docs/public/MIGRATION-SYNC-TOOLKIT.md
test -s /tmp/boundary-audit.md
printf 'boundary_audit_rc=%s upstream_doctor_rc=%s upstream_sync_button_rc=%s\n' \
  "$boundary_audit_rc" "$upstream_doctor_rc" "$upstream_sync_button_rc"
case "$boundary_audit_rc" in 0|1) ;; *) echo "Unexpected boundary_audit_rc=$boundary_audit_rc (expected 0|1)" >&2; exit 2 ;; esac
case "$upstream_doctor_rc" in 0|1) ;; *) echo "Unexpected upstream_doctor_rc=$upstream_doctor_rc (expected 0|1)" >&2; exit 2 ;; esac
case "$upstream_sync_button_rc" in 0|1) ;; *) echo "Unexpected upstream_sync_button_rc=$upstream_sync_button_rc (expected 0|1)" >&2; exit 2 ;; esac
rg -n "content marketing|public write-up|deferred|gate" docs/public/RELEASE-CHECKLIST.md
```
**Pass criteria:** command sequence executes and produces artifacts; expected policy outcomes are captured as 0/1 gate return codes; unexpected runtime/infrastructure failures abort immediately; final recommendation is GO only if no unresolved CRITICAL/HIGH blockers remain, otherwise NO-GO with explicit remediation.

## Cross-repo baseline review (completed)

### Cross-repo inventory
- Private/source baseline reviewed (`projects/graphiti`): `.github/workflows/ci.yml`, `.github/workflows/security-review.yml`, `docs/runbooks/content_workflow_data_independent_gate.md`, `scripts/runtime_pack_router.py`, `config/runtime_pack_registry.yaml`, `prd/EXEC-PUBLIC-*` and `prd/EXEC-UPSTREAM-SYNC-LANE-v1.md`.
- Public target reviewed (`projects/graphiti-openclaw`): `.github/workflows/upstream-sync.yml`, `.github/workflows/migration-sync-tooling.yml`, `docs/public/BOUNDARY-CONTRACT.md`, `docs/public/MIGRATION-SYNC-TOOLKIT.md`, `scripts/public_repo_boundary_audit.py`, `scripts/upstream_sync_doctor.py`, `scripts/public_history_export.py`.
- This repo reviewed for gate coherence: `scripts/`, `docs/public/`, `docs/runbooks/`, `extensions/`, `reports/publicization/`.

### Simplification candidates (selected / deferred)
1. **Selected:** keep integration orchestration as a documented sequence of existing scripts (no new wrapper script or framework) to preserve determinism.
2. **Selected:** enforce a single source of evidence for history-migration choice by reading `reports/publicization/history-scorecard.md` in both gate table and GO/NO-GO report.
3. **Selected:** standardize temp-path handling for state migration (`/tmp/graphiti-state-export-clean`) and explicit `--allow-overwrite` to avoid false negatives in dry-run import validation.
4. **Deferred:** full automation of a pre-merge canary script (`scripts/publicization_integration_report.py`) because this gate is intended to stay code-light and artifact-driven.

## User Stories

### US-001: Single truth for readiness
**Description:** As operator, I want one final report that tells me if publicization is safe to execute.

**Acceptance Criteria:**
- [ ] Report has clear GO/NO-GO status.
- [ ] Report lists blockers with owners and remediation.

### US-002: Regression confidence
**Description:** As maintainer, I want confidence that simplification and migration didn’t break foundations.

**Acceptance Criteria:**
- [ ] Core tests pass.
- [ ] Sync dry-run and boundary checks pass in same run.

## Functional Requirements
- FR-1: Integration report must aggregate outputs from all child PRDs.
- FR-2: Any failing gate must automatically set NO-GO.
- FR-3: Report format must be concise and reproducible.
- FR-4: Any code touched in this PRD must follow the mandatory simplification loop from the epic.

## Non-Goals (Out of Scope)
- Public content marketing draft.
- Post-cutover feature roadmap.

## Technical Considerations
- Keep integration scripts deterministic and CI-friendly.
- Minimize custom orchestration; prefer invoking existing child validators.

## Execution Plan (Serial vs Parallel)
### Critical path (serial)
1. Pull child artifacts/reports.
2. Execute integrated validation sequence.
3. Generate GO/NO-GO report.
4. Prepare cutover recommendation.

### Parallel workstreams (if any)
- None (integration is final serial gate).

### Dependency map
- Hard depends on all prior child PRDs.

## Open Questions
- Should GO require one extra manual diff review in `graphiti-openclaw` before merge?
