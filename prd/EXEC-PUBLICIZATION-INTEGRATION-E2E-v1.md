# PRD: Publicization Integration / E2E Cutover v1

## PRD Metadata
- Type: Execution
- Kanban Task: task-graphiti-publicization-integration-e2e
- Parent Epic: task-graphiti-publicization-upstream-sync
- Depends On: task-graphiti-public-boundary-contract, task-graphiti-public-refactor-pass, task-graphiti-public-history-migration, task-graphiti-upstream-sync-lane-v2, task-graphiti-pack-adapter-interface, task-graphiti-state-migration-kit, task-graphiti-public-docs-release-rails
- Preferred Engine: Either
- Owned Paths:
  - `prd/EXEC-PUBLICIZATION-INTEGRATION-E2E-v1.md`
  - `reports/publicization/integration-report.md` (new)
  - `reports/publicization/integration-checklist.md` (new)

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
- [ ] Boundary audit passes in strict mode.
- [ ] Chosen migration path validated on public repo candidate.
- [ ] Refactor validation suite passes.
- [ ] Upstream sync doctor dry-run passes.
- [ ] Adapter contract checker passes in strict mode.
- [ ] State migration kit dry-run export/check/import passes.
- [ ] Release docs checklist passes.
- [ ] Integration report includes explicit GO/NO-GO recommendation and outstanding risks.

**Validation commands (run from repo root):**
```bash
set -euo pipefail
python3 scripts/public_repo_boundary_audit.py --strict --manifest config/public_export_allowlist.yaml
python3 scripts/public_repo_history_scan.py --repo . --strict
python3 scripts/run_tests.py
python3 scripts/upstream_sync_doctor.py --repo . --dry-run
python3 scripts/extension_contract_check.py --strict
python3 scripts/state_migration_export.py --dry-run --out /tmp/graphiti-state-export
python3 scripts/state_migration_check.py --package /tmp/graphiti-state-export --dry-run
python3 scripts/state_migration_import.py --dry-run --in /tmp/graphiti-state-export
python3 scripts/publicization_integration_report.py --out reports/publicization/integration-report.md

test -s reports/publicization/integration-report.md
```
**Pass criteria:** all checks exit 0; report exists with no unresolved CRITICAL/HIGH blockers.

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
