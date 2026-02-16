# PRD: Public Foundation Refactor Pass (Simplicity-First) v1

## PRD Metadata
- Type: Execution
- Kanban Task: task-graphiti-public-refactor-pass
- Parent Epic: task-graphiti-publicization-upstream-sync
- Depends On: task-graphiti-public-boundary-contract
- Preferred Engine: Either
- Owned Paths:
  - `.github/workflows-archive/README.md`
  - `.github/workflows-archive/lint.yml`
  - `.github/workflows-archive/typecheck.yml`
  - `.github/workflows-archive/unit_tests.yml`
  - `.github/workflows/ci.yml`
  - `.github/workflows/lint.yml`
  - `.github/workflows/migration-sync-tooling.yml`
  - `README.md`
  - `config/delta_contract_policy.json`
  - `config/migration_sync_policy.json`
  - `config/public_export_allowlist.yaml`
  - `config/state_migration_manifest.json`
  - `docs/public/BOUNDARY-CONTRACT.md`
  - `docs/public/MIGRATION-SYNC-TOOLKIT.md`
  - `extensions/migration_sync/README.md`
  - `extensions/migration_sync/manifest.json`
  - `prd/EPIC-PUBLICIZATION-UPSTREAM-SYNC-SIMPLIFICATION-v1.md`
  - `prd/EXEC-PUBLIC-REFACTOR-PASS-SIMPLIFY-v1.md`
  - `scripts/ci/run_db_integration_tests.sh`
  - `scripts/ci/run_migration_sync_toolkit.sh`
  - `scripts/ci/run_pyright.sh`
  - `scripts/ci/run_ruff_lint.sh`
  - `scripts/ci/run_unit_tests.sh`
  - `scripts/delta_contract_check.py`
  - `scripts/delta_contract_migrate.py`
  - `scripts/delta_contracts.py`
  - `scripts/delta_contracts_lib/__init__.py`
  - `scripts/delta_contracts_lib/common.py`
  - `scripts/delta_contracts_lib/contract_policy.py`
  - `scripts/delta_contracts_lib/extension.py`
  - `scripts/delta_contracts_lib/inspect.py`
  - `scripts/delta_contracts_lib/package_manifest.py`
  - `scripts/delta_contracts_lib/policy.py`
  - `scripts/delta_contracts_lib/state_manifest.py`
  - `scripts/delta_tool.py`
  - `scripts/extension_contract_check.py`
  - `scripts/migration_sync_lib.py`
  - `scripts/public_boundary_policy.py`
  - `scripts/public_boundary_policy_lint.py`
  - `scripts/public_history_export.py`
  - `scripts/public_history_scorecard.py`
  - `scripts/public_repo_boundary_audit.py`
  - `scripts/state_migration_check.py`
  - `scripts/state_migration_export.py`
  - `scripts/state_migration_import.py`
  - `scripts/upstream_sync_doctor.py`
  - `tests/test_delta_contract_check.py`
  - `tests/test_delta_contract_migrate.py`
  - `tests/test_delta_pipeline_e2e.py`
  - `tests/test_delta_tool.py`
  - `tests/test_extension_contract_check.py`
  - `tests/test_public_boundary_policy.py`
  - `tests/test_public_boundary_policy_lint.py`
  - `tests/test_public_history_export.py`
  - `tests/test_public_history_scorecard.py`
  - `tests/test_public_repo_boundary_audit.py`
  - `tests/test_state_migration_kit.py`
  - `tests/test_upstream_sync_doctor.py`

## Overview
Perform a targeted refactor pass to reduce complexity before migration so the public codebase is elegant, minimal, and easier to maintain.

## Workflow Lock (from user prompt)
Apply this exact flow:
1. Review recent changes and identify simplification opportunities.
2. Refactor to remove dead code/paths, straighten logic, remove excessive parameters, remove premature optimization.
3. Run build/tests to verify behavior.
4. Suggest optional abstractions only if they clearly improve clarity.

## Mandatory Cross-Repo Baseline Review (to prevent narrow-pass regressions)
Before implementation, the agent must:
1. Review corresponding paths in `projects/graphiti` (private/source baseline) and `projects/graphiti-openclaw` (public target).
2. Produce a short cross-repo inventory in PR notes listing concrete files/directories reviewed in both repos.
3. Identify at least 3 candidate simplifications across the owned-path surface; implement selected items or explicitly defer each candidate with rationale.
4. If the PR touches only one file or one narrow function, include explicit justification for why broader owned-path opportunities were not applicable.

## Goals
- Reduce maintenance burden by deleting dead/duplicative pathways.
- Simplify control flow and interfaces in foundation modules.
- Preserve externally observable behavior and policy safety.

## Definition of Done (DoD)
**DoD checklist:**
- [ ] Dead code and obsolete branches removed from owned paths.
- [ ] Public-facing interfaces reduced/simplified where safe.
- [ ] Refactor notes include before/after rationale by module.
- [ ] Regression tests/build checks pass.
- [ ] No new complexity introduced under "optional abstraction" banner.

**Validation commands (run from repo root):**
```bash
set -euo pipefail
bash scripts/ci/run_ruff_lint.sh
bash scripts/ci/run_migration_sync_toolkit.sh
python3 tests/test_delta_contract_check.py
python3 tests/test_delta_contract_migrate.py
python3 tests/test_delta_tool.py
python3 tests/test_delta_pipeline_e2e.py
python3 tests/test_state_migration_kit.py
```
**Pass criteria:** all checks exit 0; no functionality regressions; contract/migration invariants preserved.

## User Stories

### US-001: Elegant core modules
**Description:** As maintainer, I want core modules to be readable and minimal so future changes are cheaper.

**Acceptance Criteria:**
- [ ] Cyclomatic hotspots identified and reduced in touched modules.
- [ ] Deleted lines exceed added lines for pure simplification PRs (unless safety tests require net additions).

### US-002: Stable behavior
**Description:** As operator, I want refactor improvements without changing functional outcomes.

**Acceptance Criteria:**
- [ ] Existing golden/eval checks still pass.
- [ ] Policy boundary behavior remains unchanged.

## Functional Requirements
- FR-1: Refactor scope is restricted to allowlisted foundation paths.
- FR-2: Any removed parameter must have callsite cleanup in same PR.
- FR-3: Any retained complexity must have explicit justification in PR notes.
- FR-4: No speculative framework rewrites.

## Non-Goals (Out of Scope)
- Adding new capabilities.
- Refactoring private workflow/content packs.
- Public narrative drafting.

## Technical Considerations
- Prefer small, composable functions over generic abstraction layers.
- Keep data contracts explicit; avoid implicit global state.
- Favor deletion over indirection.

## Execution Plan (Serial vs Parallel)
### Critical path (serial)
1. Produce module complexity inventory.
2. Apply low-risk simplifications.
3. Apply interface simplifications with callsite updates.
4. Run full validations and summarize deltas.

### Parallel workstreams (if any)
- Independent module refactors may run in parallel only if owned paths do not overlap.

### Dependency map
- Depends on boundary contract to define allowed foundation scope.
- Feeds history migration (avoid rewriting history multiple times).

## Execution Amendment (2026-02-15) — Delta Control-Layer Tranches 1→7

### Why this amendment exists
The implementation stream for this PRD converged on the **delta/control layer** as the highest-leverage simplification surface.
This amendment captures the shipped scope and the new steady-state contract model.

### Effective owned-path focus (override for this execution)
- `scripts/**` (delta tooling + shared contract/migration libraries)
- `config/**` (policy + contract metadata)
- `docs/public/**` (operator-facing runbooks/docs)
- `.github/workflows/**` (canonical CI gates + legacy archive policy)
- `extensions/**` (versioned extension command contract)
- `tests/**` (delta contract, migration, and pipeline coverage)

### Shipped tranche outcomes
1. **Boundary modularization + CI centralization**
   - Boundary policy library + lint CLI + shared CI shell entrypoints.
2. **Migration/sync toolkit + extension contracts baseline**
   - Upstream doctor, history export/scorecard, state export/check/import.
3. **Hardening + policy-driven metrics**
   - Safer path handling, payload integrity checks, policyized history metrics.
4. **Contract-first layer + unified command surface**
   - `delta_contract_check.py` + `delta_tool.py` canonical entrypoint.
5. **Registry/integrity dedupe + pipeline e2e**
   - Shared extension inspection + shared payload evaluation + e2e test.
6. **Modular contracts + explicit migration policy + workflow retirement**
   - `delta_contracts_lib/**`, `delta_contract_migrate.py`, workflow archive policy.
7. **Cross-contract invariants + safer execution semantics**
   - Invariant checks across policy/manifest/contract policy,
   - target-aware migration handlers,
   - fail-fast command execution on registry warnings,
   - atomic import rollback default.

### Current design contract (steady state)
- Exactly one canonical command bus: `scripts/delta_tool.py`
- Exactly one canonical contract validator entrypoint: `scripts/delta_contract_check.py`
- Exactly one canonical contract migration entrypoint: `scripts/delta_contract_migrate.py`
- Extension commands must be namespaced and versioned via `command_contract`
- State import defaults to atomic rollback semantics (`--atomic` default)

### Delta-specific validation commands (canonical)
```bash
set -euo pipefail
bash scripts/ci/run_ruff_lint.sh
bash scripts/ci/run_migration_sync_toolkit.sh
python3 tests/test_delta_contract_check.py
python3 tests/test_delta_contract_migrate.py
python3 tests/test_delta_tool.py
python3 tests/test_delta_pipeline_e2e.py
python3 tests/test_state_migration_kit.py
```

### Outcome note
This PRD execution now functions as the implementation record for the tranche-based delta refactor line.
Any future refactor should preserve these invariants unless an explicit ADR supersedes them.

## Open Questions
- Should we set an explicit complexity budget target per module (e.g., max function length / branch count)?
