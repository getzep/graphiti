# PRD: Upstream Sync Lane (zep/graphiti → graphiti-openclaw) v1

## PRD Metadata
- Type: Execution
- Kanban Task: task-graphiti-upstream-sync-lane-v2
- Parent Epic: task-graphiti-publicization-upstream-sync
- Depends On: task-graphiti-public-history-migration
- Preferred Engine: Either
- Owned Paths:
  - `prd/EXEC-UPSTREAM-SYNC-LANE-v1.md`
  - `docs/runbooks/upstream-sync-openclaw.md`
  - `scripts/upstream_sync_doctor.py`
  - `.github/workflows/upstream-sync.yml`

## Overview
Establish a repeatable upstream sync lane for `yhl999/graphiti-openclaw` with controlled PR-based updates from `zep/graphiti`.

## Mandatory Cross-Repo Baseline Review (to prevent narrow-pass regressions)
Before implementation, the agent must:
1. Review corresponding paths in `projects/graphiti` (private/source baseline) and `projects/graphiti-openclaw` (public target).
2. Produce a short cross-repo inventory in PR notes listing concrete files/directories reviewed in both repos.
3. Identify at least 3 candidate simplifications across the owned-path surface; implement selected items or explicitly defer each candidate with rationale.
4. If the PR touches only one file or one narrow function, include explicit justification for why broader owned-path opportunities were not applicable.

## Cross-Repo Inventory + Simplification Loop (2026-02-16)

### Cross-repo inventory reviewed
- `projects/graphiti/.github/workflows/ci.yml`
- `projects/graphiti/.github/workflows/security-review.yml`
- `projects/graphiti/docs/runbooks/content_workflow_data_dependent_handoff.md`
- `projects/graphiti/docs/runbooks/content_workflow_data_independent_gate.md`
- `projects/graphiti-openclaw/prd/EXEC-UPSTREAM-SYNC-LANE-v1.md`
- `projects/graphiti-openclaw/scripts/upstream_sync_doctor.py`
- `projects/graphiti-openclaw/config/migration_sync_policy.json`
- `projects/graphiti-openclaw/.github/workflows/migration-sync-tooling.yml`
- `projects/graphiti-openclaw/docs/public/MIGRATION-SYNC-TOOLKIT.md`

### Candidate simplifications (selected/deferred)
1. **Selected:** centralize sync-button decision checks in a single helper (`_evaluate_sync_button_decision`) to keep policy logic deterministic and reviewable.
2. **Selected:** add explicit `Sync button decision: ALLOW|DENY` + rationale output so operators do not infer policy from raw counters.
3. **Selected:** keep sync automation as one workflow job with shell-native steps (no framework/action sprawl), with a single metadata step for branch/date/title.
4. **Deferred:** automatic conflict resolution heuristics in workflow; deferred intentionally because it obscures intent and increases blast radius.

## Goals
- Keep public fork current without destabilizing local foundations.
- Minimize merge pain via dedicated sync branches and checklists.
- Define when GitHub "Sync fork" button is acceptable versus when PR lane is mandatory.

## Definition of Done (DoD)
**DoD checklist:**
- [x] Runbook documents remotes, branch strategy, conflict policy, and rollback.
- [x] `upstream_sync_doctor.py` validates sync preconditions and reports blockers.
- [x] Sync workflow opens/updates PRs from `upstream-sync/*` branch.
- [x] "Sync fork" button policy is explicitly documented with strict allowed conditions.
- [x] One full dry-run sync cycle completes with no unresolved blockers.

**Validation commands (run from repo root):**
```bash
set -euo pipefail
python3 scripts/upstream_sync_doctor.py --repo . --dry-run
python3 scripts/upstream_sync_doctor.py --repo . --check-sync-button-safety

test -s docs/runbooks/upstream-sync-openclaw.md
```
**Pass criteria:** doctor exits 0 for dry-run path; runbook exists; sync-button check emits explicit allow/deny decision.

## User Stories

### US-001: Predictable upstream updates
**Description:** As maintainer, I want upstream updates to arrive via controlled PRs so conflicts are reviewable.

**Acceptance Criteria:**
- [x] Sync lane uses dedicated branch naming (`upstream-sync/YYYY-MM-DD` or equivalent).
- [x] PR template/checklist includes conflict hotspots and validation commands.

### US-002: Safe use of GitHub Sync button
**Description:** As operator, I want clear rules for when the GitHub UI sync shortcut is safe.

**Acceptance Criteria:**
- [x] Policy states sync button allowed only for trivial fast-forward cases with zero local divergence.
- [x] Non-trivial sync paths are blocked and redirected to PR lane.

## Functional Requirements
- FR-1: Sync lane must support periodic cadence (**weekly Monday ET** default) with manual trigger.
- FR-2: Sync process must preserve local foundation patches via explicit conflict strategy.
- FR-3: Doctor script must detect divergence and dirty preconditions.
- FR-4: Runbook must include disaster recovery (bad merge rollback).
- FR-5: Any code touched in this PRD must follow the mandatory simplification loop from the epic.
- FR-6: If PRDs/docs are mirrored into `graphiti-openclaw`, mirror must be script-driven copy/sync (no symlinks) and validated by doctor/preflight checks.

## Non-Goals (Out of Scope)
- Auto-merging upstream PRs without review.
- Syncing private repo directly with upstream.

## Technical Considerations
- Keep sync logic obvious and tool-light.
- Prefer branch+PR transparency over hidden magic.
- Use sync button only when doctor script says safe.

## Execution Plan (Serial vs Parallel)
### Critical path (serial)
1. Define runbook and policy.
2. Build doctor preflight checks.
3. Add workflow for periodic PR creation.
4. Execute dry-run and refine.

### Parallel workstreams (if any)
- Workflow YAML and runbook drafting can proceed in parallel after policy lock.

### Dependency map
- Depends on chosen migration baseline branch.
- Feeds integration cutover PRD.

## Locked Decisions (2026-02-16)
- Weekly sync cadence is fixed at Monday ET using schedule `0 14 * * 1` (UTC anchor), plus manual `workflow_dispatch` support.
- Branch naming is fixed to `upstream-sync/YYYY-MM-DD` (ET date).
- GitHub Sync button is allowlisted only for doctor-verified trivial/no-divergence cases; otherwise PR lane is mandatory.

## Validation Outcomes (2026-02-16)

PRD validation commands were executed from repo root.

```bash
python3 scripts/upstream_sync_doctor.py --repo . --dry-run
# Upstream sync doctor report
# Clean worktree: True
# Origin-only commits: 18
# Upstream-only commits: 5
# Sync button decision: DENY
# Exit: 0

python3 scripts/upstream_sync_doctor.py --repo . --check-sync-button-safety
# Upstream sync doctor report
# Clean worktree: True
# Origin-only commits: 18
# Upstream-only commits: 5
# Sync button decision: DENY
# Issues:
# - Sync button decision is DENY; use PR-based sync lane
# Exit: 1 (expected for deny path)

test -s docs/runbooks/upstream-sync-openclaw.md
# Exit: 0
```

Dry-run sync-cycle rehearsal (conflict-policy path) also completed:

```bash
# rehearsal branch from origin/main merged upstream/main
# conflict observed: signatures/version1/cla.json
# resolution applied per runbook: git checkout --ours signatures/version1/cla.json
# REHEARSAL_STATUS=complete
# REHEARSAL_MERGE_STATUS=conflict
# REHEARSAL_MERGE_SHA=8833f79be742ce83b6663068750191bcff250a44
```

Result: no unresolved conflict files after policy-based resolution; lane is operational with explicit conflict handling.

## Open Questions
- None.
