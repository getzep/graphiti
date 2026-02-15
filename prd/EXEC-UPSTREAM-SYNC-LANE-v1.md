# PRD: Upstream Sync Lane (zep/graphiti → graphiti-openclaw) v1

## PRD Metadata
- Type: Execution
- Kanban Task: task-graphiti-upstream-sync-lane-v2
- Parent Epic: task-graphiti-publicization-upstream-sync
- Depends On: task-graphiti-public-history-migration
- Preferred Engine: Either
- Owned Paths:
  - `prd/EXEC-UPSTREAM-SYNC-LANE-v1.md`
  - `docs/runbooks/upstream-sync-openclaw.md` (new)
  - `scripts/upstream_sync_doctor.py` (new)
  - `.github/workflows/upstream-sync.yml` (new)

## Overview
Establish a repeatable upstream sync lane for `yhl999/graphiti-openclaw` with controlled PR-based updates from `zep/graphiti`.

## Goals
- Keep public fork current without destabilizing local foundations.
- Minimize merge pain via dedicated sync branches and checklists.
- Define when GitHub "Sync fork" button is acceptable versus when PR lane is mandatory.

## Definition of Done (DoD)
**DoD checklist:**
- [ ] Runbook documents remotes, branch strategy, conflict policy, and rollback.
- [ ] `upstream_sync_doctor.py` validates sync preconditions and reports blockers.
- [ ] Sync workflow opens/updates PRs from `upstream-sync/*` branch.
- [ ] "Sync fork" button policy is explicitly documented with strict allowed conditions.
- [ ] One full dry-run sync cycle completes with no unresolved blockers.

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
- [ ] Sync lane uses dedicated branch naming (`upstream-sync/YYYY-MM-DD` or equivalent).
- [ ] PR template/checklist includes conflict hotspots and validation commands.

### US-002: Safe use of GitHub Sync button
**Description:** As operator, I want clear rules for when the GitHub UI sync shortcut is safe.

**Acceptance Criteria:**
- [ ] Policy states sync button allowed only for trivial fast-forward cases with zero local divergence.
- [ ] Non-trivial sync paths are blocked and redirected to PR lane.

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

## Open Questions
- Should the weekly sync run at a fixed local wall-clock time (e.g., Monday 09:00 ET) or be manually triggered each Monday until cadence confidence is established?
