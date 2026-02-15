# PRD: Public History Migration + Cutover Scorecard v1

## PRD Metadata
- Type: Execution
- Kanban Task: task-graphiti-public-history-migration
- Parent Epic: task-graphiti-publicization-upstream-sync
- Depends On: task-graphiti-public-boundary-contract, task-graphiti-public-refactor-pass
- Preferred Engine: Either
- Owned Paths:
  - `prd/EXEC-PUBLIC-HISTORY-MIGRATION-CUTOVER-v1.md`
  - `scripts/public_history_export.py` (new)
  - `scripts/public_history_scorecard.py` (new)
  - `docs/public/HISTORY-MIGRATION.md` (new)
  - `reports/publicization/history-scorecard.md` (new)

## Overview
Generate and compare two migration candidates for `yhl999/graphiti-openclaw`:
1) filtered-history rewrite (preserve useful foundation history),
2) clean-slate foundation history (minimal curated history).

Select the winner using a scorecard biased toward elegance/simplicity and safety.

## Goals
- Preserve useful provenance where practical.
- Avoid leaking private files/history.
- Choose the simpler long-term maintenance baseline.

## Definition of Done (DoD)
**DoD checklist:**
- [ ] `public_history_export.py` can produce both migration candidates reproducibly.
- [ ] Scorecard compares candidates on privacy risk, complexity, and maintainability.
- [ ] Decision recorded with explicit rationale and rollback plan.
- [ ] Chosen candidate is prepared on a public-repo branch ready for review.

**Validation commands (run from repo root):**
```bash
set -euo pipefail
python3 scripts/public_history_export.py --mode filtered-history --dry-run
python3 scripts/public_history_export.py --mode clean-foundation --dry-run
python3 scripts/public_history_scorecard.py \
  --filtered-report reports/publicization/filtered-history.md \
  --clean-report reports/publicization/clean-foundation.md \
  --out reports/publicization/history-scorecard.md

test -s reports/publicization/history-scorecard.md
```
**Pass criteria:** both dry-runs succeed; scorecard generated; final choice documented.

## User Stories

### US-001: Safe migration choice
**Description:** As maintainer, I want objective migration criteria so we pick the safest and cleanest history path.

**Acceptance Criteria:**
- [ ] Scorecard includes leak-risk checks for both options.
- [ ] Final recommendation includes quantified tradeoffs.

### US-002: Simplicity-weighted outcome
**Description:** As owner, I want elegance prioritized over historical completeness.

**Acceptance Criteria:**
- [ ] Scorecard uses weighted criteria with maintainability > history preservation.
- [ ] If filtered-history complexity exceeds threshold, clean-slate is selected automatically.

## Functional Requirements
- FR-1: Candidate A (filtered-history) must apply allowlist/denylist filters to commit history.
- FR-2: Candidate B (clean-foundation) must reconstruct minimal commit baseline from approved foundation snapshot.
- FR-3: Scorecard dimensions: privacy risk, code simplicity, future merge conflict risk, auditability.
- FR-4: Decision threshold rule is explicit and machine-checkable.
- FR-5: Decision report includes branch names and exact cutover commands.
- FR-6: Any code touched in this PRD must follow the mandatory simplification loop from the epic.

## Non-Goals (Out of Scope)
- Upstream sync automation itself.
- Post-cutover feature development.

## Technical Considerations
- Prefer deterministic scripting over manual git surgery.
- Keep sensitive author metadata handling explicit in migration docs.
- Ensure force-push safety protocol is documented.

## Execution Plan (Serial vs Parallel)
### Critical path (serial)
1. Implement export script for both modes.
2. Run dry-run exports.
3. Generate scorecard and select winner.
4. Prepare cutover branch and decision doc.

### Parallel workstreams (if any)
- Scorecard template can be drafted while export script is implemented.

### Dependency map
- Requires boundary contract and refactor pass output.
- Unblocks upstream sync lane.

## Locked Decision (2026-02-15)
- Cutover threshold rule: choose clean-slate if filtered-history scores <80/100 overall, or if any unresolved HIGH finding remains after one remediation pass.

## Open Questions
- None.
