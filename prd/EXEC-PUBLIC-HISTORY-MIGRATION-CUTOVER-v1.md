# PRD: Public History Migration + Cutover Scorecard v1

## PRD Metadata
- Type: Execution
- Kanban Task: task-graphiti-public-history-migration
- Parent Epic: task-graphiti-publicization-upstream-sync
- Depends On: task-graphiti-public-boundary-contract, task-graphiti-public-refactor-pass
- Preferred Engine: Either
- Owned Paths:
  - `prd/EXEC-PUBLIC-HISTORY-MIGRATION-CUTOVER-v1.md`
  - `scripts/public_history_export.py`
  - `scripts/public_history_scorecard.py`
  - `docs/public/HISTORY-MIGRATION.md`
  - `reports/publicization/history-scorecard.md`

## Overview
Generate and compare two migration candidates for `yhl999/graphiti-openclaw`:
1) filtered-history rewrite (preserve useful foundation history),
2) clean-slate foundation history (minimal curated history).

Select the winner using a scorecard biased toward elegance/simplicity and safety.

## Mandatory Cross-Repo Baseline Review (to prevent narrow-pass regressions)
Before implementation, the agent must:
1. Review corresponding paths in `projects/graphiti` (private/source baseline) and `projects/graphiti-openclaw` (public target).
2. Produce a short cross-repo inventory in PR notes listing concrete files/directories reviewed in both repos.
3. Identify at least 3 candidate simplifications across the owned-path surface; implement selected items or explicitly defer each candidate with rationale.
4. If the PR touches only one file or one narrow function, include explicit justification for why broader owned-path opportunities were not applicable.

## Cross-Repo Inventory + Simplification Loop (2026-02-16)

### Cross-repo inventory reviewed
- `projects/graphiti/prd/EXEC-PUBLIC-HISTORY-MIGRATION-CUTOVER-v1.md`
- `projects/graphiti-openclaw/prd/EXEC-PUBLIC-HISTORY-MIGRATION-CUTOVER-v1.md`
- `projects/graphiti-openclaw/scripts/public_history_export.py`
- `projects/graphiti-openclaw/scripts/public_history_scorecard.py`
- `projects/graphiti-openclaw/config/public_export_allowlist.yaml`
- `projects/graphiti-openclaw/config/public_export_denylist.yaml`
- `projects/graphiti-openclaw/config/migration_sync_policy.json`

### Candidate simplifications (selected/deferred)
1. **Selected:** remove required `--report` arg in `public_history_export.py`; use deterministic default output paths by mode.
2. **Selected:** keep backward compatibility in `public_history_scorecard.py` while making report-driven inputs first-class (`--filtered-report`, `--clean-report`).
3. **Selected:** auto-write sibling JSON summaries from export to avoid duplicated/manual wiring between export and scorecard steps.
4. **Selected:** make threshold/HIGH fallback decision rule explicit in scorecard output and docs with branch + rollback commands.
5. **Deferred:** full automated filtered-history rewrite engine (out of scope for this execution PRD; would add unnecessary framework complexity).

## Goals
- Preserve useful provenance where practical.
- Avoid leaking private files/history.
- Choose the simpler long-term maintenance baseline.

## Definition of Done (DoD)
**DoD checklist:**
- [x] `public_history_export.py` can produce both migration candidates reproducibly.
- [x] Scorecard compares candidates on privacy risk, complexity, and maintainability.
- [x] Decision recorded with explicit rationale and rollback plan.
- [x] Chosen candidate is prepared on a public-repo branch ready for review.

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
- [x] Scorecard includes leak-risk checks for both options.
- [x] Final recommendation includes quantified tradeoffs.

### US-002: Simplicity-weighted outcome
**Description:** As owner, I want elegance prioritized over historical completeness.

**Acceptance Criteria:**
- [x] Scorecard uses weighted criteria with maintainability > history preservation.
- [x] If filtered-history complexity exceeds threshold, clean-slate is selected automatically.

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

## Validation Outcomes (2026-02-16)

All required commands passed from repo root:

```bash
python3 scripts/public_history_export.py --mode filtered-history --dry-run
# Report written: reports/publicization/filtered-history.md
# Metrics: privacy=26 simplicity=42 merge_conflict=54 auditability=49

python3 scripts/public_history_export.py --mode clean-foundation --dry-run
# Report written: reports/publicization/clean-foundation.md
# Metrics: privacy=97 simplicity=96 merge_conflict=92 auditability=90

python3 scripts/public_history_scorecard.py \
  --filtered-report reports/publicization/filtered-history.md \
  --clean-report reports/publicization/clean-foundation.md \
  --out reports/publicization/history-scorecard.md
# Decision: clean-foundation (Filtered-history has unresolved HIGH risk after one remediation pass.)

test -s reports/publicization/history-scorecard.md
# pass
```

## Final Decision Snapshot
- Winner: `clean-foundation`
- Winner branch: `cutover/clean-foundation`
- Rollback baseline SHA: `7f1b83f7c5749a3f68dce147f5c29e8fdf7b2840`

## Open Questions
- None.
