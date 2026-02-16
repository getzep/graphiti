# Upstream Sync Runbook (graphiti-openclaw)

## Purpose
Maintain `yhl999/graphiti-openclaw` against `getzep/graphiti` through a deterministic, reviewable PR lane.

- **Default cadence:** every Monday (America/New_York) via scheduled GitHub Action.
- **Manual mode:** `workflow_dispatch` is always available.
- **Default policy:** open/update PRs from `upstream-sync/YYYY-MM-DD` into `main`.

## Remote Strategy

Canonical remotes:

```bash
git remote -v
# origin   https://github.com/yhl999/graphiti-openclaw.git
# upstream https://github.com/getzep/graphiti.git
```

If `upstream` is missing:

```bash
git remote add upstream https://github.com/getzep/graphiti.git
git fetch upstream main --prune
```

## Branch Strategy

- Sync branches are date-stamped and immutable in meaning:
  - `upstream-sync/YYYY-MM-DD`
- Branches are always reset from `origin/main` by automation:
  - checkout/reset base: `origin/main`
  - merge source: `upstream/main`
- PR target is always `main`.

This keeps each sync attempt auditable and narrow:
- one date branch,
- one merge intent,
- one review surface.

## Workflow Policy

Workflow file: `.github/workflows/upstream-sync.yml`

Triggers:
- `schedule: 0 14 * * 1` (Monday 14:00 UTC, policy anchor for Monday ET cadence)
- `workflow_dispatch` (manual trigger)

Manual inputs:
- `branch_date` (optional, `YYYY-MM-DD`, interpreted as ET naming)
- `dry_run` (optional bool; skips push/PR mutation)

## Conflict Policy

### Fast path (clean merge)
1. Create `upstream-sync/YYYY-MM-DD` from `origin/main`.
2. Merge `upstream/main` with `--no-ff --no-edit`.
3. Push branch.
4. Open/update PR into `main`.

### Conflict path
If merge conflicts occur:
1. Workflow fails fast and does **not** push a conflicted branch.
2. Operator creates the same branch locally and resolves conflicts explicitly.
3. Operator pushes resolved branch and reuses/open PR.

Known hotspot policy:
- `signatures/version1/cla.json` is fork-owned CLA state; resolve with `--ours` unless there is an explicit maintainer decision to import upstream content for that file.

Recommended local conflict flow:

```bash
set -euo pipefail
git fetch origin main --prune
git fetch upstream main --prune
git checkout -B upstream-sync/$(TZ=America/New_York date +%F) origin/main
git merge --no-ff upstream/main
# resolve conflicts (example hotspot):
git checkout --ours signatures/version1/cla.json
git add signatures/version1/cla.json
# resolve any remaining conflicts, then:
git add -A
git commit
git push --set-upstream origin HEAD
gh pr create --base main --head "$(git branch --show-current)"
```

## GitHub "Sync fork" Button Policy (STRICT)

Use `python3 scripts/upstream_sync_doctor.py --repo . --check-sync-button-safety`.

### ALLOW only when all are true
- working tree is clean,
- `origin/main` has **0** origin-only commits vs `upstream/main`,
- `upstream/main` is ahead (there is upstream delta to sync).

### DENY in all other cases
If any condition fails, **do not use Sync button**.
Use PR lane (`upstream-sync/*`) instead.

Why strict:
- this fork carries local delta/control-layer commits,
- PR lane preserves review, rollback, and conflict visibility,
- Sync button bypasses branch-level review discipline.

## Rollback / Recovery

### A) Bad sync PR before merge
- Close PR.
- Delete branch (`upstream-sync/*`).
- Re-run workflow or trigger manual sync after fix.

### B) Bad sync merged to `main`
1. Identify merge commit SHA.
2. Revert merge commit:

```bash
git checkout main
git pull --ff-only origin main
git revert -m 1 <merge_commit_sha>
git push origin main
```

3. Open follow-up PR for root-cause correction.

### C) Catastrophic rollback (last known good)
- Restore `main` from a known good SHA only with maintainer approval.
- Prefer revert over force-push whenever possible.

## Dry-Run Cycle Evidence (2026-02-16)

Validation commands (repo root):

```bash
python3 scripts/upstream_sync_doctor.py --repo . --dry-run
python3 scripts/upstream_sync_doctor.py --repo . --check-sync-button-safety
```

Observed behavior:
- dry-run preflight exits successfully with explicit sync-button decision output,
- sync-button safety command emits deterministic **ALLOW/DENY** decision,
- when decision is DENY, operator is redirected to PR lane (`upstream-sync/*`).
