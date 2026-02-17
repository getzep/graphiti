# PRD: QMD Inversion Repo-Link Hardening v1

## PRD Metadata
- Type: Execution
- Kanban Task: task-graphiti-publicization-qmd-inversion-repo-link
- Parent Epic: task-graphiti-publicization-upstream-sync
- Depends On: task-graphiti-publicization-cron-cutover
- Preferred Engine: Codex
- Owned Paths:
  - `prd/EXEC-PUBLICIZATION-QMD-INVERSION-REPO-LINK-v1.md`
  - `scripts/runtime_memory_backend_switch.py`
  - `scripts/runtime_memory_backend_status.py`
  - `docs/MEMORY-RUNTIME-WIRING.md`
  - `tools/graphiti` (workspace symlink target policy documented in runbook)

## Overview
Ensure the one-click QMD↔Graphiti inversion path executes against the canonical OpenClaw Graphiti repo (public framework checkout with private overlay applied), with no legacy `clawd-graphiti` path dependency.

## Objectives
- Hard-bind switch/status scripts to canonical repo-root resolution and fail loudly on legacy path targets.
- Document canonical runtime target (`projects/graphiti-openclaw-runtime`) and private overlay apply flow.
- Add dry-run verification steps for both directions (`qmd_primary` and `graphiti_primary`) against canonical runtime checkout.

## Definition of Done
- [ ] `runtime_memory_backend_switch.py` and `runtime_memory_backend_status.py` contain no hardcoded or implicit `projects/graphiti` / `clawd-graphiti` path dependency.
- [ ] `docs/MEMORY-RUNTIME-WIRING.md` includes canonical runtime checkout + private overlay apply flow.
- [ ] `tools/graphiti` symlink policy is documented and validated.
- [ ] Both commands succeed in dry-run against canonical runtime checkout:
  - `--target graphiti_primary --dry-run`
  - `--target qmd_primary --dry-run`

## Validation
```bash
set -euo pipefail

# ensure no legacy repo references remain in switch/status/docs
! rg -n "clawd-graphiti|projects/graphiti\b" \
  scripts/runtime_memory_backend_switch.py \
  scripts/runtime_memory_backend_status.py \
  docs/MEMORY-RUNTIME-WIRING.md

# verify canonical runtime checkout exists
test -d /Users/archibald/clawd/projects/graphiti-openclaw-runtime/.git

# verify tool symlink target
test "$(readlink /Users/archibald/clawd/tools/graphiti)" = "../projects/graphiti-openclaw-runtime"

# backend switch dry-runs on canonical runtime checkout
python3 scripts/runtime_memory_backend_switch.py --repo /Users/archibald/clawd/projects/graphiti-openclaw-runtime --target graphiti_primary --dry-run >/tmp/qmd-inversion-graphiti.json
python3 scripts/runtime_memory_backend_switch.py --repo /Users/archibald/clawd/projects/graphiti-openclaw-runtime --target qmd_primary --dry-run >/tmp/qmd-inversion-qmd.json

test -s /tmp/qmd-inversion-graphiti.json
test -s /tmp/qmd-inversion-qmd.json
```

## Risks / Notes
- This PRD should not introduce private workflow content into public repo.
- Operational private wiring remains in `graphiti-openclaw-private` and is applied as overlay before runtime switching.
