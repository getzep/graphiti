# PRD: Publicization Cron Cutover v1

## PRD Metadata
- Type: Execution
- Kanban Task: task-graphiti-publicization-cron-cutover
- Parent Epic: task-graphiti-publicization-upstream-sync
- Depends On: task-graphiti-publicization-db-backup-wiring
- Preferred Engine: Codex
- Owned Paths:
  - `prd/EXEC-PUBLICIZATION-CRON-CUTOVER-v1.md`
  - `.gitignore`
  - `ingest/queue.py`
  - `scripts/run_incremental_ingest.py`
  - `scripts/ingest_trigger_done.py`
  - `scripts/mcp_ingest_sessions.py`
  - `scripts/registry_status.py`
  - `scripts/runtime_memory_backend_switch.py`
  - `scripts/runtime_memory_backend_status.py`
  - `scripts/workflow_pack_shadow_compare.py`
  - `config/runtime_memory_backend_profiles.json`
  - `docs/MEMORY-RUNTIME-WIRING.md`
  - `reports/publicization/cron-cutover-plan.md`
  - `evals/cases/synthetic.workflow_pack_shadow_compare.json`
  - `ingest/`

## Overview
Ship a deterministic, reversible cron/runtime cutover lane in the public framework while keeping environment-specific operational values in private overlay.

## Definition of Done
- [ ] Incremental ingest trigger/worker scripts exist and run in dry-run mode.
- [ ] Runtime backend switch/status scripts support both `qmd_primary` and `graphiti_primary` dry-run operations.
- [ ] Cron cutover plan template exists with staged enablement and rollback checklist.
- [ ] No legacy `clawd-graphiti` path dependency remains in cutover scripts/docs.
- [ ] Runtime validation works against canonical checkout linked by `tools/graphiti`.

## Validation commands (run from repo root)
```bash
set -euo pipefail

cd /Users/archibald/clawd/projects/graphiti-openclaw-integration-e2e

rm -f /tmp/cron-cutover-ingest.db

python3 scripts/ingest_trigger_done.py \
  --db-path /tmp/cron-cutover-ingest.db \
  --source done \
  --session-key dryrun-session \
  --ts 2026-02-16T18:00:00Z \
  --dry-run

python3 scripts/run_incremental_ingest.py \
  --db-path /tmp/cron-cutover-ingest.db \
  --once --dry-run --no-schedule
python3 scripts/registry_status.py --db-path /tmp/cron-cutover-ingest.db --json >/tmp/registry-status.json

python3 scripts/runtime_memory_backend_status.py >/tmp/backend-status.json
python3 scripts/runtime_memory_backend_switch.py --target graphiti_primary --dry-run >/tmp/backend-switch-graphiti.json
python3 scripts/runtime_memory_backend_switch.py --target qmd_primary --dry-run >/tmp/backend-switch-qmd.json

python3 scripts/workflow_pack_shadow_compare.py \
  --cases evals/cases/synthetic.workflow_pack_shadow_compare.json \
  --out /tmp/workflow-pack-shadow-compare.md

test -s /tmp/registry-status.json
test -s /tmp/backend-status.json
test -s /tmp/backend-switch-graphiti.json
test -s /tmp/backend-switch-qmd.json
test -s /tmp/workflow-pack-shadow-compare.md
test -s reports/publicization/cron-cutover-plan.md

! rg -n "clawd-graphiti|projects/clawd-graphiti" \
  scripts/runtime_memory_backend_switch.py \
  scripts/runtime_memory_backend_status.py \
  scripts/run_incremental_ingest.py \
  scripts/ingest_trigger_done.py \
  docs/MEMORY-RUNTIME-WIRING.md
```
