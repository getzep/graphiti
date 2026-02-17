# Cron Cutover Plan Template

This file is a public template for staged cron/runtime cutover.

## Stage 0 — Preconditions
- DB backup wiring dry-run passed
- runtime checkout synced from public main
- private overlay applied
- rollback owner assigned

## Stage 1 — Shadow mode
- run ingest trigger and worker in dry-run mode
- run workflow shadow compare report
- capture status snapshots
- define go/no-go threshold (error rate, coverage, drift)

## Stage 2 — Controlled enablement
- enable incremental ingest lane for bounded window
- monitor queue, latency, failures
- keep backend profile on `qmd_primary` unless explicit switch approved

## Stage 3 — Optional backend flip
- dry-run `graphiti_primary`
- execute flip only if guardrails pass
- monitor and compare against shadow baseline

## Rollback checklist
- switch backend to `qmd_primary`
- pause cron lane
- retain logs/evidence for postmortem
- reopen cutover review

## Legacy deprecation gate
Deprecate/archive legacy `clawd-graphiti` only after:
- runtime checkout validated on `graphiti-openclaw-runtime`
- private overlay repo is source of truth for operational wiring
- one full clean cutover cycle with no rollback required
