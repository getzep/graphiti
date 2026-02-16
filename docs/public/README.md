# Graphiti OpenClaw Public Foundation

Graphiti OpenClaw is a **public foundation fork** of Graphiti. It preserves reusable core primitives and adds a narrow publicization control layer for repeatable releases.

## Scope and non-goals

### Scope
- Generalizable ingestion and memory runtime foundations.
- Deterministic boundary governance for what can be exported.
- Migration/sync tooling and release gates owned by this repo.
- Minimal documentation required to keep public releases predictable.

### Non-goals
- Personal workflow packs, private examples, or content-marketing workflows.
- Private secrets, credentials, state snapshots, or local operator artifacts.
- Product feature development beyond publication, boundary, and release rails.

## Foundation substrate vs private overlays

This repo is built as a foundation layer. It is expected that private teams layer on additional packs and workflows.

- **Foundation substrate (public):** `graphiti_core/**`, `mcp_server/**`, `server/**`, `tests/**`.
- **Private overlays (do not migrate):** workflow packs, content packs, private runbooks, and private operational state.

## Security and release rails

Operational boundaries and release requirements live here:

- `SECURITY-BOUNDARIES.md` — threat-oriented boundary model, what is blocked, and why.
- `RELEASE-CHECKLIST.md` — mandatory pre-release gate sequence.
- `WHAT-NOT-TO-MIGRATE.md` — explicit exclusions before public cutover.

## Quick setup

1. Read `SECURITY-BOUNDARIES.md` and confirm boundary policy status.
2. Run through `RELEASE-CHECKLIST.md` before any public release operation.
3. Keep private overlays and examples outside the foundation paths listed above.

## Operating rule

Everything that does not belong in a stable, reusable foundation should be treated as private layering and handled as externalized workflow/context input instead of being merged into this public core.
