# Memory Runtime Wiring

This document defines the runtime backend switch contract for memory retrieval.

## Runtime model

Two backend profiles are supported:

- `qmd_primary` (production default)
- `graphiti_primary` (operator opt-in)

Profiles are declared in:

- `config/runtime_memory_backend_profiles.json`

Current active state is stored in:

- `config/.runtime_memory_backend_state.json`

## Operator commands

Run from repository root.

```bash
python3 scripts/runtime_memory_backend_status.py
python3 scripts/runtime_memory_backend_switch.py --target graphiti_primary --dry-run
python3 scripts/runtime_memory_backend_switch.py --target qmd_primary --dry-run
python3 scripts/runtime_memory_backend_switch.py --target graphiti_primary --execute
python3 scripts/runtime_memory_backend_switch.py --revert --execute
```

## Guardrails

- group-safe gating must stay enabled in all active profiles
- shadow compare should remain enabled during cutover
- one-command revert must always be available after a switch

## Public/private split

Public repo contains generic switch/status framework and example/default profile config.

Private operational overlays may replace profile values at deploy time (for environment-specific behavior), but should not change the switch contract surface.

## Security considerations

The ingest pipeline processes raw session transcripts, memory files, and conversation
data that inherently contain personally identifiable information (PII).  This is by
design — the pipeline's purpose is to extract knowledge from these sources.

**Directories that contain PII at runtime** (all gitignored):

- `evidence/` — parsed evidence documents
- `state/` — ingest registry DB, queue state
- `logs/` — worker execution logs

**Input validation:** All user-supplied identifiers (`group_id`, `session_key`,
`source`) are validated against a strict allowlist pattern
(`[A-Za-z0-9][A-Za-z0-9._:@-]{0,254}`) before use in subprocess arguments or
database keys.  See `ingest/queue.py:validate_identifier()`.

**Error handling:** Worker error messages use structured tags (`error_type:ClassName`)
rather than raw exception messages to avoid leaking internal state.

**Subprocess execution:** All subprocess calls use list-form arguments (never
`shell=True`), preventing shell injection even if an identifier were to bypass
validation.

## Canonical runtime checkout

Operational runtime should execute from the canonical runtime checkout linked by:

- `tools/graphiti -> ../projects/graphiti-openclaw-runtime`

Apply private overlay before operations:

```bash
/Users/archibald/clawd/projects/graphiti-openclaw-private/scripts/apply-overlay.sh \
  /Users/archibald/clawd/projects/graphiti-openclaw-runtime
```
