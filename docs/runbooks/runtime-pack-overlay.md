# Runtime Pack Overlay (Public/Private Split)

## Model
- **Public repo (`graphiti-openclaw`)**: generic router/tooling + example configs.
- **Private overlay repo**: real consumer profiles, pack registry, workflow packs, operational reports.

## Why
This keeps reusable framework code open while preventing private workflow/content wiring from living in the public default branch.

## Public defaults
- `config/runtime_pack_registry.yaml` contains example pack entries.
- `config/runtime_consumer_profiles.yaml` contains example profile mappings.
- `workflows/example_*.pack.yaml` are example pack files.

## Runtime router contract (canonical)
The runtime router (`scripts/runtime_pack_router.py`) now supports both legacy YAML and JSON config names.
It resolves these config files in order:

- Registry: `config/runtime_pack_registry.json` → `config/runtime_pack_registry.yaml`
- Profiles: `config/runtime_consumer_profiles.json` → `config/runtime_consumer_profiles.yaml`

Router output includes:
- `selected_packs` and `dropped_packs` (deterministic routing decision)
- `decision_path` (auditable routing trace)
- `budget_summary`
- `injection_text` for prompt injection consumers (CLR/Antfarm wrappers)

### Multi-group retrieval matrix per pack
Each pack can declare retrieval topology under:

```json
"retrieval": {
  "group_ids_by_mode": {
    "default": ["s1_sessions_main", "s1_curated_refs"],
    "short": ["..."],
    "long": ["..."]
  },
  "chatgpt_lane": {
    "group_id": "s1_chatgpt_history",
    "allow_scoped": true,
    "allow_global": false
  }
}
```

At runtime, the router resolves `group_ids` from the active pack mode and applies chatgpt inclusion policy by profile `chatgpt_mode`.

### Scoped ChatGPT lane policy in code
Profiles may set:
- `chatgpt_mode = off|scoped|global`

Default is `scoped` when omitted.

Effective behavior is code-enforced at routing time:
- `off`: never include chatgpt lane
- `scoped`: include only if pack retrieval policy allows scoped
- `global`: include only if pack retrieval policy allows global

## Engineering learnings materialization
When `--materialize` is passed, the router can materialize engineering learnings into injection text.
Current source contract:
- `state/engineering/loops/clr_learnings.latest.jsonl`
- `state/engineering/loops/antfarm_learnings.latest.jsonl`

This allows CLR/Antfarm runtime injection even when engineering learnings are still converging in graph backends.

Canonical consumer pattern now includes both CLR-backed and classic Antfarm lanes (private overlay-defined profiles), e.g.:
- CLR/manual lanes: `clr_manual`, `clr_antfarm`
- `feature-dev-clr` roles: planner/developer/reviewer
- classic `feature-dev` roles: planner/developer/reviewer
- `bug-fix` roles: triage/fix
- `security-audit` roles: scan/fix

## Private overlay pattern
In your private repo, keep real versions of:
- `config/runtime_pack_registry.json`
- `config/runtime_consumer_profiles.json`
- `workflows/*.pack.yaml`
- `reports/publicization/backup-readiness.md`

Then copy/sync these files into your deployment working tree before running runtime routing.

## Guardrail
Do not commit private workflow names, private task mappings, or operational readiness evidence to the public repository.
