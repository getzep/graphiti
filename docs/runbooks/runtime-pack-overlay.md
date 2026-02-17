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

## Private overlay pattern
In your private repo, keep real versions of:
- `config/runtime_pack_registry.yaml`
- `config/runtime_consumer_profiles.yaml`
- `workflows/*.pack.yaml`
- `reports/publicization/backup-readiness.md`

Then copy/sync these files into your deployment working tree before running runtime routing.

## Guardrail
Do not commit private workflow names, private task mappings, or operational readiness evidence to the public repository.
