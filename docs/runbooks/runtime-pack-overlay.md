# Runtime Pack Overlay (Public/Private Split)

## Model
- **Public repo (`graphiti-openclaw`)**: generic router/tooling + example configs.
- **Private overlay repo (`graphiti-openclaw-private`)**: real consumer profiles, pack registry, workflow packs, operational reports, runtime hardening rails.

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
- `config/cron_ingest_schedule.yaml`
- `config/cron_canary_jobs.json`
- `workflows/*.pack.yaml`
- `reports/publicization/backup-readiness.md`
- private runtime scripts/truth modules (`scripts/*.py`, `truth/*.py`, `schemas/*.sql`)

Then apply overlay into your runtime working tree before running cron/runtime operations.

## Canonical runtime workflow

```bash
python3 /path/to/graphiti-openclaw-private/scripts/rebuild_runtime.py \
  --runtime-repo /path/to/graphiti-openclaw-runtime \
  --preserve-state
```

What this does:
1. Verifies private `overlay-manifest.json` checksums.
2. Resets runtime checkout to public `origin/main`.
3. Re-applies private overlay files.
4. Enforces drift guard (no unexpected runtime drift outside overlay manifest surface).

## Guardrails
- Do not commit private workflow names, private task mappings, or operational readiness evidence to public repo.
- Do not use cross-repo symlinks for PRDs/config/docs.
- Treat runtime checkout as disposable and reproducible-by-command.

## Related runbooks
- `docs/runbooks/upstream-sync-openclaw.md`
- `docs/runbooks/state-migration.md`
- `docs/runbooks/publicization-backup-cutover.md`
