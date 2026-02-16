# PRD: Graphiti Publicization + Upstream Sync Simplification Program

## PRD Metadata
- Type: Epic
- Kanban Task: task-graphiti-publicization-upstream-sync
- Preferred Engine: Either

## Overview
Build a safe, repeatable path to publish a generalized Graphiti foundation in `yhl999/graphiti-openclaw` (forked from `zep/graphiti`) while keeping the codebase elegant and low-maintenance.

This program explicitly excludes private workflow/content packs and any personal/private artifacts.

## Goals
- Publish only reusable foundations (ingest/runtime/policy/evals scaffolding) to the public fork.
- Preserve no private data or private project logic in public history.
- Keep the resulting public codebase simpler than current private state (lower maintenance burden).
- Establish an upstream sync lane with predictable conflict handling.
- Define a clean extension model so persona-specific workflow/content packs can attach via stable adapters.
- Define clean migration rails for runtime data (`graph history`, `ingest registry`, related state) without baking private data into the public repo.
- Defer public narrative/content-marketing draft until ingest backlog clears and content workflows are fully operational.

## Program-wide Agent Rule (mandatory)
For any child PRD that touches code, implementation agents must run the following simplification loop:
1. Review changes and identify simplification opportunities.
2. Refactor to remove dead code/paths, straighten logic, remove excessive parameters, and remove premature optimization.
3. Run build/tests to verify behavior.
4. Suggest optional abstractions only if they clearly improve clarity.

## Cross-Repo Review Gate (mandatory)
For any child PRD that touches code, the implementation agent must review and compare both repositories before coding:
- `projects/graphiti` (private/source baseline)
- `projects/graphiti-openclaw` (public target)

PR notes must include a cross-repo inventory (paths reviewed), at least 3 simplification candidates, and explicit deferrals for any unselected candidates.
Single-file/narrow-function refactors require explicit justification for why broader owned-path opportunities were not applicable.

## Global Definition of Done (DoD)
**Global DoD checklist:**
- [ ] Public export boundary is codified as an allowlist and enforced by audit tooling.
- [ ] Public repo candidate passes secret/privacy/history scans with no CRITICAL findings.
- [ ] Refactor pass is complete on exported foundations (dead paths removed, logic simplified, no behavioral regressions).
- [ ] Migration strategy is selected by scorecard (filtered-history rewrite vs clean-slate foundation history) with explicit rationale.
- [ ] Upstream sync lane is operational (branch strategy + runbook + dry-run verification).
- [ ] Public docs/release rails are complete and explicitly mark excluded private packs.
- [ ] Adapter interface contract is implemented/documented so private workflow/content packs can plug into the public foundation without patching core.
- [ ] State migration kit can export/import required runtime data cleanly across setups (graph history + ingest registry + related metadata).
- [ ] Content marketing/public write-up drafting remains gated OFF until workflow-readiness gate is met.

**Global validation commands:**
```bash
set -euo pipefail
python3 scripts/public_repo_boundary_audit.py --strict --manifest config/public_export_allowlist.yaml
python3 scripts/public_repo_history_scan.py --repo . --strict
python3 scripts/run_tests.py
python3 scripts/upstream_sync_doctor.py --repo . --dry-run
python3 scripts/extension_contract_check.py --strict
python3 scripts/state_migration_check.py --dry-run
test -s reports/publicization/integration-report.md
```
**Pass criteria:** all commands exit 0; no CRITICAL/HIGH privacy leaks; integration report generated.

## Child PRDs (Execution Units)
- task-graphiti-public-boundary-contract — Public foundation boundary contract + allowlist tooling — Owned paths: `config/public_export_allowlist.yaml`, `scripts/public_repo_boundary_audit.py`, `docs/public/BOUNDARY-CONTRACT.md`
- task-graphiti-public-refactor-pass — Simplicity-first refactor pass on public foundations — Owned paths: `ingest/**`, `runtime/**`, `scripts/**`, `evals/**` (foundation-only scope)
- task-graphiti-public-history-migration — History migration + cutover scorecard (filtered-history vs clean-slate) — Owned paths: `scripts/public_history_export.py`, `scripts/public_history_scorecard.py`, `docs/public/HISTORY-MIGRATION.md`
- task-graphiti-upstream-sync-lane-v2 — Upstream sync lane and conflict runbook — Owned paths: `docs/runbooks/upstream-sync-openclaw.md`, `scripts/upstream_sync_doctor.py`, `.github/workflows/upstream-sync.yml`
- task-graphiti-pack-adapter-interface — Extension adapter/interface contract for workflow/content packs — Owned paths: `docs/public/EXTENSION-INTERFACE.md`, `scripts/extension_contract_check.py`, `extensions/**`
- task-graphiti-state-migration-kit — Runtime state migration kit (graph history + ingest registry + metadata) — Owned paths: `scripts/state_migration_export.py`, `scripts/state_migration_import.py`, `scripts/state_migration_check.py`, `docs/runbooks/state-migration.md`
- task-graphiti-public-docs-release-rails — Public docs + release rails + content-draft gate — Owned paths: `docs/public/README.md`, `docs/public/SECURITY-BOUNDARIES.md`, `docs/public/RELEASE-CHECKLIST.md`
- task-graphiti-publicization-integration-e2e — Integration/E2E cutover verification — Owned paths: `reports/publicization/integration-report.md`, `prd/EXEC-PUBLICIZATION-INTEGRATION-E2E-v1.md`

## Child PRD DAG (machine-readable)
```yaml
antfarm_plan:
  max_concurrency: 2
  children:
    - id: task-graphiti-public-boundary-contract
      depends_on: []
    - id: task-graphiti-public-refactor-pass
      depends_on: [task-graphiti-public-boundary-contract]
    - id: task-graphiti-public-history-migration
      depends_on: [task-graphiti-public-boundary-contract, task-graphiti-public-refactor-pass]
    - id: task-graphiti-upstream-sync-lane-v2
      depends_on: [task-graphiti-public-history-migration]
    - id: task-graphiti-pack-adapter-interface
      depends_on: [task-graphiti-public-boundary-contract, task-graphiti-public-refactor-pass]
    - id: task-graphiti-state-migration-kit
      depends_on: [task-graphiti-public-history-migration, task-graphiti-pack-adapter-interface]
    - id: task-graphiti-public-docs-release-rails
      depends_on: [task-graphiti-public-boundary-contract, task-graphiti-public-refactor-pass, task-graphiti-pack-adapter-interface]
    - id: task-graphiti-publicization-integration-e2e
      depends_on:
        - task-graphiti-public-boundary-contract
        - task-graphiti-public-refactor-pass
        - task-graphiti-public-history-migration
        - task-graphiti-upstream-sync-lane-v2
        - task-graphiti-pack-adapter-interface
        - task-graphiti-state-migration-kit
        - task-graphiti-public-docs-release-rails
```

## Parallelization / Merge Strategy
- Start with boundary contract first (serial gate).
- Then run up to two branches in parallel (max 2):
  - refactor pass,
  - adapter interface contract.
- History migration starts only after refactor pass completes (to avoid rewriting history twice).
- State migration kit starts only after migration baseline and adapter contract are both settled.
- Upstream sync lane opens after migration branch choice is finalized.
- Docs/release rails follow boundary + refactor + adapter outputs.
- Integration/E2E PRD lands last and validates the full cutover.
- Open PRs as draft early; only mark ready when local validation commands pass.

## Dependencies / Integration Notes
- Source private repo: `yhl999-org/clawd-graphiti`
- Target public repo: `yhl999/graphiti-openclaw`
- Upstream source: `zep/graphiti`
- Public architecture is intentionally two-layer:
  - Layer 1: general public foundation repo.
  - Layer 2: optional private overlays (persona/workflow/content packs) via adapters/interfaces.
- "Sync fork" button is allowed only for trivial fast-forward syncs; non-trivial sync remains PR-based via `upstream-sync/*` lane.

## Locked Decisions (2026-02-15)
- **R1=C:** Program split = Epic + 5+ execution PRDs.
- **R2=B with safety valve:** primary plan is filtered-history migration; fallback to clean-slate foundation history if scorecard shows lower complexity/risk.
- **R3=C:** public export uses allowlist-only boundary.
- **R4=C:** upstream sync model = dedicated sync lane + periodic PRs.
- **R5=C:** two-pass refactor (light pre-migration cleanup + post-migration simplicity pass).
- **R6:** optimize for elegance/simplicity over preserving every private-era implementation detail.
- **R7:** content marketing draft is deferred until content workflows are fully set and ingest backlog gate is cleared.
- **R8:** upstream sync cadence default = **weekly Monday ET**.
- **R9:** refactor-pass prompt is mandatory for any implementation agent touching code in this epic.
- **R10:** migration decision rule = try filtered-history first; if filtered-history scores <80/100 overall or any unresolved HIGH finding remains after one remediation pass, switch to clean-slate foundation history.
- **R11:** final public repo must be general-purpose and consumable by non-VC users; persona-specific packs remain external overlays.
- **R12:** runtime data migration (graph history/registry) is supported by explicit migration tooling, not by checking private state into git.
- **R13:** public core must run with **zero extensions installed** (no private overlay dependency at boot/runtime baseline).
- **R14:** extension API surface must stay minimal and versioned (`api_version`); no ad-hoc private hooks in core.
- **R15:** strict one-way layering: overlays may depend on core, core may not import overlays.
- **R16:** one canonical config model + schema; no hidden defaults spread across multiple codepaths.
- **R17:** state migration is a first-class product surface (versioned format, checksums, dry-run, rollback, compatibility checks).
- **R18:** CI must include migration round-trip validation (export -> import -> integrity check) as a release gate.
- **R19:** complexity budget gate for foundation PRs: reduce complexity by default; any increase requires explicit one-line rationale.
- **R20:** provide a single operator preflight entrypoint for health/audit/sync/migration checks to minimize tribal runbook drift.
- **R21:** one-path-per-capability rule: exactly one blessed path each for ingest, sync, migration, and extension loading in core (no parallel legacy paths in steady state).
- **R22:** feature-flag expiry rule: temporary flags must include owner + expiry date; expired flags are removed, not accumulated.
- **R23:** dependency budget rule: each new runtime dependency requires explicit rationale + owner + review of standard-library/simple-script alternative.
- **R24:** extension contract semver rule: breaking adapter changes require a compatibility window + migration note.
- **R25:** state/data compatibility rule: schema changes must ship with migrator and rollback path, or they do not ship.
- **R26:** idempotency rule: ingest/sync/migration operations must be safe to re-run without manual cleanup.
- **R27:** deprecation hygiene rule: deprecated modules must have a removal milestone/date at introduction.
- **R28:** architectural change discipline: new core layers/abstractions require short ADR with alternatives and removal criteria.
- **R29:** release gate rule: docs/runbooks are executable artifacts; broken commands fail CI.
- **R30:** boundary enforcement in CI: cross-layer import violations (core -> overlays) fail CI automatically.
- **R31:** golden-path test gate: fast baseline suite (boot, ingest, sync, migration, extension-load) must pass on every release PR.
- **R32:** no cross-repo symlink rule for PRDs/config/docs; use copy/sync generation instead (portable clones, no dangling external paths).
- **R33:** canonical-doc mirroring rule: public repo may contain mirrored PRDs/docs, but source-of-truth remains explicit and mirror process is scripted.
- **R34:** cross-contract invariant rule: contract validation must enforce relationships across policy/manifest/contract-policy artifacts (not schema-only per file).
- **R35:** command-surface safety rule: command bus fails fast on extension registry warnings unless explicit local override is passed.
- **R36:** atomic state import rule: migration import defaults to rollback-capable atomic behavior; non-atomic mode is explicit and debug-only.
- **R37:** migration handler rule: contract-policy targets must map to explicit migrator handlers with version gates.

## Constraint -> Enforcement Map
- **R13, R15, R30** -> `task-graphiti-pack-adapter-interface` + `task-graphiti-publicization-integration-e2e` (`extension_contract_check.py --strict` + layer import check in CI).
- **R14, R24** -> `task-graphiti-pack-adapter-interface` (versioned contract + compatibility checks in `extension_contract_check.py`).
- **R16, R21, R22** -> `task-graphiti-public-refactor-pass` + `task-graphiti-public-docs-release-rails` (single config schema + flag policy documented and linted).
- **R17, R18, R25, R26, R31** -> `task-graphiti-state-migration-kit` + `task-graphiti-publicization-integration-e2e` (`state_migration_export/import/check` dry-run + round-trip gate).
- **R19, R23, R28** -> `task-graphiti-public-refactor-pass` (complexity delta note + dependency rationale + ADR requirement in PR template/checklist).
- **R20, R29** -> `task-graphiti-upstream-sync-lane-v2` + `task-graphiti-public-docs-release-rails` (single preflight entrypoint + executable runbook validation).
- **R32, R33** -> `task-graphiti-public-docs-release-rails` + `task-graphiti-upstream-sync-lane-v2` (scripted docs/PRD mirror, no symlink policy check).

## Open Questions
- None at epic level.
