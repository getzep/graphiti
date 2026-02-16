# PRD: Extension Adapter / Interface Contract (Workflow + Content Packs) v1

## PRD Metadata
- Type: Execution
- Kanban Task: task-graphiti-pack-adapter-interface
- Parent Epic: task-graphiti-publicization-upstream-sync
- Depends On: task-graphiti-public-boundary-contract, task-graphiti-public-refactor-pass
- Preferred Engine: Either
- Owned Paths:
  - `prd/EXEC-PACK-ADAPTER-INTERFACE-CONTRACT-v1.md`
  - `docs/public/EXTENSION-INTERFACE.md`
  - `extensions/__init__.py`
  - `extensions/contracts.py`
  - `extensions/loader.py`
  - `extensions/migration_sync/manifest.json`
  - `scripts/extension_contract_check.py`

## Overview
Define a stable extension interface so private/custom workflow and content packs can plug into the public core without modifying core foundation code.

## Mandatory Cross-Repo Baseline Review (to prevent narrow-pass regressions)
Before implementation, the agent must:
1. Review corresponding paths in `projects/graphiti` (private/source baseline) and `projects/graphiti-openclaw` (public target).
2. Produce a short cross-repo inventory in PR notes listing concrete files/directories reviewed in both repos.
3. Identify at least 3 candidate simplifications across the owned-path surface; implement selected items or explicitly defer each candidate with rationale.
4. If the PR touches only one file or one narrow function, include explicit justification for why broader owned-path opportunities were not applicable.

## Cross-Repo Baseline Inventory (completed)
### Private/source baseline reviewed (`projects/graphiti`)
- `config/runtime_pack_registry.yaml`
- `scripts/runtime_pack_router.py`
- `schemas/runtime_pack_router.schema.json`
- `exports/manifest.json`

### Public target reviewed (`projects/graphiti-openclaw`)
- `extensions/migration_sync/manifest.json`
- `docs/public/MIGRATION-SYNC-TOOLKIT.md`
- `scripts/extension_contract_check.py`
- `scripts/delta_contracts_lib/extension.py`
- `scripts/delta_contracts_lib/inspect.py`
- `prd/EXEC-PACK-ADAPTER-INTERFACE-CONTRACT-v1.md`

## Simplification Loop (candidates + rationale)
| Candidate | Decision | Rationale |
|---|---|---|
| Centralize contract parsing/validation in a dedicated `extensions/contracts.py` model (instead of ad-hoc checks inside checker script) | **Selected** | Reduces duplication, gives one stable versioned contract surface, and makes future API version expansion easier. |
| Move extension discovery/loading into a fail-safe loader (`extensions/loader.py`) and keep checker as a thin CLI | **Selected** | Improves maintainability and isolates failures per extension while preserving clear diagnostics. |
| Enforce `api_version` strictly for every existing manifest immediately | **Selected (incremental)** | Added explicit `api_version: 1` to the existing `migration_sync` manifest to remove compatibility-noise while preserving the fallback logic for external legacy packs. |
| Add YAML manifest support in v1 | **Deferred** | JSON-only keeps parser complexity low and deterministic for CI; YAML can be added in a later contract version if needed. |

## Goals
- Keep public core generic and clean.
- Allow private/persona overlays to attach through explicit contracts.
- Prevent custom packs from creating hidden coupling to internal core modules.

## Definition of Done (DoD)
**DoD checklist:**
- [x] Extension contract types/interfaces are defined and versioned.
- [x] Loader path supports registering external packs without editing core logic.
- [x] Contract checker validates extension compatibility and fails with clear diagnostics.
- [x] Public docs explain adapter model with examples and anti-patterns.
- [x] Refactor-pass simplification loop is applied to all touched code.

**Validation commands (run from repo root):**
```bash
set -euo pipefail
bash scripts/ci/run_ruff_lint.sh
python3 scripts/extension_contract_check.py --strict
python3 -m unittest tests/test_extension_contract_check.py
python3 -m compileall extensions scripts
```
**Pass criteria:** all commands exit 0; contract checker reports no compatibility violations.

**Validation note:** `scripts/run_tests.py` is not present in this repository, so the PRD now uses
repo-true commands that exercise extension contract behavior and CI lint parity directly.
Ruff is intentionally run first (before any Python commands that may import modules) to avoid local `__pycache__` side-effects in lint wrappers.

## User Stories

### US-001: Core stays generic
**Description:** As maintainer, I want custom packs to plug in externally so the core repo remains broadly useful.

**Acceptance Criteria:**
- [x] Core can run with zero custom packs installed.
- [x] Installing/removing a pack does not require editing core files.

### US-002: Overlay safety
**Description:** As operator, I want clear compatibility checks before enabling a pack.

**Acceptance Criteria:**
- [x] Incompatible pack versions fail fast with actionable error messages.
- [x] Contract checker output includes fix guidance.

## Functional Requirements
- FR-1: Contract versioning must be explicit (`api_version` or equivalent).
- FR-2: Extension registration must be declarative (config/manifest), not hardcoded.
- FR-3: Loader must isolate extension failures and preserve core startup diagnostics.
- FR-4: Contract checker must support `--strict` CI mode.
- FR-5: Docs must include a minimal extension template.
- FR-6: Any code touched in this PRD must follow the mandatory simplification loop from the epic.

## Non-Goals (Out of Scope)
- Migrating existing private packs into public repo.
- Building a marketplace/distribution system.

## Technical Considerations
- Keep interfaces small and explicit.
- Prefer composition hooks over inheritance-heavy plugin frameworks.
- Keep extension loading optional and fail-safe.

## Execution Plan (Serial vs Parallel)
### Critical path (serial)
1. Define extension contract surface.
2. Implement minimal loader + registration flow.
3. Implement contract checker.
4. Add docs and template.
5. Run validations.

### Parallel workstreams (if any)
- Docs/template drafting can run in parallel after contract surface is finalized.

### Dependency map
- Depends on boundary contract + refactor pass outputs.
- Feeds state migration kit and docs/release rails.

## Open Questions
- Should extension manifests live in YAML only, or support JSON as well?
  - **Decision for v1:** JSON manifest only (simpler parser + deterministic CI). YAML support deferred.
