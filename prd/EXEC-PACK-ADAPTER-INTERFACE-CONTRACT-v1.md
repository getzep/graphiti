# PRD: Extension Adapter / Interface Contract (Workflow + Content Packs) v1

## PRD Metadata
- Type: Execution
- Kanban Task: task-graphiti-pack-adapter-interface
- Parent Epic: task-graphiti-publicization-upstream-sync
- Depends On: task-graphiti-public-boundary-contract, task-graphiti-public-refactor-pass
- Preferred Engine: Either
- Owned Paths:
  - `prd/EXEC-PACK-ADAPTER-INTERFACE-CONTRACT-v1.md`
  - `docs/public/EXTENSION-INTERFACE.md` (new)
  - `extensions/__init__.py` (new)
  - `extensions/contracts.py` (new)
  - `extensions/loader.py` (new)
  - `scripts/extension_contract_check.py` (new)

## Overview
Define a stable extension interface so private/custom workflow and content packs can plug into the public core without modifying core foundation code.

## Goals
- Keep public core generic and clean.
- Allow private/persona overlays to attach through explicit contracts.
- Prevent custom packs from creating hidden coupling to internal core modules.

## Definition of Done (DoD)
**DoD checklist:**
- [ ] Extension contract types/interfaces are defined and versioned.
- [ ] Loader path supports registering external packs without editing core logic.
- [ ] Contract checker validates extension compatibility and fails with clear diagnostics.
- [ ] Public docs explain adapter model with examples and anti-patterns.
- [ ] Refactor-pass simplification loop is applied to all touched code.

**Validation commands (run from repo root):**
```bash
set -euo pipefail
cd .

python3 scripts/extension_contract_check.py --strict
python3 scripts/run_tests.py --target extensions
python3 -m compileall extensions scripts
```
**Pass criteria:** all commands exit 0; contract checker reports no compatibility violations.

## User Stories

### US-001: Core stays generic
**Description:** As maintainer, I want custom packs to plug in externally so the core repo remains broadly useful.

**Acceptance Criteria:**
- [ ] Core can run with zero custom packs installed.
- [ ] Installing/removing a pack does not require editing core files.

### US-002: Overlay safety
**Description:** As operator, I want clear compatibility checks before enabling a pack.

**Acceptance Criteria:**
- [ ] Incompatible pack versions fail fast with actionable error messages.
- [ ] Contract checker output includes fix guidance.

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
