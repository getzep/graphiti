# PRD: Publicization Adapter Wiring for Existing Workflow/Content Packs v1

## PRD Metadata
- Type: Execution
- Kanban Task: task-graphiti-publicization-adapter-wiring
- Parent Epic: task-graphiti-publicization-upstream-sync
- Depends On: task-graphiti-pack-adapter-interface, task-vc-policy-binding-integration-e2e
- Preferred Engine: Codex
- Owned Paths:
  - `prd/EXEC-PUBLICIZATION-ADAPTER-WIRING-v1.md`
  - `config/runtime_pack_registry.yaml`
  - `config/runtime_consumer_profiles.yaml`
  - `workflows/vc_memo_drafting.pack.yaml`
  - `workflows/vc_deal_brief.pack.yaml`
  - `workflows/vc_diligence_questions.pack.yaml`
  - `workflows/vc_ic_prep.pack.yaml`
  - `scripts/runtime_pack_router.py`
  - `tests/test_runtime_pack_router.py`
  - `tests/test_vc_policy_profile_routing.py`

## Overview
Wire existing workflow/content packs into the adapter interfaces so runtime selection is explicit, deterministic, and test-covered.

## Goals
- Eliminate ad-hoc pack wiring paths for existing VC workflows.
- Keep wiring declarative via registry/profile files.
- Ensure private/group-safe scope behavior is deterministic across consumers.
- Migrate existing workflow/content packs and private overlay pack wiring from legacy `clawd-graphiti` into this repo (`projects/graphiti-openclaw`) as the single source of truth.

## Definition of Done (DoD)
**DoD checklist:**
- [x] Existing VC workflows are adapter-routed through `runtime_pack_router` with no hidden hardcoded pack selection.
- [x] `runtime_pack_registry.yaml` and `runtime_consumer_profiles.yaml` are aligned (no dangling pack keys).
- [x] Router output validates and is reproducible for memo/deal-brief/diligence/ic-prep consumers.
- [x] Routing tests cover positive path + misconfiguration path.
- [x] Existing workflow/content packs needed for current operations are migrated and wired in `projects/graphiti-openclaw` (not legacy `clawd-graphiti`).
- [x] No runtime adapter/profile references to legacy `clawd-graphiti` repo paths remain.

**Validation commands (run from repo root):**
```bash
set -euo pipefail
python3 scripts/runtime_pack_router.py \
  --consumer main_session_vc_memo \
  --workflow-id vc_memo_drafting \
  --step-id draft \
  --repo . \
  --task "Draft memo" \
  --validate \
  --out /tmp/router-vc-memo.json

python3 scripts/runtime_pack_router.py \
  --consumer main_session_vc_deal_brief \
  --workflow-id vc_deal_brief \
  --step-id compose \
  --repo . \
  --task "Prepare deal brief" \
  --validate \
  --out /tmp/router-vc-deal-brief.json

python3 scripts/runtime_pack_router.py \
  --consumer main_session_vc_ic_prep \
  --workflow-id vc_ic_prep \
  --step-id synthesize \
  --repo . \
  --task "Prepare IC brief" \
  --validate \
  --out /tmp/router-vc-ic-prep.json

python3 -m unittest \
  tests/test_runtime_pack_router.py \
  tests/test_vc_policy_profile_routing.py -v

test -s /tmp/router-vc-memo.json
test -s /tmp/router-vc-deal-brief.json
test -s /tmp/router-vc-ic-prep.json
test -s workflows/vc_memo_drafting.pack.yaml
test -s workflows/vc_deal_brief.pack.yaml
test -s workflows/vc_diligence_questions.pack.yaml
test -s workflows/vc_ic_prep.pack.yaml
! rg -n "clawd-graphiti|projects/clawd-graphiti" config/runtime_pack_registry.yaml config/runtime_consumer_profiles.yaml workflows scripts/runtime_pack_router.py
```
**Pass criteria:** all commands exit 0; router outputs validate; tests pass with no consumer/profile drift; no legacy `clawd-graphiti` path references remain in adapter wiring surfaces.

## User Stories

### US-001: Deterministic adapter wiring
**Description:** As operator, I want workflow packs to route through one adapter contract so behavior is explainable and maintainable.

**Acceptance Criteria:**
- [ ] Pack selection is profile-driven, not implicit in workflow code.
- [ ] Scope restrictions are explicit and test-covered.

### US-002: Low-friction future extension
**Description:** As maintainer, I want adding a new pack or workflow to require config + tests, not structural rewrites.

**Acceptance Criteria:**
- [ ] Adding one new pack requires only registry/profile updates plus tests.
- [ ] Existing workflows continue to route unchanged.

## Functional Requirements
- FR-1: Consumer profile → pack plan mapping must be deterministic.
- FR-2: Required packs must fail closed when missing.
- FR-3: Group-safe and private scopes must be validated per profile entry.
- FR-4: Router output format must stay schema-valid.
- FR-5: Existing private workflow/content pack wiring required for current operations must be represented in this repo's registry/profile model.
- FR-6: Adapter/router/config surfaces must not rely on legacy `clawd-graphiti` repo paths.

## Non-Goals (Out of Scope)
- New pack types beyond currently shipped pack set.
- Public article/content publishing logic.

## Technical Considerations
- Prefer declarative config over imperative glue code.
- Keep router behavior model-agnostic.
- Avoid introducing a second routing abstraction.

## Execution Plan (Serial vs Parallel)
### Critical path (serial)
1. Normalize registry/profile entries.
2. Wire workflow IDs to stable consumer profiles.
3. Add/adjust router tests.
4. Run validation commands.

### Parallel workstreams (if any)
- Workflow metadata cleanup can run in parallel with test fixture updates.

### Dependency map
- Depends on interface-contract work.
- Feeds cron cutover and final GO gate confidence.

## Open Questions
- Should diligence and IC-prep share one profile by default, or remain separate for future stricter policy divergence?
