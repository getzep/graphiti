# PRD: Public Docs + Release Rails (Foundation-Only) v1

## PRD Metadata
- Type: Execution
- Kanban Task: task-graphiti-public-docs-release-rails
- Parent Epic: task-graphiti-publicization-upstream-sync
- Depends On: task-graphiti-public-boundary-contract, task-graphiti-public-refactor-pass
- Preferred Engine: Either
- Owned Paths:
  - `prd/EXEC-PUBLIC-DOCS-RELEASE-RAILS-v1.md`
  - `docs/public/README.md` (new)
  - `docs/public/SECURITY-BOUNDARIES.md` (new)
  - `docs/public/RELEASE-CHECKLIST.md` (new)
  - `docs/public/WHAT-NOT-TO-MIGRATE.md` (new)

## Overview
Create public-facing docs and release guardrails so external users get a clean, generalizable foundation while private/operator-specific assets remain excluded.

## Goals
- Make repo purpose and scope immediately clear.
- Document security/privacy boundaries and excluded assets.
- Add release checklist gates for repeatable public updates.
- Explicitly defer content marketing draft until readiness conditions are met.

## Definition of Done (DoD)
**DoD checklist:**
- [ ] Public README explains architecture, scope, and setup for generalized use.
- [ ] Security boundaries doc lists excluded classes and rationale.
- [ ] "What not to migrate" doc explicitly blocks private workflow/content packs.
- [ ] Release checklist includes mandatory pre-publish checks.
- [ ] Docs include content-marketing gate: drafting stays deferred until ingest backlog/workflow readiness criteria are true.

**Validation commands (run from repo root):**
```bash
set -euo pipefail
cd .

test -s docs/public/README.md
test -s docs/public/SECURITY-BOUNDARIES.md
test -s docs/public/RELEASE-CHECKLIST.md
test -s docs/public/WHAT-NOT-TO-MIGRATE.md
rg -n "content marketing|public write-up|deferred|gate" docs/public/RELEASE-CHECKLIST.md
```
**Pass criteria:** all docs exist and include explicit deferral gate language.

## User Stories

### US-001: New maintainer clarity
**Description:** As an external maintainer, I want to understand what this repo is and is not.

**Acceptance Criteria:**
- [ ] README contains a clear scope statement and non-goals.
- [ ] README links to security boundaries and release checklist.

### US-002: Leak prevention through docs
**Description:** As owner, I want docs to enforce boundaries so accidental migration of private assets is less likely.

**Acceptance Criteria:**
- [ ] Excluded asset classes are explicit and concrete.
- [ ] Release checklist has a mandatory boundary-audit step.

## Functional Requirements
- FR-1: README must separate "foundation substrate" from "private packs/examples".
- FR-2: Security boundaries doc must include threat-oriented rationale.
- FR-3: Release checklist must be executable and concise.
- FR-4: Docs must remain generic (no personal/company private references).
- FR-5: If any helper code/scripts are introduced in this PRD, they must follow the mandatory simplification loop from the epic.
- FR-6: Cross-repo symlinks for PRDs/config/docs are prohibited; mirrored artifacts must be produced by scripted copy/sync flow.

## Non-Goals (Out of Scope)
- Drafting/publishing marketing thread/article.
- Implementing new product features.

## Technical Considerations
- Keep docs concise and operational.
- Avoid duplicating policy details across files; link where possible.

## Execution Plan (Serial vs Parallel)
### Critical path (serial)
1. Draft scope + exclusions docs.
2. Draft release checklist.
3. Add readiness gate language for deferred public write-up.
4. Validate links and consistency.

### Parallel workstreams (if any)
- README and security boundaries can be drafted in parallel.

### Dependency map
- Depends on boundary contract for accurate exclusions.
- Should reflect simplified structure from refactor pass.

## Open Questions
- Whether to include a minimal public roadmap section now or leave as issue-based planning only.
