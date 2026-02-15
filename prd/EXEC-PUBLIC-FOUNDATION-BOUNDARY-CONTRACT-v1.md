# PRD: Public Foundation Boundary Contract + Allowlist Audit v1

## PRD Metadata
- Type: Execution
- Kanban Task: task-graphiti-public-boundary-contract
- Parent Epic: task-graphiti-publicization-upstream-sync
- Depends On: N/A
- Preferred Engine: Either
- Owned Paths:
  - `prd/EXEC-PUBLIC-FOUNDATION-BOUNDARY-CONTRACT-v1.md`
  - `config/public_export_allowlist.yaml`
  - `config/public_export_denylist.yaml`
  - `scripts/public_repo_boundary_audit.py`
  - `docs/public/BOUNDARY-CONTRACT.md`

## Overview
Create a default-closed public-export boundary with deterministic allow/block classification and a reproducible markdown audit report.

## Mandatory Cross-Repo Baseline Review (to prevent narrow-pass regressions)
Before implementation, the agent must:
1. Review corresponding paths in `projects/graphiti` (private/source baseline) and `projects/graphiti-openclaw` (public target).
2. Produce a short cross-repo inventory in PR notes listing concrete files/directories reviewed in both repos.
3. Identify at least 3 candidate simplifications across the owned-path surface; implement selected items or explicitly defer each candidate with rationale.
4. If the PR touches only one file or one narrow function, include explicit justification for why broader owned-path opportunities were not applicable.

## Goals
- Enforce allowlist-first public scope with denylist overrides.
- Surface ambiguous files explicitly.
- Provide strict mode for CI/release gates.

## Definition of Done (DoD)
**DoD checklist:**
- [ ] Allowlist and denylist manifests exist and are documented.
- [ ] Audit script classifies ALLOW/BLOCK/AMBIGUOUS deterministically.
- [ ] Markdown report includes summary counts and offending paths.
- [ ] Strict mode exits non-zero when non-ALLOW findings exist.

**Validation commands (run from repo root):**
```bash
set -euo pipefail
python3 scripts/public_repo_boundary_audit.py \
  --manifest config/public_export_allowlist.yaml \
  --denylist config/public_export_denylist.yaml \
  --report /tmp/boundary-audit.md

test -s /tmp/boundary-audit.md
python3 -m py_compile scripts/public_repo_boundary_audit.py
```
**Pass criteria:** commands exit 0, report file exists with summary table.

## Functional Requirements
- FR-1: Denylist rules are evaluated before allowlist.
- FR-2: Non-matching files are AMBIGUOUS.
- FR-3: Script reads git-tracked files by default.
- FR-4: Script supports optional untracked-file inclusion.
- FR-5: Report includes remediation hints.

## Non-Goals (Out of Scope)
- History rewrite.
- Upstream sync automation.
- Pack migration.
