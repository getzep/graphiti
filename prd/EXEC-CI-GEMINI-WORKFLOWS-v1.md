# PRD: Add Baseline CI + Gemini Security Workflows v1

## PRD Metadata
- Type: Execution
- Kanban Task: task-graphiti-publicization-upstream-sync
- Parent Epic: task-graphiti-publicization-upstream-sync
- Depends On: PR #1 branch state
- Preferred Engine: Either
- Owned Paths:
  - `prd/EXEC-CI-GEMINI-WORKFLOWS-v1.md`
  - `.github/workflows/ci.yml`
  - `.github/workflows/gemini-security.yml`

## Overview
Add explicit, repo-local GitHub Actions for baseline CI and Gemini Security analysis so the repository has straightforward, visible automation on PRs.

## Goals
- Add a single baseline CI workflow (`ci.yml`) that runs on PRs/pushes.
- Add a Gemini security workflow (`gemini-security.yml`) that runs on PRs and manual dispatch.
- Keep workflow setup simple and maintainable.

## Definition of Done (DoD)
**DoD checklist:**
- [ ] `.github/workflows/ci.yml` exists and has lint, typecheck, and unit test jobs.
- [ ] `.github/workflows/gemini-security.yml` exists and runs Gemini security analysis on PRs.
- [ ] Gemini job safely skips when `GEMINI_API_KEY` is not configured.
- [ ] Workflow files are syntactically valid YAML and committed.

**Validation commands (run from repo root):**
```bash
set -euo pipefail
test -f .github/workflows/ci.yml
test -f .github/workflows/gemini-security.yml
python3 tests/test_public_repo_boundary_audit.py
```

**Pass criteria:** files exist and local validation exits 0.

## Functional Requirements
- FR-1: CI workflow triggers on `push` to `main` and `pull_request` to `main`.
- FR-2: CI workflow includes at least lint/typecheck/tests jobs.
- FR-3: Gemini security workflow triggers on `pull_request` and `workflow_dispatch`.
- FR-4: Gemini workflow uses `google-github-actions/run-gemini-cli` and security extension prompt.
- FR-5: Missing `GEMINI_API_KEY` must not hard-fail the workflow.

## Non-Goals (Out of Scope)
- Replacing existing advanced workflows.
- Enforcing required-check branch protections.
- Secret provisioning automation.
