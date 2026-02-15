# PRD: Boundary Audit Strict-Mode Regression Coverage v1

## PRD Metadata
- Type: Execution
- Kanban Task: issue-5
- Parent Epic: task-graphiti-publicization-upstream-sync
- Depends On: N/A
- Preferred Engine: Either
- Owned Paths:
  - `prd/EXEC-BOUNDARY-AUDIT-STRICT-TESTS-v1.md`
  - `tests/test_public_repo_boundary_audit.py`

## Overview
Add regression tests for `scripts/public_repo_boundary_audit.py` strict-mode behavior.

## Goals
- Prove `--strict` exits non-zero when BLOCK/AMBIGUOUS findings exist.
- Prove `--strict` exits zero when all files are ALLOW.
- Prove non-strict mode does not fail on ambiguous findings.

## Definition of Done (DoD)
**DoD checklist:**
- [ ] Test covers strict success path (all ALLOW -> exit 0).
- [ ] Test covers strict failure path (AMBIGUOUS -> exit 1).
- [ ] Test covers non-strict ambiguous path (exit 0).
- [ ] Report output file is asserted in tests.

**Validation commands (run from repo root):**
```bash
set -euo pipefail
python3 tests/test_public_repo_boundary_audit.py
```
**Pass criteria:** test run exits 0 and all strict/non-strict assertions pass.

## Functional Requirements
- FR-1: Tests execute script as subprocess to validate real CLI behavior.
- FR-2: Tests use temporary git repos to control file sets deterministically.
- FR-3: Tests must not require network or external services.

## Non-Goals (Out of Scope)
- Refactoring boundary script implementation.
- Expanding policy semantics.
