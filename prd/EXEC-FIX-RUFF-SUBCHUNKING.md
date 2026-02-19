# PRD: Fix Ruff lint in mcp_ingest_sessions sub-chunking

## Goal
Pass Ruff lint for PR #54 by fixing type annotations, import order, and variable naming issues in `scripts/mcp_ingest_sessions.py`.

## Owned Paths:
- prd/EXEC-FIX-RUFF-SUBCHUNKING.md
- scripts/mcp_ingest_sessions.py

## DoD
- Ruff lint passes for `scripts/mcp_ingest_sessions.py`.
- No behavior changes beyond lint fixes (types/imports/variable names).

## Validation
- `python3 -m ruff check scripts/mcp_ingest_sessions.py`
