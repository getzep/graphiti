# Verify PRD: Sessions Sub-Chunking (Public Repo)

## PRD Metadata
- Type: Verify-only
- Scope: Deterministic sub-chunking for large session evidence in mcp_ingest_sessions.py

Owned Paths:
- prd/VERIFY-SESSIONS-SUBCHUNKING-v1.md
- scripts/mcp_ingest_sessions.py
- docs/runbooks/sessions-subchunking.md

## Goal
Verify correctness of deterministic sub-chunking feature:
1. `mcp_ingest_sessions.py` — split oversized evidence into sub-chunks at enqueue time with stable `:pN` key suffixes.
2. `docs/runbooks/sessions-subchunking.md` — new runbook documenting the sub-chunking design.

## Key Invariants
- Sub-chunking is deterministic: same input → same sub-chunks.
- Sub-chunk keys use `:p0`, `:p1`, ... suffixes for stable dedup.
- Registry dedup is preserved (content hash per sub-chunk).
- Only group IDs in `_SUBCHUNK_GROUP_IDS` are sub-chunked.
- No prompt injection vectors via evidence content.
- Script remains idempotent on re-run.
- `subchunk_evidence()` raises ValueError if max_chars <= 0 (prevents infinite loop).
- `--subchunk-size` argparse validation rejects non-positive values.

## Definition of Done
- [ ] Sub-chunking is deterministic (same input → same output)
- [ ] Sub-chunk keys use stable `:pN` suffixes
- [ ] Registry dedup preserved per sub-chunk
- [ ] Only _SUBCHUNK_GROUP_IDS groups are sub-chunked
- [ ] No prompt injection vectors via evidence content
- [ ] Runbook docs/runbooks/sessions-subchunking.md exists and is accurate
- [ ] subchunk_evidence() raises ValueError if max_chars <= 0
- [ ] --subchunk-size argparse validation rejects <= 0

## Validation Commands
```bash
# Script is valid Python
python3 -m py_compile scripts/mcp_ingest_sessions.py
# Runbook exists
test -f docs/runbooks/sessions-subchunking.md
# Guard: subchunk_evidence raises on max_chars <= 0
python3 -c "
from scripts.mcp_ingest_sessions import subchunk_evidence
try:
    subchunk_evidence('test', 'k', 0)
    raise SystemExit('FAIL: no ValueError for max_chars=0')
except ValueError:
    pass
try:
    subchunk_evidence('test', 'k', -1)
    raise SystemExit('FAIL: no ValueError for max_chars=-1')
except ValueError:
    pass
print('PASS: subchunk_evidence guard works')
"
```
