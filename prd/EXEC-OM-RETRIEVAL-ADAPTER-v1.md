# PRD: OM Retrieval Adapter v1

## PRD Metadata
- Type: Execution
- Kanban Task: task-unified-ingestion-smart-cutter-v1
- Parent Epic: N/A
- Depends On: merged OM lane-tagging fix (#133)
- Preferred Engine: Either
- Owned Paths:
  - `prd/EXEC-OM-RETRIEVAL-ADAPTER-v1.md`
  - `mcp_server/src/graphiti_mcp_server.py`
  - `mcp_server/src/services/search_service.py`
  - `mcp_server/src/services/neo4j_service.py`
  - `mcp_server/tests/test_search_om_lane.py`
  - `mcp_server/tests/test_search_lane_isolation.py`
  - `docs/runbooks/om-operations.md`

## Overview
Enable practical retrieval for `s1_observational_memory` without converting OM nodes into Graphiti `Entity` objects.

## Constraints
- No OMNode→Entity conversion.
- OM results only when lane includes `s1_observational_memory`.
- Preserve existing behavior for non-OM lanes.
- Fail closed on OM path errors.

## Definition of Done (DoD)
**DoD checklist:**
- [x] OM-lane queries return OM-backed results from OM primitives.
- [x] Non-OM lane queries remain unchanged.
- [x] OM retrieval errors fail closed without leaking cross-lane data.
- [x] Added tests pass.

**Validation commands (run from repo root):**
```bash
cd mcp_server
uv sync --group dev
uv run pytest tests/test_search_om_lane.py tests/test_search_lane_isolation.py -q
```
