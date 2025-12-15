# Implementation Plan

- [x] 1. Add ontology specs _Requirements: 1, 2, 3, 4_
- [x] 2. Stabilize async ingest lifecycle _Requirements: 3.1, 3.2, 3.3_
  - [x] 2.1 Keep a single Graphiti instance in `app.state` _Requirements: 3.2_
  - [x] 2.2 Start/stop worker in app lifespan _Requirements: 3.1_
  - [x] 2.3 Ensure worker continues after failures _Requirements: 3.3_
- [x] 3. Implement `agent_memory_v1` ontology registry _Requirements: 2.1, 2.2, 2.3_
  - [x] 3.1 Add `graph_service/ontologies/agent_memory_v1.py` _Requirements: 2.1_
  - [x] 3.2 Add `graph_service/ontologies/registry.py` resolver _Requirements: 1.2, 1.3, 1.4_
- [x] 4. Extend ingest API for schema selection _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - [x] 4.1 Add `schema_id` to `AddMessagesRequest` _Requirements: 1.1_
  - [x] 4.2 Apply schema per message in `POST /messages` _Requirements: 1.2, 1.3, 1.4_
  - [x] 4.3 Validate unknown schema ids _Requirements: 1.5_
- [x] 5. Add server tests and docs/demo _Requirements: 3.3, 4.1, 4.2_
  - [x] 5.1 Add/adjust pytest discovery so root tests stay isolated _Requirements: 3.3_
  - [x] 5.2 Add demo under `examples/` or `server/README.md` _Requirements: 4.1, 4.2_
- [ ] 6. Verify, commit, push, PR _Requirements: 1, 2, 3, 4_

## Current Status Summary

- Phase: implementation (in progress)
- Next: implement tasks 2–5, then run root + server `make check` and open a PR.
