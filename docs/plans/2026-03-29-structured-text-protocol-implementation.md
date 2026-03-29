# Graphiti Structured Text Protocol Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace schema-first extraction for the Graphiti extraction and dedupe flows with a shared structured-text protocol plus tolerant local parsing.

**Architecture:** Introduce `graphiti_core.llm_compat` for tagged-text parsing, add explicit LLM response modes so extraction paths can request raw text instead of schema-constrained JSON, and migrate node/edge extraction, summaries, and dedupe call sites to build Graphiti objects locally from parsed records.

**Tech Stack:** Python 3.10+, Pydantic v2, pytest, `uv`, existing Graphiti prompts and maintenance modules.

---

### Task 1: Response modes
- Add `structured_text`, `structured_json`, and `plain_text` support to the LLM client contract.
- Verify schema injection happens only for `structured_json`.
- Verify OpenAI-compatible clients can return raw text without forcing JSON parsing.

### Task 2: Parser layer
- Add `graphiti_core.llm_compat` with block extraction, field parsing, task-specific parsed records, and tolerant builders.
- Cover wrapper noise, unknown fields, missing trailing `END ITEM`, duplicate keys, and defensive integer-list parsing.

### Task 3: Prompt migration
- Update extraction and dedupe prompts to instruct models to return tagged plain text instead of JSON objects.
- Preserve current semantic guidance while swapping the wire format contract.

### Task 4: Node flows
- Request `structured_text` in node extraction, summary, and dedupe paths.
- Parse tagged text locally and keep downstream node creation/dedup semantics intact.

### Task 5: Edge flows
- Request `structured_text` in edge extraction and edge dedupe paths.
- Parse tagged text locally, keep current edge creation behavior, and preserve tolerant datetime handling.

### Task 6: Verification and branch completion
- Run focused unit tests, broader maintenance/LLM-client regression tests, ruff, and pyright.
- Commit on the isolated branch, push to origin, and verify CI runs on GitHub.

## Completion Notes

### Final implementation scope

- `graphiti_core.llm_client` now supports `structured_text` end-to-end for both the generic OpenAI-compatible client and the default `OpenAIClient` path.
- The graph service bootstrap now wires `model_name`, `small_model_name`, and `embedding_model_name` into the instantiated Graphiti clients instead of mutating only the primary LLM client after construction.
- Added a regression test for the OpenAI structured-text path and a service bootstrap regression test to prevent future MiniMax/OpenAI-compatible configuration drift.

### Local Docker deployment findings

The local `zepai/graphiti:latest` image used in Docker does not automatically exercise this branch's code unless both of the following are replaced:

- `graphiti_core` in the runtime site-packages directory
- `server/graph_service` in `/app/graph_service`

For the local MiniMax-backed deployment used during verification, the effective bind mounts were:

- `graphiti_core -> /app/.venv/lib/python3.12/site-packages/graphiti_core`
- `server/graph_service -> /app/graph_service`

Mounting only the system Python site-packages path was insufficient because the service process runs under `/app/.venv/bin/python`.

### Provider compatibility notes

Two runtime issues surfaced while validating against the local MiniMax proxy:

1. Structured-text extraction call sites passed `response_mode='structured_text'`, but the default `OpenAIClient` path initially did not accept or honor that mode.
2. The graph service did not initially propagate `SMALL_MODEL_NAME`, causing `ModelSize.small` requests to fall back to `gpt-4.1-nano`, which MiniMax rejected with `400 unknown model`.

These are now covered by code and regression tests.

### End-to-end verification notes

- `/messages` ingestion, `/episodes/{group_id}`, `/search`, and `/get-memory` were verified locally against Neo4j plus the MiniMax/Ollama proxy stack.
- The local MiniMax-backed extraction flow is relatively slow. In practice, one episode took roughly a minute to finish processing, and a two-message queue completed after roughly 70 to 140 seconds depending on the extraction stage.
- `GET /get-memory` requires `center_node_uuid` to be present in the request body; use `null` when there is no center node.
