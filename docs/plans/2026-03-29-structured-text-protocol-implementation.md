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
