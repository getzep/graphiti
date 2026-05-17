# Plan — LongMemEval bench harness

## Approach

Add a new `longmemeval/` package under `tests/evals/`. Three async phases — ingest (one `group_id` per user, sequential `add_episode`), query (per question: `search_` → format → reader prompt → answer), score (wrap upstream LongMemEval scoring). The runner parallelizes across users via `semaphore_gather` (bounded by `SEMAPHORE_LIMIT` and LLM rate limits). Output is a JSON results artifact plus `BASELINE.md` that becomes the published snapshot; both are committed to git so changes are reviewable in PRs.

## Steps

1. Vendor LongMemEval-S under `tests/evals/data/longmemeval/`. Add a `README.md` documenting source URL, upstream SHA, and license.
2. Add `tests/evals/longmemeval/__init__.py`, `dataset.py` (load + parse upstream JSONL into typed dataclasses), `ingest.py` (sessions → `add_episode`), `query.py` (questions → `search_` → reader prompt → answer), `score.py` (wrap upstream scoring scripts), `run.py` (CLI entry).
3. Add `tests/evals/longmemeval/reader_prompt.py` — single-shot prompt template that takes `(question, retrieved_facts)` and returns a short answer. Versioned (`v1`) like the rest of `graphiti_core/prompts/`.
4. Add `make eval-longmemeval` target gated behind `RUN_EVALS=1`. Target invokes the CLI with smoke defaults (`--users 5 --subset s`).
5. Add `tests/evals/longmemeval/regression_subset.json` — 50 stratified questions (10 per ability). Smoke target uses this set by default; `--full` flag runs the complete subset.
6. After first end-to-end run, commit `tests/evals/longmemeval/BASELINE.md` with the matrix and a one-paragraph commentary.
7. Cross-link from `docs/specs/ai-quality.md` "Regression checks" section.
8. (Stretch) parameterize the runner to emit a CSV results matrix across `(llm, embedder, reranker)` triples for easy comparison.

## Dependencies / order

Step 1 before 2 (imports need the data). Step 3 before 5 (regression subset is exercised end-to-end). Step 4 before 5. Steps 6 and 7 last.
