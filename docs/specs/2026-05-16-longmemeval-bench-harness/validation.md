# Validation — LongMemEval bench harness

## Automated tests

- `tests/evals/longmemeval/test_dataset_loader.py` — unit test asserting the upstream JSONL schema parses into the typed dataclasses; fails loudly on schema drift.
- `tests/evals/longmemeval/test_ingest_smoke.py` — runs one synthetic user end-to-end with a mocked LLM/embedder; asserts the right number of episodes land and that `group_id` partitioning is honored.
- `tests/evals/longmemeval/test_reader_prompt.py` — mock LLM, asserts the prompt produces a one-line answer string and handles empty retrieval.
- The full harness is **not** part of `make test`. It runs under `RUN_EVALS=1 make eval-longmemeval` (smoke), `RUN_EVALS=1 make eval-longmemeval-full` (full set), or the CLI directly.

## Smoke checks

```bash
RUN_EVALS=1 uv run python -m tests.evals.longmemeval.run \
  --subset s --users 5 \
  --model gpt-5.5 --embedder text-embedding-3-large \
  --reranker openai
```

Should finish in under 15 minutes with no exceptions, produce `tests/evals/longmemeval/results/<timestamp>.json`, and print a per-ability accuracy summary.

## Manual criteria

- Read 5 random `(question, retrieved_facts, model_answer, ground_truth)` rows from the output JSON. Reader answers shouldn't be obviously truncated or schema-violating.
- Scoring numbers move in the expected direction when a known-good prompt is intentionally swapped for a known-bad one (sanity check that scoring isn't returning a constant).

## AI eval plan

- **Success criteria**: LongMemEval-S overall accuracy ≥ 70% on the baseline stack (GPT-5.5 + Voyage 4 Large + OpenAI reranker, `COMBINED_HYBRID_SEARCH_CROSS_ENCODER` recipe). Per-ability accuracies reported separately. A failing run does not block merge of the harness itself — the harness only needs to *measure*; raising the number is a separate effort.
- **Eval dataset**: LongMemEval-S vendored under `tests/evals/data/longmemeval/`, ~500 questions, ~115k-token chat histories per user (~40 sessions each).
- **Regression set**: `tests/evals/longmemeval/regression_subset.json` — 50 questions, 10 per ability, runs in <10 min on the baseline stack.
- **Cadence**: regression subset gated on every release PR; full LongMemEval-S nightly + before any 0.x release tag; LongMemEval-M ad-hoc when targeting long-context regressions.

## Risks & rollback

- **Failure modes**: upstream LongMemEval dataset format drifts and our vendored snapshot mismatches their scoring scripts; LLM provider rate-limits make runs flaky; results regress and we can't tell whether a prompt change, a model change, or a search-recipe change caused it.
- **Rollback**: keep `BASELINE.md` snapshotted per-commit so regressions are diff-detectable. Revert PR cleanly removes the entire package; nothing in `graphiti_core/` depends on it.

## Open questions

- Vendoring vs git submodule: snapshot at an upstream commit SHA, or submodule with a pinned commit? Lean toward vendor-with-SHA for simplicity; revisit if upstream license forbids redistribution.
- Do we publish the numbers externally (Zep blog, README badge) or keep them internal first? Marketing decision, not technical.
- Reader model identity vs ingestion model: same model for both, or separate? Lean toward same — eliminates a confound — but stretch goal is to parameterize.
