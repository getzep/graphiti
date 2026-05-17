# Requirements — LongMemEval bench harness

## Outcome

A reproducible LongMemEval-S / LongMemEval-M harness lives in `tests/evals/longmemeval/`, with a `BASELINE.md` that publishes Graphiti's score per `(LLM, embedder, reranker)` triple and is regenerated against new defaults and major prompt changes.

## Users affected

Maintainers (need a hard regression number when changing prompts, models, or search recipes), open-source users (need to trust the framework's published numbers), and competitive positioning — the last public Graphiti score is 63.8% on GPT-4o while Memanto (89.8%), ByteRover (>92%) and Supermemory (SOTA) have all published higher numbers in 2026.

## In scope

- Vendor LongMemEval-S (~500 questions, ~115k-token chat histories, ~40 sessions/user) under `tests/evals/data/longmemeval/` and pin to a specific upstream SHA.
- Ingest driver: LongMemEval sessions → ordered `Graphiti.add_episode` calls, one `group_id` per LongMemEval user, preserving session timestamps as `reference_time`.
- Query driver: each question runs through `Graphiti.search_(SearchConfig=COMBINED_HYBRID_SEARCH_CROSS_ENCODER)` followed by a thin single-shot reader prompt (`tests/evals/longmemeval/reader_prompt.py`).
- Scoring: wrap the upstream LongMemEval scoring scripts; output per-ability breakdown (info extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention) and overall accuracy.
- CLI entrypoint `tests/evals/longmemeval/run.py` exposing `--subset {s,m} --model … --embedder … --reranker … --users N --concurrency K`.
- `BASELINE.md` checked into the harness directory, regenerated after each baseline run, diff-reviewable in PRs.
- A `tests/evals/longmemeval/regression_subset.json` of 50 questions for sub-10-minute smoke runs.

## Out of scope

- Beating the SOTA in this PR. The purpose is to install the measurement.
- Running the full harness on every PR — too slow / too expensive. Regression subset only on PRs; full set nightly + before each release.
- Other benchmarks (LoCoMo, custom internal evals). Could come later.
- Building a reader-side agent loop — single-shot reader keeps retrieval quality isolated from agent design.

## Decisions

- Vendor the dataset (not git submodule). Reason: pin a snapshot for reproducibility; upstream availability has historically changed.
- One graph per LongMemEval user via `group_id = user_id`. Reason: matches Graphiti's partitioning model.
- Single-shot reader, no agent loop. Reason: isolates retrieval quality from agent design; competitive numbers (Memanto, ByteRover) are also reported this way.
- Baseline run: GPT-5.5 (LLM) + Voyage 4 Large (embedder) + OpenAI reranker (cross-encoder). Reason: matches the published "best available" stack as of May 2026; cheap-stack baseline (GPT-4.1-mini + text-embedding-3-large) reported as secondary row.
