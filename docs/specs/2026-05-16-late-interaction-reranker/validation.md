# Validation — late-interaction reranker

## Automated tests

- `tests/cross_encoder/test_late_interaction_reranker.py` — mock-mode tests cover the `rank` contract (order preserved when scores tie, returns the same number of passages as input). Integration test uses a tiny CPU-friendly checkpoint to keep CI under 60s.
- `tests/search/test_late_interaction_recipe.py` — end-to-end with a mocked LLM and a real (small-checkpoint) reranker on a synthetic graph; asserts the recipe produces top-K results in MaxSim order.
- `tests/cross_encoder/test_late_interaction_model_download.py` — skipped by default; runs in nightly with `RUN_NETWORK_TESTS=1`. Verifies the default checkpoint downloads and loads.

## Smoke checks

```bash
uv run python examples/quickstart/quickstart_late_interaction.py
# Runs against local Neo4j; downloads the ColBERT v2 checkpoint on first run.
# Top edges should look meaningfully different from the RRF baseline.
```

## Manual criteria

- On a multilingual sample (en / es / ja queries against an en-dominant graph), the top-5 results look qualitatively better than the RRF baseline.
- p50 latency under 500 ms for k=20 on CPU on a modern laptop. Acceptable for interactive use; if higher, document GPU recommendation.

## AI eval plan

- **Success criteria**:
  - LongMemEval-S overall accuracy with `LATE_INTERACTION_HYBRID_SEARCH` is **at least 3 points above** `COMBINED_HYBRID_SEARCH_CROSS_ENCODER` on the same `(llm, embedder)` pair.
  - Multilingual mini-bench precision@5 ≥ 0.85 on a hand-labeled 50-question set spanning English, Spanish, Japanese.
  - No more than 2-point regression on the LongMemEval "abstention" subscore (over-confident reranking is a known failure mode).
- **Eval dataset**: LongMemEval-S regression subset (from the LongMemEval harness spec) + new `tests/evals/multilingual_retrieval/` (50 questions × 3 languages).
- **Regression set**: the new recipe under default config on the 50-question LongMemEval subset.
- **Cadence**: per-PR for any change in late-interaction code; full LongMemEval-S on release.

## Risks & rollback

- **Failure modes**: model download timeouts in air-gapped CI; ColBERT v2 license / redistribution surprises; per-query latency too high for interactive use; quality drop on monolingual English vs cross-encoder; sentence-transformers version conflicts with existing extras.
- **Rollback**: feature is opt-in via recipe and never the default. Revert PR cleanly removes the new files, the enum values, and the recipe; no callers in the default path depend on it.

## Open questions

- Default local checkpoint: ColBERT v2 vs ColBERTv2-multilingual vs Wholembed v3 weights when public. Decide at implementation time based on availability and license.
- Hosted-vendor preference: Voyage rerank-2 MaxSim vs Cohere Rerank v4 vs both. Depend on which is GA + has a Python SDK at PR time.
- Cache token-level embeddings between queries? Memory vs latency tradeoff; lean "no caching in v1, revisit after profile."
