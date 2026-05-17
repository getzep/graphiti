# Plan — late-interaction reranker

## Approach

Add a new subclass of `CrossEncoderClient` that wraps a late-interaction model and exposes the same `rank(query, passages)` interface as existing rerankers. No changes to the storage layer or to `SearchConfig` semantics, only a new reranker enum value and recipe. The local implementation uses `sentence-transformers` and downloads weights on first run. The hosted variant ships as a thin SDK wrapper if/when a vendor API is GA.

## Steps

1. Add `graphiti_core/cross_encoder/late_interaction_reranker.py` with:
   - `LateInteractionRerankerClient(CrossEncoderClient)` abstract base — defines a `_max_sim(query_token_embeddings, passage_token_embeddings)` helper and the `rank` contract.
   - `ColBERTLocalRerankerClient` concrete — sentence-transformers backend, checkpoint configurable via `model_name` (default `'colbert-ir/colbertv2.0'`).
2. Reuse existing `[project.optional-dependencies].sentence-transformers` extra; document the requirement in `README.md` and in the new module's docstring.
3. Add `EdgeReranker.LATE_INTERACTION` and `NodeReranker.LATE_INTERACTION` in `graphiti_core/search/search_config.py`. Wire dispatch in `graphiti_core/search/search.py` — the dispatch site calls the late-interaction client identically to other cross-encoders.
4. Add the recipe `LATE_INTERACTION_HYBRID_SEARCH` in `graphiti_core/search/search_config_recipes.py` with sane defaults (`limit=20`, `pre_k=100`, edge + node late-interaction, episode + community remain RRF).
5. (Optional, this PR or follow-up) hosted variant: `graphiti_core/cross_encoder/voyage_rerank_client.py` or `cohere_rerank_client.py`, depending on which vendor's late-interaction endpoint is GA at PR time. Gate behind a new optional extra.
6. Add `tests/cross_encoder/test_late_interaction_reranker.py` (mock + integration using a tiny CPU-friendly checkpoint).
7. Add `tests/search/test_late_interaction_recipe.py` — end-to-end with mocked LLM and real reranker on a small synthetic graph.
8. Add `examples/quickstart/quickstart_late_interaction.py` — same shape as the existing quickstart, with the new recipe.
9. Add a new mini-benchmark `tests/evals/multilingual_retrieval/` (50 hand-labeled questions, 3 languages) for the AI eval plan.
10. Update `docs/specs/ai-quality.md` "AI surface area" and "Cross encoders" sections.

## Dependencies / order

Step 1 before 3. Step 3 before 4. Steps 6 and 7 after 4. Step 5 can be a separate PR if no vendor API is GA in time.
