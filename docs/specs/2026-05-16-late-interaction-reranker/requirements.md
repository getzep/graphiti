# Requirements — late-interaction reranker

## Outcome

Graphiti users can swap in a **late-interaction** cross-encoder (ColBERT-style / Wholembed-v3-style MaxSim over multi-vector token embeddings) for higher search precision, especially on multilingual and code-heavy corpora. Closes the gap with competitors (ByteRover, Memanto, Supermemory) that have published LongMemEval results using late-interaction retrieval in 2026.

## Users affected

Developers building retrieval-heavy agents (search-first agents, multilingual KBs, code search). Maintainers benchmarking against ColBERT-using competitors. Anyone whose `COMBINED_HYBRID_SEARCH_CROSS_ENCODER` results feel "almost right" but mis-rank near-duplicates.

## In scope

- New abstraction `LateInteractionRerankerClient(CrossEncoderClient)` in `graphiti_core/cross_encoder/late_interaction_reranker.py` that takes `(query, passages)` and returns `[(passage, score)]` using a two-stage approach: pre-filter by single-vector embedding (existing path) → score top-K with MaxSim over token-level embeddings.
- Reference local implementation `ColBERTLocalRerankerClient` backed by `sentence-transformers` (ColBERT v2 checkpoint by default; configurable).
- Optional hosted implementation if a vendor late-interaction API is GA at PR time (Voyage rerank-2 MaxSim or Cohere Rerank v4) — under `voyage_rerank_client.py` or equivalent.
- New enum values `EdgeReranker.LATE_INTERACTION` and `NodeReranker.LATE_INTERACTION` in `graphiti_core/search/search_config.py`.
- New recipe `LATE_INTERACTION_HYBRID_SEARCH` in `search/search_config_recipes.py`.
- Example `examples/quickstart/quickstart_late_interaction.py`.
- Update `docs/specs/ai-quality.md` AI surface area + cross_encoder section.

## Out of scope

- Storing per-token vectors at ingest time. Multi-vector indexing is a much larger project; this PR keeps the storage layer unchanged.
- GPU-specific optimization. CPU baseline first; revisit if latency demands it.
- Replacing the default cross-encoder. Late interaction is opt-in via `SearchConfig`.

## Decisions

- Two-stage architecture (vector / BM25 prune → late-interaction score on top-K). Reason: keeps storage unchanged; mirrors how ByteRover and Memanto report their numbers.
- Ship both a local and (when GA) a hosted implementation. Reason: local serves privacy-conscious users; hosted serves latency-sensitive users.
- Opt-in via `SearchConfig` recipe, not a default. Reason: model download + extra CPU latency.
- ColBERT v2 as the default local checkpoint. Reason: license, widely benchmarked, decent multilingual generalization; revisit if Wholembed v3 weights become publicly redistributable.
