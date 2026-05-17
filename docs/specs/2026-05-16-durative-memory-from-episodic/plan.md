# Plan — durative memory from episodic

## Approach

A new memory layer on top of the existing episodic / entity / community structure, distinguished by an edge subtype (`DurativeEdge`) and a separate maintenance op. The derivation pass takes a per-entity sliding window of episodes since the entity's last derivation timestamp and asks the LLM to emit durative predicates. Dedup runs through the existing `dedupe_edges` path with a label filter. Search exposes the new layer via a new `SearchConfig` field so callers opt in; existing call sites are unaffected. Inline derivation is rate-limited per group; a CLI / API entry point lets operators backfill historical data.

## Steps

1. **Spike (1–2 days)** — throwaway code on the `wizard_of_oz` example set to validate that a `summarize_durative` prompt produces useful predicates before committing API shape. Capture findings in `docs/specs/2026-05-16-durative-memory-from-episodic/spike-notes.md`.
2. Add `DurativeEdge(EntityEdge)` in `graphiti_core/edges.py` with the extra fields (`derivation_window`, `source_episode_uuids`, `confidence`) and label `'DURATIVE'`.
3. Add `graphiti_core/prompts/summarize_durative.py` with v1 prompt + Pydantic schema. Versioned (`prompt_library.summarize_durative.v1`).
4. Add `graphiti_core/utils/maintenance/durative_operations.py::derive_durative_facts(...)`. The inline call site is `Graphiti.add_episode`, gated on `self.enable_durative` and a per-group cooldown (`MIN_EPISODES_BETWEEN_DURATIVE_DERIVATIONS`, default 5).
5. Add `Graphiti.__init__(..., enable_durative: bool = False)` and propagate the flag.
6. Add `SearchConfig.durative_config: DurativeConfig` (mirrors `node_config` shape — `search_methods`, `reranker`, `sim_min_score`, etc.). Wire dispatch in `graphiti_core/search/search.py`.
7. Add recipe `DURATIVE_HYBRID_SEARCH_RRF` in `graphiti_core/search/search_config_recipes.py`.
8. Add invalidation in `utils/maintenance/edge_operations.resolve_extracted_edges`: when a newly resolved episodic edge contradicts an existing durative edge for the same `(subject_uuid, predicate)`, set the durative edge's `expired_at = now()`.
9. MCP: add `include_durative: bool = False` to `search_memory_facts`. When true, results carry a `kind: 'episodic' | 'durative'` field.
10. Tests under `tests/utils/maintenance/test_durative_operations.py` and `tests/search/test_durative_search.py`.
11. Add a `tests/evals/longmemeval/run --enable-durative` flag to re-baseline.
12. Update `docs/specs/ai-quality.md` AI surface area; update `docs/specs/mission.md` if the layered model becomes a public claim.

## Dependencies / order

Step 1 before everything (the spike may force API changes). Steps 2–3 before 4. Steps 5–7 sequenced linearly. Step 8 last in the core PR. Step 11 depends on the LongMemEval harness spec already landing.
