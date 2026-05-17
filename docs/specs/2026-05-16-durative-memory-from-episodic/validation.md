# Validation — durative memory from episodic

## Automated tests

- `tests/utils/maintenance/test_durative_operations.py` — mock LLM emits a stable durative fact across 5 episodes; assert that dedup collapses it to a single `DurativeEdge` and `confidence` rises with each supporting episode.
- `tests/utils/maintenance/test_durative_invalidation.py` — an episode contradicting an existing durative edge sets its `expired_at`; historical query at an earlier `reference_time` still returns the original durative fact.
- `tests/search/test_durative_search.py` — `DURATIVE_HYBRID_SEARCH_RRF` recipe returns only durative edges; combined recipe returns both with `kind` distinguishable.
- `tests/test_graphiti_mock.py::test_add_episode_durative_disabled_by_default` — with `enable_durative=False`, no durative LLM call fires and `EntityEdge` counts are unchanged.
- `tests/test_graphiti_mock.py::test_add_episode_durative_cooldown` — inline derivation skips when fewer than `MIN_EPISODES_BETWEEN_DURATIVE_DERIVATIONS` episodes have elapsed for a group.

## Smoke checks

```bash
# With derivation on, a chatty user produces durative facts
uv run python -c "
import asyncio
from graphiti_core import Graphiti
async def main():
    g = Graphiti('bolt://localhost:7687', 'neo4j', 'pw', enable_durative=True)
    for i in range(20):
        await g.add_episode(name=f'msg-{i}', episode_body='Alice always prefers Adidas shoes', source_description='chat', reference_time=...)
    results = await g.search_('what does Alice prefer', config=DURATIVE_HYBRID_SEARCH_RRF)
    print(results.edges)
asyncio.run(main())
"
# Expect: at least one DurativeEdge for (Alice, prefers, Adidas)

# With derivation off, same loop produces zero durative facts
```

## Manual criteria

- Spot-check 10 derivations on real-looking data. Spurious durative facts (e.g., things mentioned once) should be rare — under 10% of emissions.
- The summary text on a `DurativeEdge` reads as a generalization, not a verbatim quote from a single episode.

## AI eval plan

- **Success criteria**:
  - LongMemEval-S "knowledge updates" subscore improves **≥ 5 points** with `enable_durative=True` vs `False` on the same `(llm, embedder, reranker)` triple.
  - Overall accuracy does not regress on other abilities (info extraction, multi-session, temporal, abstention) by more than 1 point.
  - LLM-call count per episode rises by less than 30% on average (durative is supposed to be cheap-ish).
- **Eval dataset**: LongMemEval-S full set (from the LongMemEval harness spec).
- **Regression set**: 50-question subset filtered to "knowledge updates" + "abstention" abilities.
- **Cadence**: per-PR on the regression subset for any change to `summarize_durative.py` or `derive_durative_facts`; full set before release.

## Risks & rollback

- **Failure modes**: hallucinated durative facts ("X always Y" emitted from a single mention); cost explosion on chatty users; invalidation is too aggressive and erases real preferences; overall accuracy drops because durative summaries become a noisy alternative to ground-truth episodes; durative edges fight with custom `entity_types` ontology.
- **Rollback**: flag default `False` in 0.30 means the feature is dark for everyone unless opted in. Explicit `enable_durative=False` re-disables. Revert PR cleanly removes the new module; no schema migration needed (the `DurativeEdge` rows that exist remain queryable as plain edges with a `DURATIVE` label).

## Open questions

- Edge subtype (`DurativeEdge`) vs separate node type (`DurativeFact`). The spike (step 1 of the plan) decides; lean edge.
- Confidence: derived as `len(support_episode_uuids) / window_size`, or emitted by the LLM? LLM-output is more flexible but less honest. Decide during spike.
- Cooldown policy: every N episodes, every T hours, or both? Defer to telemetry-driven default after first month of opted-in use.
- Should we surface `DurativeEdge` results separately in the MCP `search_memory_facts` response (`{"facts": [...], "durative_facts": [...]}`) or interleave? Lean separate for first release so callers can introspect.
