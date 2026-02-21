# Retrieval Trust Scoring

Trust scoring is an optional post-reranking boost that surfaces verified facts higher in search results without penalizing unverified entities.

## How It Works

After standard RRF (Reciprocal Rank Fusion) scoring, an additive boost is applied based on each result's `trust_score` property:

```
final_score = rrf_score + (trust_score × trust_weight)
```

- **`trust_score`** — a float property on Entity nodes and RELATES_TO edges in the graph (0.0–1.0). Set by your sync scripts based on your promotion pipeline. Missing/NULL values are treated as 0.0 (no boost, no penalty).
- **`trust_weight`** — a multiplier controlling boost strength. Default: `0.15`. Set via `GRAPHITI_TRUST_WEIGHT` env var or `SearchConfig.trust_weight`.

## Scoring Tiers (Reference)

These are the recommended tiers. Operators can assign any float value.

| Status | `trust_score` | Boost at weight=0.15 | Rationale |
|---|---|---|---|
| Promoted (human-approved or auto-promoted core truth) | 1.0 | +0.15 | Highest confidence — verified through promotion policy |
| Corroborated (multiple independent evidence sources) | 0.6 | +0.09 | Strong convergent evidence, not yet human-verified |
| Standard candidate (single source, in pipeline) | 0.25 | +0.0375 | Entered the promotion pipeline — minimal "we've seen this" signal |
| Not in pipeline (no `trust_score` property) | NULL → 0.0 | +0.00 | Neutral baseline. No penalty. Ranks purely on RRF relevance. |

## Calibration

With Graphiti's default `rank_const=1` and 2 search methods (BM25 + cosine similarity), RRF scores range roughly from 0.2 (rank #10 in both) to 2.0 (rank #1 in both).

At the default `trust_weight=0.15`:
- A promoted fact ranked #8 by relevance can overtake an unpromoted result at #6 — meaningful lift
- A promoted fact ranked #20 cannot overtake an unpromoted #1 — relevance still dominates
- Content pack entities with NULL trust_score are completely unaffected

## Configuration

### Environment Variable

```bash
export GRAPHITI_TRUST_WEIGHT=0.15  # default; set to 0 to disable
```

### SearchConfig

```python
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_RRF_TRUST,    # trust-boosted node search
    EDGE_HYBRID_SEARCH_RRF_TRUST,    # trust-boosted edge search
)

# Or configure directly:
config = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
        reranker=EdgeReranker.rrf,
    ),
    trust_weight=0.15,  # set to 0.0 to disable
)
```

## Populating `trust_score`

The public repo provides the reranker but does NOT populate trust scores — that's your promotion pipeline's job. Typical approach:

1. **Candidates DB** tracks which graph extractions have been reviewed/promoted
2. **Sync script** reads candidate statuses and batch-writes `trust_score` properties to the graph via Cypher:

```cypher
-- Set trust_score on promoted RELATES_TO edges
UNWIND $uuids AS uuid
MATCH ()-[r:RELATES_TO {uuid: uuid}]->()
SET r.trust_score = $score

-- Aggregate to entity nodes (max of connected edge scores)
MATCH (n:Entity)-[r:RELATES_TO]-()
WHERE r.trust_score IS NOT NULL
WITH n, max(r.trust_score) AS max_trust
SET n.trust_score = max_trust
```

3. Run the sync as part of daily graph maintenance (after dedup and timeline repair)

### UUID Matching

The sync script matches candidates to graph edges via the UUID stored in `evidence_refs_json[0].evidence_id`. Each candidate row in the candidates DB records the RELATES_TO edge UUID at extraction time, enabling a direct 1:1 match — no fuzzy matching required.

### Migration Note: FalkorDB → Neo4j (or any graph backend change)

If candidates were migrated from a prior graph backend (e.g., FalkorDB → Neo4j via re-extraction), the `evidence_refs_json[0].evidence_id` fields may contain UUIDs from the **old** backend that no longer exist in the new graph. In this case:

- Edge UUID matching will silently produce zero matches for affected candidates
- Trust scores will not be applied to the migrated facts, even if they were approved
- **Resolution:** Write a re-linking script that matches candidates to new-graph RELATES_TO edges via `subject` + `predicate` (or `fact` text similarity), then updates `evidence_refs_json` with the new UUIDs before running the sync

This is a known gap after any graph re-extraction. Candidates created directly in the target graph (post-migration) will have correct UUIDs and sync normally.

## Interaction with Content & Workflow Packs

Content packs inject context **parallel** to normal retrieval via the Pack Injector plugin hook. The pack router internally calls the same MCP search endpoints, but since content pack entities typically have no `trust_score` property, trust boosting has zero effect on pack retrieval — results rank purely on relevance.

| Retrieval path | Trust boost applies? | Notes |
|---|---|---|
| Normal agent search (MCP tools) | ✅ Yes | Promoted facts rank higher |
| Pack injection (plugin hook) | Technically yes, but neutral | Content entities have NULL trust_score |
| Pack materialization (`--materialize`) | ❌ No | Reads JSONL artifacts, bypasses graph |
| QMD fallback | ❌ No | Separate retrieval backend |

## Backwards Compatibility

- `trust_weight=0` (or unset `GRAPHITI_TRUST_WEIGHT`) → identical behavior to vanilla RRF
- Missing `trust_score` properties → no boost, no penalty, no errors
- The feature is fully opt-in: if you never set `trust_score` on any nodes, search is unchanged
