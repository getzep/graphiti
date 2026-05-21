# Graphiti Node Summary Pipelines

This document outlines how entity summaries are generated today and the pragmatic changes proposed to gate expensive LLM calls using novelty and recency heuristics.

## Current Execution Flow

```
+------------------+       +-------------------------------+
| Episode Ingest   | ----> | retrieve_episodes(last=3)     |
+------------------+       +-------------------------------+
            |                         |
            v                         v
+----------------------+       +---------------------------+
| extract_nodes        |       | extract_edges             |
|  (LLM + reflexion)   |       |  (LLM fact extraction)    |
+----------------------+       +---------------------------+
            |                         |
            | dedupe candidates       |
            v                         v
+--------------------------+   +---------------------------+
| resolve_extracted_nodes  |   | resolve_extracted_edges   |
|  (deterministic + LLM)   |   |  (LLM dedupe, invalidates)|
+--------------------------+   +---------------------------+
            |                         |
            +----------- parallel ----+
                        |
                        v
        +-----------------------------------------+
        | extract_attributes_from_nodes           |
        |  - LLM attribute fill (optional)        |
        |  - LLM summary refresh (always runs)    |
        +-----------------------------------------+
```

### Current Data & Timing Characteristics

- **Previous context**: only the latest `EPISODE_WINDOW_LEN = 3` episodes are retrieved before any LLM call.
- **Fact availability**: raw `EntityEdge.fact` strings exist immediately after edge extraction; embeddings are produced inside `resolve_extracted_edges`, concurrently with summary generation.
- **Summary inputs**: the summary prompt only sees `node.summary`, node attributes, episode content, and the three-episode window. It does *not* observe resolved fact invalidations or final edge embeddings.
- **LLM usage**: every node that survives dedupe invokes the summary prompt, even when the episode is low-information or repeat content.

## Proposed Execution Flow

```
+------------------+       +-------------------------------+
| Episode Ingest   | ----> | retrieve_episodes(last=3)     |
+------------------+       +-------------------------------+
            |                         |
            v                         v
+----------------------+       +---------------------------+
| extract_nodes        |       | extract_edges             |
|  (LLM + reflexion)   |       |  (LLM fact extraction)    |
+----------------------+       +---------------------------+
            |                         |
            | dedupe candidates       |
            v                         v
+--------------------------+   +---------------------------+
| resolve_extracted_nodes  |   | build_node_deltas         |
|  (deterministic + LLM)   |   |  - new facts per node     |
+--------------------------+   |  - embed facts upfront    |
            |                 |  - track candidate flips   |
            |                 +-------------+-------------+
            |                               |
            |                               v
            |                 +---------------------------+
            |                 | resolve_extracted_edges   |
            |                 |  (uses prebuilt embeddings|
            |                 |   updates NodeDelta state)|
            |                 +-------------+-------------+
            |                               |
            |                               v
            |                 +---------------------------+
            |                 | summary_gate.should_refresh|
            |                 |  - fact hash drift         |
            |                 |  - embedding drift         |
            |                 |  - negation / invalidation |
            |                 |  - burst & staleness rules |
            |                 +------+------+------------+
            |                        |     |
            |                        |     +------------------------------+
            |                        |                                    |
            |                        v                                    v
            |         +---------------------------+      +-----------------------------+
            |         | skip summary (log cause)  |      | extract_summary LLM call    |
            |         | update metadata only      |      | update summary & metadata   |
            |         +---------------------------+      +-----------------------------+
            v
+-------------------------------+
| add_nodes_and_edges_bulk      |
+-------------------------------+
```

### Key Proposed Changes

1. **NodeDelta staging**
   - Generate fact embeddings immediately after edge extraction (`create_entity_edge_embeddings`) and group fact deltas by target node.
   - Record potential invalidations and episode timestamps within the delta so the summary gate has full context.

2. **Deterministic novelty checks**
   - Maintain a stable hash of active facts (new facts minus invalidated). Skip summarisation when the hash matches the stored value.
   - Compare pooled fact embeddings to the persisted summary embedding; trigger refresh only when cosine drift exceeds a tuned threshold.
   - Force refresh whenever the delta indicates polarity changes (contradiction/negation cues).

3. **Recency & burst handling**
   - Track recent episode timestamps per node in metadata. If multiple episodes arrive within a short window, accumulate deltas and defer the summary until the burst ends or a hard cap is reached.
   - Enforce a staleness SLA (`last_summary_ts`) so long-lived nodes eventually refresh even if novelty remains low.

4. **Metadata persistence**
   - Persist gate state on each entity (`_graphiti_meta`: `fact_hash`, `summary_embedding`, `last_summary_ts`, `_recent_episode_times`, `_burst_active`).
   - Update metadata whether the summary runs or is skipped to keep the gating logic deterministic across ingests.

5. **Observability**
   - Emit counters for `summary_skipped`, `summary_refreshed`, and reasons (unchanged hash, low drift, burst deferral, staleness). Sample a small percentage of skipped cases to validate heuristics.

## Implementation Snapshot

| Area | Current | Proposed |
| ---- | ------- | -------- |
| Summary trigger | Always per deduped node | Controlled by `summary_gate` using fact hash, embedding drift, negation, burst, staleness |
| Fact embeddings | Produced during edge resolution (parallel) | Produced immediately after extraction and reused downstream |
| Fact availability in summaries | Not available | Encapsulated in `NodeDelta` passed into summary gate |
| Metadata on node | Summary text + organic attributes | Summary text + `_graphiti_meta` (hash, embedding, timestamps, burst state) |
| Recency handling | None | Deque of recent episode timestamps + burst deferral |
| Negation detection | LLM-only inside edge resolution | Propagated into gate to force summary refresh |

These adjustments retain the existing inline execution model—no scheduled jobs—while reducing unnecessary LLM calls and improving determinism by grounding summaries in the same fact set that backs the graph.
