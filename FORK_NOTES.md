# redevops-io/graphiti — fork notes

This is a fork of [`getzep/graphiti`](https://github.com/getzep/graphiti) maintained by RedevOps.
`main` tracks upstream; RedevOps changes land on feature branches (prefixed `redevops/`) so the fork
stays cleanly rebaseable.

## G1 — source hydration (`redevops/g1-source-hydration`)

**Why.** Graphiti extracts an LLM `fact` string per `EntityEdge` and records the source episode
UUIDs on `EntityEdge.episodes`. Search returns the *facts*, which are lossy relative to the raw
turns they came from. In our knowledge-aware routing benchmark this made temporal retrieval
under-perform plain hybrid document retrieval whenever the task was really "find the session that
says this" (the extracted fact drops detail the raw turns keep). Graphiti's real edge — point-in-time
reasoning over a large, revised history — is orthogonal to that recall gap.

**What.** Optional **source hydration**: recover the raw source turns behind a fact, on demand, in
one batched round-trip — without changing the write path or the persisted schema.

- `EntityEdge.source_episodes` — a transient, non-persisted (`exclude=True`) cache of the raw source
  `EpisodicNode`s a fact was extracted from.
- `EntityEdge.hydrated_text(separator, max_chars)` — the non-lossy retrieval text: the raw source
  content when hydrated, falling back to `fact` otherwise.
- `graphiti_core/search/source_hydration.py` — pure, DB-free helpers (`collect_episode_uuids`,
  `attach_source_episodes`, `deduped_source_episodes`), unit-testable in isolation.
- `Graphiti.hydrate_edge_sources(edges, driver=None)` — batch-fetches every distinct source episode
  once and hydrates the edges in place.
- `Graphiti.search(..., hydrate_sources=True)` and `Graphiti.search_(..., hydrate_sources=True)` —
  opt-in flags; `search_` also surfaces the distinct sources on `SearchResults.episodes` when the
  search config didn't already retrieve them.

**Compatibility.** Fully backward compatible: the flags default to `False`, the new field is excluded
from serialization (never reaches the DB), and no existing signature changed meaning. Tests:
`tests/test_source_hydration.py`.

**Usage.**
```python
edges = await graphiti.search("when did they move?", hydrate_sources=True)
for edge in edges:
    fact = edge.fact                 # lossy, LLM-extracted
    raw  = edge.hydrated_text()      # non-lossy source turns (falls back to fact)
```
