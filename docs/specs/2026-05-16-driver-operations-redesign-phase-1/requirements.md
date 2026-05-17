# Requirements — driver operations redesign Phase 1

## Outcome

Driver implementations become the core of the codebase. Adding a new backend reduces to "implement a `GraphDriver` plus the operations files," with data models (`EntityNode`, `EntityEdge`, ...) reduced to pure Pydantic. Phase 1 ships **without breaking** existing public methods on data models; the new namespace API (`graphiti.nodes.entity.save(node)`) becomes the documented idiom.

## Users affected

Maintainers adding new backends (next likely: LadybugDB, Memgraph). Advanced users building custom drivers. Indirectly all users: the new namespace API is more discoverable in IDEs and notebooks (`graphiti.nodes.<tab>`).

## In scope

- Per-object operations ABCs under `graphiti_core/driver/operations/`: `entity_nodes.py`, `entity_edges.py`, `episode_nodes.py`, `episodic_edges.py`, `community_nodes.py`, `community_edges.py`, `saga_nodes.py`.
- Namespace wrappers under `graphiti_core/namespaces/` that own embeddings + tracing and delegate I/O to ops.
- `graphiti.nodes.entity.save(node)`, `graphiti.edges.entity.save(edge)`, ... wired off the existing `NodeNamespace` / `EdgeNamespace`.
- Per-driver concrete ops classes under `driver/{neo4j,falkordb,kuzu,neptune,ladybug}/operations/` (Neo4j first; the others can be sequenced per PR).
- **Phase 1 keeps** existing `EntityNode.save()`, `EntityEdge.save()` as thin shims that forward to the namespace API.
- `graphiti.driver.transaction()` async context manager API (default impl delegates to existing `.session()`).
- Promote the long-form rationale from `spec/driver-operations-redesign.md` to `docs/specs/2026-05-16-driver-operations-redesign-phase-1/redesign.md`; the `spec/` doc can then be deleted in a follow-up.

## Out of scope

- **Phase 2 removal** of methods on `EntityNode` / `EntityEdge`. That is a separate breaking PR slotted for 0.31 or 0.32.
- Changing the public `Graphiti.add_episode` / `search` / `search_` signatures.
- Rewriting the LLM extraction or dedup paths (`utils/maintenance/*.py`).
- Adding new operations beyond what the data models already expose.

## Decisions

- Operations classes are flat — one class per object type — not a single mega-ops class. Reason: cohesion; matches the existing `spec/driver-operations-redesign.md` choice.
- Embedding generation lives at the namespace layer, not in ops or in driver. Reason: keeps drivers focused on I/O; lets the embedder dependency stay out of driver imports.
- Transaction context manager hangs off `driver`, not `Graphiti`. Reason: matches the existing `driver.session()` mental model; lifetime is explicit.
- Phase 1 is **non-breaking**. Reason: the spec says so, existing tests rely on data-model methods, and a breaking change in the same PR as a redesign is unreviewable.
- Per-backend migration is one PR per backend after Neo4j lands. Reason: keeps each diff reviewable; non-Neo4j users aren't blocked.
