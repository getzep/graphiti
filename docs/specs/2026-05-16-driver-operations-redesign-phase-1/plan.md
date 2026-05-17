# Plan — driver operations redesign Phase 1

## Approach

Promote the draft `spec/driver-operations-redesign.md` to executable form. Build the new shape underneath the existing one, then re-route data-model methods to call the new namespace API. No public-facing signature changes in this PR. Backends migrate one at a time, Neo4j first — it has the broadest test coverage in `tests/`. The data-model methods stay intact as forwarders during Phase 1; Phase 2 (separate PR) deletes them.

## Steps

1. Move `spec/driver-operations-redesign.md` to `docs/specs/2026-05-16-driver-operations-redesign-phase-1/redesign.md` as the long-form rationale. Leave a stub at the old path pointing to the new one (or delete after one cycle).
2. Define operations ABCs in `graphiti_core/driver/operations/` — one file per object type: `entity_nodes.py`, `entity_edges.py`, `episode_nodes.py`, `episodic_edges.py`, `community_nodes.py`, `community_edges.py`, `saga_nodes.py`. Each defines `save / get_by_uuid / get_by_uuids / get_by_group_ids / delete / delete_by_uuids` as abstract async methods.
3. Build `graphiti_core/namespaces/`: `entity.py`, `episodic.py`, `community.py`, `saga.py`. Each namespace wraps the per-object ops, owns embedding generation (calls `embedder.create_batch(...)`), and threads tracing spans via the existing `Tracer`.
4. Re-implement `graphiti.nodes` and `graphiti.edges` to expose `.entity`, `.episodic`, `.community`, `.saga` attributes wired to the namespaces. The existing `NodeNamespace` / `EdgeNamespace` classes become thin facades that re-export the namespaces.
5. Add `GraphDriver.transaction()` returning an async context manager. Default implementation in `driver/driver.py` delegates to existing `.session()`; per-driver overrides can land in step 6.
6. Implement Neo4j concrete ops in `driver/neo4j/operations/*.py` — port the logic out of the data-model methods (`EntityNode.save`, `EntityEdge.save`, etc.). The methods now call into the namespace.
7. Replace the bodies of `EntityNode.save`, `EntityEdge.save`, `EpisodicNode.save`, `CommunityNode.save`, `SagaNode.save` (and the matching `get_by_*`, `delete*`) with thin forwarders to the namespace. **Signatures unchanged.**
8. Open follow-up PRs to port FalkorDB, Kuzu/LadybugDB, Neptune — one per backend.
9. Document the new namespace API in `README.md` under a "Idiomatic API" section. Add a one-paragraph migration note pointing existing users at both styles.

## Dependencies / order

Step 1 is independent. Step 2 before 3. Step 3 before 4. Step 6 before 7. Step 8 is sequenced after 7 lands cleanly. Step 9 last.
