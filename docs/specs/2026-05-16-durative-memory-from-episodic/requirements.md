# Requirements — durative memory from episodic

## Outcome

Graphiti promotes recurring or enduring patterns from episodic data into a **durative-memory layer** (e.g., "Kendra prefers Adidas shoes [persistent]" alongside the discrete episodic facts), so an agent can ask "what does X *generally* prefer" without scanning every episode. Inspired by "Beyond Dialogue Time: Temporal Semantic Memory for Personalized LLM Agents" (Jan 2026) and aligned with LongMemEval's "knowledge updates" ability.

## Users affected

Agent developers whose users carry stable preferences, roles, or habits ("vegetarian," "lives in Madrid," "works the closing shift"). Anyone whose retrieval keeps returning twenty episodic facts that all mean the same thing. The LongMemEval harness, which rewards durative-style answers on the "knowledge updates" subset.

## In scope

- `DurativeEdge(EntityEdge)` subclass with label `'DURATIVE'`, carrying `derivation_window: tuple[datetime, datetime]`, `source_episode_uuids: list[str]`, `confidence: float`.
- A new LLM-driven prompt `graphiti_core/prompts/summarize_durative.py` that takes a per-entity sliding window of episodes and emits structured durative predicates `(subject_uuid, predicate, object_uuid_or_literal, support_episode_uuids, valid_from, valid_until?)`.
- `utils/maintenance/durative_operations.py::derive_durative_facts(group_ids=None, since=None)` — runnable both inline (during `Graphiti.add_episode`) and as a batch CLI / API call.
- `Graphiti(..., enable_durative=False)` constructor flag, default **off** in 0.30 (true only after AI eval green).
- A per-group cooldown so inline derivation doesn't fire on every episode (e.g., "no more than one derivation per group per N episodes," configurable).
- `SearchConfig.durative_config: DurativeConfig` alongside the existing edge/node/episode/community configs.
- New recipe `DURATIVE_HYBRID_SEARCH_RRF` in `search_config_recipes.py`.
- Invalidation rule in `resolve_extracted_edges`: when an episodic edge contradicts an existing durative edge for the same predicate, set the durative edge's `expired_at`.

## Out of scope

- User-defined custom durative types (Phase 2; first ship a generic `DurativeEdge`).
- Reasoning *over* durative facts (chain-of-thought). Agents do that themselves.
- Streaming derivation. Phase 1 is batch + on-add only.
- Surfacing durative facts in the existing MCP `search_memory_facts` tool with no opt-in flag. Add `include_durative: bool = false` to the MCP tool signature to keep behavior backward-compatible.

## Decisions

- Durative facts live as **edges** (`DurativeEdge`), not nodes — they are predicates over entities, sharing the `EntityEdge` shape. Reason: keeps search/retrieval architecture uniform; aligns with the existing prescribed/learned ontology.
- Inline derivation runs **after** standard `add_episode` extraction, bounded by `SEMAPHORE_LIMIT` and a per-group cooldown. Reason: avoids LLM cost explosion on chatty users.
- The derivation prompt emits a Pydantic schema matching the `EntityEdge` field set so dedup reuses the existing `dedupe_edges` path with a label filter. Reason: avoids a parallel deduplication system.
- `enable_durative=False` default in 0.30, with `True` gated on a positive eval result for LongMemEval "knowledge updates." Reason: this is a behavior-changing, cost-affecting feature; opt-in until proven.
