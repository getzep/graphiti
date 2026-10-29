# Spanner Graph Driver for Graphiti: Feasibility and Plan

**Status:** Proposal, reviewed (not yet an RFC; see section 12 for the process)
**Date:** 2026-10-28
**Target:** graphiti-core 0.31.x, Spanner Graph (GA), `google-cloud-spanner` 3.6x+
**Related:** issue #1198 (open feature request), issue #1077 (closed duplicate), PR #2086
(Oracle 26ai driver, the first non-Cypher backend attempt), `spec/driver-operations-redesign.md`

## 1. Verdict

Feasible. No hard blocker was found in Spanner Graph or in Graphiti. But it is not a port
of the existing Cypher queries to a new dialect. Spanner Graph reads with ISO GQL and writes
only through SQL DML or the Mutation API, so the driver must be a native SQL/GQL backend that
implements Graphiti's operation interfaces directly, the way the Oracle PR (#2086) does.

Three facts drive the design:

1. Graphiti's new operations interfaces (`graphiti_core/driver/operations/`, 96 abstract
   methods) are **not** on the `add_episode` or `search` path. Those paths still build Cypher
   inline and consult two older hooks on the driver, `graph_operations_interface` (85 methods)
   and `search_interface` (14 methods), before falling back to Cypher. A Spanner driver must
   install both the new ops and the old hooks.
2. Two Cypher sites in core have no hook at all and sit on real user paths (saga lookup,
   episode removal). They need a small core change before any non-Cypher backend can run
   `add_episode(saga=...)` or `remove_episode` unmodified. The repo's test helpers also use
   raw Cypher and need provider-neutral counterparts.
3. Spanner has no GQL mutations, no multi-label nodes outside a fixed schema, and its own
   TF/IDF-style scorer rather than BM25. All three have workable mappings (section 6) but
   they are design departures, not syntax changes.

The core mappings were executed against the Spanner emulator v1.5.59 on 2026-10-28
(section 14): property-graph DDL over interleaved edge tables, GQL reads with parameters,
quantified paths with a parameterised depth, full-text `SEARCH`/`SCORE` in SQL and GQL on
nodes and edges, `COSINE_DISTANCE` on node and edge embeddings, JSON attributes, parallel
edges between the same pair, cascade deletes and partitioned DML by `group_id`.

Size, re-baselined after review against the Oracle PR (6,319 lines of driver package,
6,511 lines of tests, 457 elsewhere): expect 8,000 to 12,000 lines including tests and 6 to
9 weeks of focused engineering. The Neptune plus Kuzu ops commit (#1235, +5,966) is a poor
comparison because it reused existing Cypher and shipped no tests. Review latency at
getzep/graphiti is the larger schedule risk: no new in-tree backend has merged since August
2025, and the ArcadeDB driver PR has waited seven months.

## 2. What Graphiti actually requires of a backend

Findings from the audit of `graphiti_core/` at HEAD (0.31.2), confirmed by an independent
second pass.

### 2.1 Three code paths reach the driver

| Path | Who uses it | Hook consulted | Falls back to raw Cypher? |
|---|---|---|---|
| `graphiti.nodes.*` / `graphiti.edges.*` namespaces | Public API only, nothing internal | new `driver.<kind>_ops` (96 methods) | No. Raises `NotImplementedError` if ops are missing |
| `nodes.py`, `edges.py` model methods and `utils/bulk_utils.py` | `add_episode`, `add_episode_bulk`, `remove_episode`, communities, sagas | legacy `driver.graph_operations_interface` | Yes. 64 guarded sites. Most catch `NotImplementedError`; the bulk save hook and the two `*_load_embeddings_bulk` hooks do not |
| `search/search_utils.py` | `search`, `search_`, dedup during ingestion | legacy `driver.search_interface` | Yes. 12 guarded sites; five call the hook without try/except, so every hook method must exist. Seven more `execute_query` calls have no hook but are test-only |

Supporting facts:

- `GraphProvider` is switched on at 119 sites in 13 core files. Unknown providers fall
  through to Neo4j Cypher silently. Adding `GraphProvider.SPANNER` branches everywhere is the
  path Neptune and Kuzu took; it is the path the maintainers' spec wants to retire.
- `bulk_utils.add_nodes_and_edges_bulk` always calls `driver.session().execute_write(fn,
  ..., driver=driver)`, then inside `fn` uses the legacy hook's four `*_save_bulk` methods
  when present. The driver's session must call `fn(tx, *args, **kwargs)` with a transaction
  object first and pass the `driver=` keyword through. The callback calls the embedder inside
  the transaction, so the driver must not use the client's retrying `run_in_transaction`;
  a buffered mutation batch committed after the callback returns is the right shape.
- Rows reaching the bulk hook have `attributes` flattened into top-level keys; the Spanner
  adapter must rebuild the JSON `attributes` column from the non-core keys.
- `GraphMaintenanceOperations.remove_communities` is concrete, not abstract, and contains
  Cypher. The Spanner graph ops class must override it.
- Record parsers (`driver/record_parsers.py`) expect flat dicts with fixed keys and an
  `attributes` dict. The Oracle driver and Kuzu both ship their own parsers because they
  store attributes as JSON. Spanner will do the same.
- Pagination in core is descending: `uuid < $cursor ORDER BY uuid DESC`.

### 2.2 Legacy hook methods with no counterpart among the 96 ops methods

These are consulted on real paths and must be implemented explicitly in the Spanner legacy
adapter. They are the reason the adapter is not purely mechanical.

| Legacy method | Consulted from | When it matters |
|---|---|---|
| `node_get_by_names` | `nodes.py:646` via `node_operations.py:554` | Every `add_episode` dedup. Wrapped in `except Exception`, so a missing implementation silently degrades dedup quality |
| `node_get_by_attribute_values` | `node_operations.py:498` | Only when identity properties are configured |
| `saga_get_previous_episode_uuid` | `graphiti.py:552` | `add_episode(saga=...)` |
| `saga_get_episode_contents` | `graphiti.py:583` | `summarize_saga` |
| `get_embeddings_for_communities` (search hook) | `search_utils.py:1993` | Community MMR reranking |
| `node_load_embeddings_bulk`, `edge_load_embeddings_bulk` | `search_utils.py:1956`, `:2034` | MMR; called without try/except |

Public-API-only legacy methods (not on the ingestion or search path): `edge_get_between_nodes`,
`edge_get_by_node_uuid`, `community_node_load_name_embedding`.

### 2.3 Unguarded Cypher sites in core and tests

| Site | On a user path? | Fix |
|---|---|---|
| `graphiti.py:518` `_get_or_create_saga` (Saga lookup by name and group) | Yes, whenever `add_episode(..., saga=...)` is used | New non-abstract `SagaNodeOperations.get_by_name` (section 8) |
| `graphiti.py:2197` `remove_episode` MENTIONS count per entity | Yes, `remove_episode` | New non-abstract `EpisodeNodeOperations.count_mentions_of_entity` (section 8). Counting edges, not distinct episodes, preserves current semantics |
| `utils/maintenance/edge_operations.py:1116,1141` `filter_existing_duplicate_of_edges` | No. Only `tests/test_graphiti_mock.py` calls it | Skip for Spanner in that test |
| `search_utils.get_relevant_nodes`, `get_relevant_edges`, `get_edge_invalidation_candidates` | No. Test-only | Skip for Spanner in `test_graphiti_mock.py` |
| `tests/helpers_test.py:200-225` `get_node_count`, `get_edge_count` | Test harness, used 20+ times by `test_node_int.py` and `test_edge_int.py` | Add a Spanner branch running SQL `COUNT(*)` |

The Oracle PR patched the first two with `if driver.provider == GraphProvider.ORACLE`
branches inside `graphiti.py`. That works but adds exactly the per-provider branching the
spec's Phase 2 wants to remove. This plan proposes provider-agnostic guards instead
(section 8).

## 3. What Spanner Graph offers

All statuses from the Spanner release notes as of 2026-10-20. Docs now live under
`docs.cloud.google.com/spanner/docs/`.

| Capability | Status | Notes for Graphiti |
|---|---|---|
| Spanner Graph, ISO GQL reads | GA (Dec 2024) | `GRAPH g MATCH ... RETURN`, params are `@name`, quantified paths `-[e:A\|B]->{1,3}`, label expressions `IS Entity & Person`, path modes `WALK` (default) and `TRAIL` |
| GQL mutations (INSERT/SET/DELETE/MERGE) | Not available | All writes via SQL DML (`INSERT OR UPDATE`) or Mutation API `insert_or_update` |
| Property graph over tables | GA | `CREATE OR REPLACE PROPERTY GRAPH`; no `ALTER PROPERTY GRAPH`. Edges are tables with `SOURCE KEY` / `DESTINATION KEY`; column names can be exposed under different property names with `PROPERTIES (col AS name)` |
| Dynamic labels and properties | GA (Sep 2025) | One STRING label per row, JSON properties, lowercase-only names, values come back as JSON, and a dynamic-label node table cannot coexist with other node tables. Not a fit for `:Entity:Person` plus typed columns |
| Full-text search | GA (Oct 2024), in GQL since Dec 2024 | `TOKENLIST` generated columns, `CREATE SEARCH INDEX`, `SEARCH()` and `SCORE()` on nodes and edges. `SCORE` is Spanner's own TF/IDF-style ranker with `idf_weight` and `bigram_weight` options; not BM25. Non-text columns are indexed with `TOKEN()` and filtered with `ARRAY_INCLUDES_ANY`. Read-only transactions only |
| Exact vector search (KNN) | GA (Oct 2024) | `COSINE_DISTANCE` in SQL and GQL on nodes and edges; both arguments must share the element type, zero vectors raise. Matches what FalkorDB, Kuzu and Neptune drivers do today (brute force plus threshold) |
| ANN vector indexes | GA (Mar 2025), pre-filtering May 2025 | Nodes only in GQL; distance must be the sole ORDER BY key; `min_score` becomes a post-filter; `ARRAY<FLOAT32>(vector_length=>N)` required |
| JSON, ARRAY<STRING>, TIMESTAMP | GA | TIMESTAMP is UTC, nanosecond; client returns tz-aware UTC datetimes, which `parse_db_date` accepts |
| Interleaved tables | GA | Child's leading key columns must match the parent's key by name and type; parent row must exist before a child row is inserted; `ON DELETE CASCADE` only with `INTERLEAVE IN PARENT` |
| Python client | 3.71.0 (Sep 2026), Python >=3.10 | Sync API is stable. Native asyncio client added in 3.64.0 (Apr 2026) and marked experimental |
| Emulator | v1.5.59 (Oct 2026) | Supports graphs, dynamic labels, JSON, vector indexes and `APPROX_*`; search indexes are partial (scoring differs); one read-write transaction at a time; rejects a zero lower bound on quantified paths |
| Free tier | 90-day trial instance | Needs a billing account; includes Graph, FTS, KNN, ANN. Not a standing CI option. After the trial, Graph needs the paid Enterprise edition |
| Quotas | | 80,000 mutations per commit (one per column written plus every index entry, including cascaded child rows), 10 MiB per non-commit request, 100 MiB commit |

Dependency check: the Spanner client needs `protobuf>=6.33.5`, `grpcio`, `opentelemetry-sdk`,
`google-cloud-monitoring`. The repo's `uv.lock` already resolves protobuf 6.33.5, grpcio 1.76.0
and opentelemetry-sdk 1.39.1, so no pin conflict is expected.

## 4. Architecture decision

**Decision:** Build `graphiti_core/driver/spanner/` as a native SQL/GQL backend. The primary
surface is the eleven new ops classes, which is what the maintainers' Phase 2 will call. The
two legacy hooks are thin adapters over those ops plus the handful of extra methods in
section 2.2, and are explicitly disposable once Phase 2 lands. The driver never translates
Cypher. `execute_query` accepts GoogleSQL and GQL only and raises a clear error if it receives
Cypher, with a test that proves ingestion and search never trigger that error.

Rejected alternatives:

| Alternative | Why not |
|---|---|
| Translate Cypher to GQL/SQL inside `execute_query` | No GQL writes exist, so every MERGE/SET/UNWIND would need a parser plus a planner. Fragile, and the Cypher strings in core are not stable |
| Add `GraphProvider.SPANNER` branches at the 119 switch sites, plus provider query variants in `node_db_queries.py` and `edge_db_queries.py` as CONTRIBUTING step 5 asks | Roughly doubles the query templates in core, conflicts with the maintainers' Phase 2 plan, and still cannot express writes since the branches are Cypher-shaped. The Oracle PR made the same deviation; the RFC should say so openly |
| Single node table with `DYNAMIC LABEL` and `DYNAMIC PROPERTIES` (LangChain flexible schema) | Only one dynamic label per row, lowercase-only names, JSON-typed values, and the table cannot coexist with other node tables. Graphiti needs `Entity` plus user labels plus typed, indexed columns |
| One Spanner database per `group_id` (FalkorDB style) | Schema DDL per tenant is slow and expensive on Spanner, and the open FalkorDB routing bugs (#1795, #1842, #1876, #2033) show how fragile that model is in Graphiti. Use a `group_id` column |
| Wait for the maintainers' Phase 2 cleanup | No date, and Phase 2 is the only path that would make this smaller. Design so the work ports forward: the ops classes are exactly what Phase 2 will call |

## 5. Schema design

Fixed tables per Graphiti node kind and edge kind. The Oracle PR is the reference precedent
(JSON attributes, fixed typed columns, SQL property graph); Kuzu uses the same array-label
and JSON-attribute pattern but is deprecated and should not be cited as the model. Edge
tables are interleaved in their source node table, which is Spanner's recommended graph
layout and gives `ON DELETE CASCADE` for outgoing edges. Embedding dimension comes from a
driver parameter that defaults to `EMBEDDING_DIM`; when it is `None` the columns are plain
`ARRAY<FLOAT32>` (exact KNN still works, ANN does not).

```sql
CREATE TABLE Entity (
  uuid            STRING(MAX) NOT NULL,                  -- user-supplied uuids may exceed 36 chars
  group_id        STRING(MAX) NOT NULL,
  name            STRING(MAX) NOT NULL,
  labels          ARRAY<STRING(MAX)> NOT NULL,          -- user labels + 'Entity'
  summary         STRING(MAX),
  attributes      JSON,                                  -- dynamic attributes, decoded in Python
  name_embedding  ARRAY<FLOAT32>(vector_length=>1024),
  created_at      TIMESTAMP   NOT NULL,
  search_tokens   TOKENLIST AS (TOKENIZE_FULLTEXT(CONCAT(name, ' ', COALESCE(summary, '')))) HIDDEN,
  group_tokens    TOKENLIST AS (TOKEN(group_id)) HIDDEN,          -- exact-match filter inside the search index
  label_tokens    TOKENLIST AS (TOKEN(labels)) HIDDEN,
) PRIMARY KEY (uuid);
CREATE INDEX EntityByGroup ON Entity(group_id, uuid DESC);
CREATE SEARCH INDEX EntitySearch ON Entity(search_tokens, group_tokens, label_tokens);

CREATE TABLE Episodic (
  uuid, group_id, name, source, source_description, content, valid_at, created_at,
  entity_edges ARRAY<STRING(MAX)>, episode_metadata JSON,
  search_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(CONCAT(content, ' ', source, ' ', source_description))) HIDDEN,
  group_tokens  TOKENLIST AS (TOKEN(group_id)) HIDDEN
) PRIMARY KEY (uuid);
CREATE INDEX EpisodicByGroupValidAt ON Episodic(group_id, valid_at DESC);
CREATE SEARCH INDEX EpisodicSearch ON Episodic(search_tokens, group_tokens);

CREATE TABLE Community (uuid, group_id, name, summary, name_embedding, created_at,
                        search_tokens, group_tokens) PRIMARY KEY (uuid);
CREATE SEARCH INDEX CommunitySearch ON Community(search_tokens, group_tokens);

CREATE TABLE Saga (uuid, group_id, name, summary, first_episode_uuid, last_episode_uuid,
                   last_summarized_at, last_summarized_episode_valid_at, created_at) PRIMARY KEY (uuid);
CREATE INDEX SagaByGroupName ON Saga(group_id, name);

CREATE TABLE RelatesTo (
  uuid             STRING(MAX) NOT NULL,   -- SOURCE node uuid: Spanner requires the child's leading
                                           -- key column to carry the parent key's name and type
  edge_uuid        STRING(MAX) NOT NULL,   -- the Graphiti edge uuid
  target_node_uuid STRING(MAX) NOT NULL,
  group_id         STRING(MAX) NOT NULL,
  name             STRING(MAX) NOT NULL,
  fact             STRING(MAX) NOT NULL,
  fact_embedding   ARRAY<FLOAT32>(vector_length=>1024),
  episodes         ARRAY<STRING(MAX)>,
  attributes       JSON,
  created_at TIMESTAMP NOT NULL, valid_at TIMESTAMP, invalid_at TIMESTAMP,
  expired_at TIMESTAMP, reference_time TIMESTAMP,
  search_tokens    TOKENLIST AS (TOKENIZE_FULLTEXT(CONCAT(name, ' ', fact))) HIDDEN,
  group_tokens     TOKENLIST AS (TOKEN(group_id)) HIDDEN,
) PRIMARY KEY (uuid, edge_uuid), INTERLEAVE IN PARENT Entity ON DELETE CASCADE;
CREATE UNIQUE INDEX RelatesToByEdgeUuid ON RelatesTo(edge_uuid);
CREATE INDEX        RelatesToByTarget   ON RelatesTo(target_node_uuid, uuid);
CREATE SEARCH INDEX RelatesToSearch     ON RelatesTo(search_tokens, group_tokens);

-- Mentions   (Episodic  -> Entity),   interleaved in Episodic
-- HasMember  (Community -> Entity),   interleaved in Community
-- HasEpisode (Saga      -> Episodic), interleaved in Saga
-- NextEpisode(Episodic  -> Episodic), interleaved in Episodic
-- each: PRIMARY KEY (uuid, edge_uuid), UNIQUE INDEX on edge_uuid, INDEX on target_node_uuid

CREATE PROPERTY GRAPH graphiti
  NODE TABLES (Entity, Episodic, Community, Saga)
  EDGE TABLES (
    RelatesTo   SOURCE KEY (uuid) REFERENCES Entity (uuid)
                DESTINATION KEY (target_node_uuid) REFERENCES Entity (uuid)
                LABEL RELATES_TO
                PROPERTIES (edge_uuid AS uuid, uuid AS source_node_uuid, target_node_uuid,
                            group_id, name, fact, fact_embedding, episodes, attributes,
                            created_at, valid_at, invalid_at, expired_at, reference_time,
                            search_tokens, group_tokens),
    Mentions    SOURCE KEY (uuid) REFERENCES Episodic (uuid)
                DESTINATION KEY (target_node_uuid) REFERENCES Entity (uuid)   LABEL MENTIONS
                PROPERTIES (edge_uuid AS uuid, uuid AS source_node_uuid, target_node_uuid, group_id, created_at),
    HasMember   ... Community -> Entity   LABEL HAS_MEMBER,
    HasEpisode  ... Saga -> Episodic      LABEL HAS_EPISODE,
    NextEpisode ... Episodic -> Episodic  LABEL NEXT_EPISODE
  );
```

Design notes:

- **Labels.** `labels ARRAY<STRING>` holds the user-defined types plus `Entity`. Label
  filters become `ARRAY_INCLUDES_ANY(labels, @labels)` (or `'Person' IN UNNEST(n.labels)`
  in GQL). Implementation note: the emulator rejects `ARRAY_INCLUDES_ANY` on a TOKENLIST
  column, so queries filter on the source columns (`labels`, `group_id`); the `TOKEN()`
  columns stay in the search index so the index can still serve those predicates.
- **Attributes.** Plain JSON column, decoded in a Spanner-specific record parser. Not
  `DYNAMIC PROPERTIES`, because of the lowercase-name rule and JSON-typed values.
  `SearchFilters.property_filters` is declared but unused in core, so nothing is lost today.
- **Edge keys.** PK `(uuid, edge_uuid)` for interleaving. Spanner rejects a child table
  whose leading key column does not match the parent key's name and type (`Table RelatesTo
  references parent key column uuid at incorrect position 1`, verified on the emulator). The
  property graph's `PROPERTIES (edge_uuid AS uuid, uuid AS source_node_uuid)` makes GQL see
  the Graphiti names; SQL queries alias the same way. A unique index on `edge_uuid` serves
  `get_by_uuid` and `delete_by_uuids`; the target index serves reverse traversal and
  `get_by_node_uuid`. Parallel RELATES_TO facts between the same pair are preserved because
  `edge_uuid` is in the key (the LangChain flexible schema loses them).
- **Interleaving enforces the source node.** Inserting an edge whose source row does not
  exist fails the whole commit, where Neo4j's `MATCH ... MERGE` silently writes nothing.
  Inside `add_nodes_and_edges_bulk` nodes precede edges in one batch, so this is safe. For
  single `EntityEdge.save` calls the driver raises a clear `SpannerMissingEndpointError`
  instead of silently dropping the edge. This is a deliberate behavioural difference and
  belongs in the README. The target node is not enforced by interleaving; add a `NOT
  ENFORCED` informational FK for documentation only (accepted by the emulator).
- **Detach-delete semantics and the mutation limit.** Deleting a node cascades its
  outgoing edges, and every cascaded edge row counts its own index entries (two global
  indexes plus a search index per edge table) toward the 80,000-mutation commit limit. A
  hub entity with tens of thousands of edges would exceed it. The node-delete helper
  therefore deletes incoming and outgoing edges in chunks through the target and primary
  indexes before deleting the node row. The same ordering applies to group deletes (section
  6).
- **Group isolation.** `group_id` column everywhere, in every secondary index, and as a
  `TOKEN(group_id)` TOKENLIST inside every search index so group filtering is index
  accelerated (`STORING` alone is not). Search indexes are not partitioned by `group_id`,
  because Graphiti passes `group_ids=None` to mean "all groups" and a partitioned index
  requires equality on the partition column.
- **FLOAT32 arrays.** The client rejects Python ints inside `ARRAY<FLOAT32>` values and
  parameters (`Could not parse 1 as a FLOAT32`), so the codec coerces every element to
  `float` and binds query vectors explicitly as `ARRAY<FLOAT32>`.
- **Multi-database.** `with_database(name)` maps to another Spanner database in the same
  instance. `clone()` returns `self`, like Neo4j.

## 6. Query mapping

| Graphiti operation (Neo4j today) | Spanner mechanism |
|---|---|
| `MERGE (n:Entity {uuid}) SET n:Labels SET n = $map` | `batch.insert_or_update('Entity', cols, rows)` with all columns written. Same replace-all semantics because the schema is fixed (verified: a partial-column upsert keeps omitted columns, so always write every column) |
| `UNWIND $rows ... MERGE` bulk save | Same mutation call with up to `batch_size` rows per commit; count mutations per row (columns + index entries + generated columns) and chunk to stay well under 80,000 |
| `MATCH ... DETACH DELETE n` | Chunked deletes of incoming edges (target indexes) and outgoing edges (primary key prefix), then `batch.delete('Entity', KeySet)` |
| `delete_by_group_id`, `clear_data(group_ids)` | Partitioned DML on each **edge** table by `group_id` first, then on each node table. Running it on the parent alone fails with "too many mutations" when interleaved children carry global indexes (documented PDML limitation). Non-atomic, which matches Neo4j's batched behaviour |
| `clear_data()` all | Same order without filter, or drop and recreate tables |
| `get_by_uuid(s)`, `get_by_group_ids(limit, cursor)` | SQL `SELECT ... WHERE uuid IN UNNEST(@uuids)`; cursor is `uuid < @cursor ORDER BY uuid DESC LIMIT @n`, matching core |
| `get_by_names`, `get_by_attribute_values` (legacy only) | SQL on `Entity` by `name IN UNNEST(@names)`; `JSON_VALUE(attributes, '$.key') = @v` per configured identity property |
| `load_embeddings`, `*_load_embeddings_bulk`, `get_embeddings_for_*` | `SELECT uuid, name_embedding FROM Entity WHERE uuid IN UNNEST(@uuids)` and the edge/community equivalents |
| Node fulltext (`db.index.fulltext.queryNodes`) | `SELECT ..., SCORE(search_tokens, @q) AS score FROM Entity WHERE SEARCH(search_tokens, @q) AND group_id IN UNNEST(@g) [AND ARRAY_INCLUDES_ANY(labels, @labels)] ORDER BY score DESC LIMIT @k`, executed in a read-only snapshot (filters on the source columns; `ARRAY_INCLUDES_ANY` on TOKENLIST columns is rejected by the emulator) |
| Edge fulltext (`queryRelationships`) | Same on `RelatesTo`; join `Entity` twice only when endpoint filters are set |
| `build_fulltext_query` (Lucene syntax today) | Sanitise to Spanner rquery syntax: tokens joined with `OR` (rquery defaults to AND; both `OR` and `\|` work, verified), Lucene operators stripped, truncated at `max_query_length`. Group filtering moves out of the query string into a `group_id IN UNNEST(@g)` predicate |
| Node/edge/community similarity (`vector.similarity.cosine` + `min_score`) | SQL on the base table: `(2 - SAFE.COSINE_DISTANCE(name_embedding, @v)) / 2 AS score ... WHERE score > @min ORDER BY score DESC LIMIT @k`, `@v` bound as `ARRAY<FLOAT32>`. The `(1 + cos) / 2` scale and the strict `>` match Neo4j's `vector.similarity.cosine` and the FalkorDB driver, so the default `min_score` of 0.6 keeps the same meaning. `SAFE.` turns a stored zero vector into NULL instead of an error; an all-zero query vector returns `[]` without querying. Exact KNN; parity with FalkorDB/Kuzu/Neptune. SQL rather than GQL: on the emulator a GQL `RETURN ... ORDER BY score DESC` over edge embeddings came back unsorted, so the limited rows are also sorted in Python. ANN via `APPROX_COSINE_DISTANCE` plus vector index is a follow-up for nodes |
| Node BFS (`[:RELATES_TO|MENTIONS*1..N]`) | `GRAPH graphiti MATCH TRAIL (o WHERE o.uuid IN UNNEST(@origins))-[:RELATES_TO\|MENTIONS]->{1,@depth}(n:Entity) WHERE n.group_id = o.group_id AND n.group_id IN UNNEST(@g) RETURN DISTINCT n.uuid, ... LIMIT @k`. `TRAIL` matches Cypher's no-repeated-relationship semantics (Spanner's default `WALK` does not). The same-group predicate mirrors the Neo4j query and prevents cross-group leaks when `group_ids` is `None`. The depth bound can be a query parameter (verified on the emulator) |
| Edge BFS (`*0..N-1` then `-[e:RELATES_TO]->`) | SQL over `GRAPH_TABLE(...)`: two `TRAIL` patterns combined with `UNION DISTINCT` (depth-1 edges directly, then `{1,@depth_minus_1}` hops followed by the edge), returning edge keys only, then a join on `RelatesTo` for the full rows. Emulator 1.5.59 does accept a `{0,n}` lower bound (an older emulator issue reported otherwise), but `RETURN DISTINCT` rejects ARRAY/JSON columns and a `LIMIT` after a GQL `UNION` binds only to the last branch, hence the key-only projection and the outer SQL `LIMIT`. Node BFS uses the same `GRAPH_TABLE` shape with `n.group_id = origin.group_id` |
| `node_distance_reranker` | `MATCH (c:Entity {uuid: @center})-[:RELATES_TO]-(n:Entity WHERE n.uuid IN UNNEST(@uuids)) RETURN DISTINCT n.uuid`; scoring stays in Python |
| `episode_mentions_reranker` | `MATCH (:Episodic)-[:MENTIONS]->(n:Entity WHERE n.uuid IN UNNEST(@uuids)) RETURN n.uuid, COUNT(*)` |
| `retrieve_episodes(reference_time, last_n, saga)` | SQL on `Episodic` using `EpisodicByGroupValidAt`; saga variant joins `HasEpisode` |
| `saga_get_previous_episode_uuid`, `saga_get_episode_contents` (legacy only) | SQL over `HasEpisode` joined to `Episodic`, ordered by `created_at` then `uuid` |
| `get_mentioned_nodes`, `get_communities_by_nodes` | One-hop GQL `MATCH` |
| `get_community_clusters` | One GQL query per group returning `(n.uuid, m.uuid, edge_count)`, then reuse `driver/operations/graph_utils.label_propagation` |
| `determine_entity_community` | Two-hop GQL from the entity through `RELATES_TO` to neighbours' `HAS_MEMBER` communities, majority vote in Python |
| `SearchFilters` (labels, edge types, uuids, date ranges) | Spanner-specific builder returning `(where_fragments, params, param_types)`; date operators map one-to-one, labels use `IN UNNEST` or `ARRAY_INCLUDES_ANY`, OR-of-ANDs preserved |
| Datetimes | Pass tz-aware datetimes as TIMESTAMP params; read back `DatetimeWithNanoseconds` (UTC) and downcast to `datetime` |
| Return shapes | Explicit column lists mapped to the dict keys `record_parsers` expect; `attributes` from JSON. No `properties(n)` equivalent exists in GQL |

## 7. Driver package layout

Mirrors the Neo4j/Neptune split required by CONTRIBUTING.md and the Oracle PR's layout.

```
graphiti_core/driver/spanner_driver.py            # public import: SpannerDriver, SpannerConfig
graphiti_core/driver/spanner/
  __init__.py
  driver.py            # SpannerDriver(GraphDriver): client, pool, provider, ops wiring, execute_query guard
  config.py            # project, instance, database, graph_name, embedding_dim | None, credentials, emulator host
  session.py           # SpannerSession(GraphDriverSession): run / execute_write(fn, *args, **kwargs) / close
  transaction.py       # SpannerTransaction(Transaction): buffered mutation batch, commit after the callback
  codecs.py            # Python <-> Spanner values and param_types, JSON attributes, datetimes, float coercion
  record_parsers.py    # Spanner-specific parsers (JSON attributes, aliasing of edge_uuid/uuid)
  schema.py            # DDL generator (tables, indexes, search indexes, property graph) parameterised by dim
  fulltext.py          # Lucene-ish -> rquery sanitiser (build_fulltext_query)
  filters.py           # SearchFilters -> WHERE fragments + params
  sql/                 # query text builders grouped by table (nodes.py, edges.py, graph.py, search.py)
  operations/          # 11 ops classes implementing driver/operations/*.py ABCs
                       #   plus the two new non-abstract methods from section 8
  legacy.py            # GraphOperationsInterface + SearchInterface adapters: delegate to ops, plus the
                       #   section 2.2 methods that have no ops counterpart. Disposable after Phase 2
tests/driver/spanner/
  conftest.py          # emulator fixture (SPANNER_EMULATOR_HOST), schema create/drop per session, embedding_dim=384
  test_schema.py, test_codecs.py, test_fulltext_query.py, test_filters.py        # hermetic
  test_no_cypher_fallback.py                                                      # ingestion + search never hit execute_query with Cypher
  test_ops_*.py                                                                   # per ops class against the emulator
  test_hub_delete.py                                                              # node with many edges deletes within the mutation limit
examples/quickstart/quickstart_spanner.py
```

Driver behaviour:

- **Async.** Wrap the sync client in `asyncio.to_thread` with a bounded worker pool sized to
  `max_coroutines`. The Spanner session pool handles connection concurrency. Re-evaluate the
  experimental `AsyncClient` once it is no longer marked experimental; keep the I/O boundary
  in one module so the switch is local.
- **`execute_query`.** Accepts GoogleSQL or GQL. Routes `GRAPH`/`SELECT`/`WITH` to a
  read-only snapshot and DML to a read-write transaction. If the text looks like Cypher
  (`$param`, `MERGE`, `DETACH DELETE`, `MATCH` without a leading `GRAPH`), raise
  `SpannerCypherNotSupportedError` naming the call site. Loud failure beats silent misuse.
- **Transactions.** `transaction()` yields a buffered mutation batch committed on clean exit
  and discarded on exception. `session().execute_write(fn, *args, **kwargs)` calls
  `fn(tx, *args, **kwargs)` with that batch and commits afterwards; it never uses the
  client's retrying `run_in_transaction`, because the bulk callback calls the embedder.
  Ops with `tx=None` open their own batch. Reads never run inside a write transaction, which
  also keeps `SEARCH()` in read-only snapshots.
- **Indices.** `build_indices_and_constraints` applies the DDL idempotently (`IF NOT EXISTS`,
  `CREATE OR REPLACE PROPERTY GRAPH`) and waits on the long-running operations.
  `delete_all_indexes` drops only indexes it created.
- **Telemetry.** `Graphiti._get_provider_type` derives the provider from the class name; add
  `spanner` there (Kuzu and Neptune currently report `unknown`).

## 8. Required core changes (PR-A, small and provider-agnostic)

The first draft of this section guarded the two unhooked sites with `if driver.saga_node_ops
is not None`. Review showed that all four in-tree drivers already set every `*_ops` property
(`neo4j_driver.py:83-84`, `falkordb_driver.py:169-170`, `kuzu_driver.py:163-164`,
`neptune_driver.py:189-190`), so that guard would have rerouted Neo4j, FalkorDB, Kuzu and
Neptune away from their current queries. The corrected approach adds **non-abstract** methods
whose default raises `NotImplementedError`, so existing drivers keep their exact behaviour.

1. `graphiti_core/driver/driver.py`: add `GraphProvider.SPANNER = 'spanner'`.
2. `graphiti_core/driver/operations/saga_node_ops.py`: add non-abstract
   `get_by_name(executor, name, group_id) -> SagaNode | None` with a default that raises
   `NotImplementedError`. In `graphiti.py` `_get_or_create_saga`: try it, and on
   `NotImplementedError` fall through to the existing Cypher. Optionally implement it for
   Neo4j with the same Cypher the inline block uses today.
3. `graphiti_core/driver/operations/episode_node_ops.py`: add non-abstract
   `count_mentions_of_entity(executor, entity_uuid) -> int` (counts MENTIONS edges, not
   distinct episodes, matching the current `count(*)`). In `graphiti.py` `remove_episode`:
   try it, fall back to the inline Cypher on `NotImplementedError`.
4. `graphiti_core/graphiti.py` `_get_provider_type`: add the `spanner` label.
5. `tests/helpers_test.py`: register the backend behind `ENABLE_SPANNER` with
   `SPANNER_EMULATOR_HOST`, `SPANNER_PROJECT`, `SPANNER_INSTANCE`, `SPANNER_DATABASE`;
   construct the driver with `embedding_dim=384` (the test embedder's size); add Spanner
   branches to `get_node_count` and `get_edge_count` that run SQL `COUNT(*)`.
6. `tests/test_graphiti_mock.py`: skip the four tests that call the test-only Cypher
   helpers (`get_relevant_nodes`, `get_relevant_edges`, `get_edge_invalidation_candidates`,
   `filter_existing_duplicate_of_edges`) when the provider is Spanner.

Nothing in PR-A changes a query or a result for an existing provider. PR-A does not by itself
unblock the Oracle PR, which also needs a native `remove_episode`, deterministic MENTIONS ids
and saga cursor changes; it does remove the need for Oracle's two `if provider == ORACLE`
branches in `graphiti.py`.

Optional, to propose in the RFC rather than include: generic ops-backed adapters for the two
legacy hooks (`OpsBackedGraphOperationsInterface`, `OpsBackedSearchInterface`) would let any
driver with complete `*_ops` run the ingestion path. They cannot be fully mechanical because
of the section 2.2 methods, so the RFC should ask the maintainers whether they want those
methods promoted into the ops ABCs as non-abstract defaults.

## 9. Phased plan

| Phase | Scope | Exit criteria | Effort |
|---|---|---|---|
| 0. Spike | **Done 2026-10-28 on the emulator** (section 14). Remaining: repeat the same script against a real trial instance (ranking, GQL `ORDER BY`, mutation counting, `{0,n}` paths), and try SQL ANN on the edge table | Real-instance run matches the emulator on every check except documented ranking differences | 1 day once a trial instance exists |
| 1. RFC | Comment on #1198 with this design (appendix A), link the spike, ask for `rfc-approved`. Reference #2086 and the redesign spec. State the deviation from CONTRIBUTING step 5 openly. Sign the CLA | Maintainer response or 2 weeks elapsed | 1 day plus waiting |
| 2. PR-A | Core changes from section 8 with unit tests using `MagicMock(spec=GraphDriver)`, same style as `tests/driver/test_neo4j_driver_routing.py` | `make check` green; existing DB job unchanged; a routing test proves Neo4j still runs its inline queries | 3 to 4 days |
| 3. Driver core | `spanner/` package: config, driver, session, transaction, codecs, schema, record parsers; node and edge ops (9 classes); `legacy.py` adapters including the section 2.2 methods; `test_no_cypher_fallback`; hub-delete chunking | `tests/test_node_int.py` and `tests/test_edge_int.py` pass on the emulator with the PR-A helper changes | 2.5 to 3 weeks |
| 4. Search and maintenance | `SearchOperations` (14 methods), `GraphMaintenanceOperations` (8 incl. `remove_communities`), filters, fulltext sanitiser, `TRAIL` traversals | `tests/test_graphiti_mock.py` passes on the emulator (minus the four skipped Cypher-helper tests); `graphiti.search()` returns results for all four kinds with every reranker | 1.5 to 2 weeks |
| 5. CI, packaging, docs | `pyproject.toml` extra `spanner = ["google-cloud-spanner>=3.64"]` plus `dev`; `uv lock`; `.github/workflows/spanner-driver-tests.yml` running the emulator with `docker run --network host` like the Neo4j job; README section (including the interleaving behavioural note and the Enterprise-edition requirement) and `examples/quickstart/quickstart_spanner.py`; `TYPE_CHECKING` import guard per CONTRIBUTING | CI green on a fork; `pyright` clean; Socket firewall report reviewed | 4 to 5 days |
| 6. PR-B | Open the driver PR with `Fixes #1198`, test evidence from emulator and one real instance (90-day trial), AI-assistance disclosure | Intake labels clear; review in progress | 1 day plus review latency |
| 7. Follow-ups | ANN vector index for nodes (`APPROX_COSINE_DISTANCE`, pre-filtered by `group_id`); `AsyncClient` migration; MCP server and `server/` configuration (`mcp_server/src/config/schema.py`, `factories.py`, `server/graph_service/config.py`); `idf_weight`/`bigram_weight` tuning | Separate issues | Later |

Total: 6 to 9 weeks of focused work before review latency. Suggested order of PRs: PR-A
first because it is small and reviewable in isolation, and because it is a prerequisite for
shipping the driver out-of-tree. It is not a cure for review stalls: the same maintainers
review both.

## 10. Testing and CI

- **Hermetic unit tests** (no DB): schema DDL generation, codecs (JSON attributes,
  datetimes, FLOAT32 coercion, NULL `param_types`), fulltext sanitiser, filter builder,
  mutation counting/chunking, `execute_query` Cypher guard.
- **No-Cypher proof**: wrap `execute_query` with a recorder, run `add_episode` (with and
  without `saga`) and `remove_episode` with mocked LLM/embedder (reuse the fixtures from
  `tests/test_graphiti_mock.py`) and `search` with every method and reranker, assert the
  guard never fired. This is the single most persuasive test for reviewers.
- **Behavioural-difference tests**: edge save with a missing source node raises
  `SpannerMissingEndpointError`; a node with 5,000 edges deletes successfully (chunking);
  a 1,000-row bulk save stays under the mutation limit.
- **Emulator integration**: `gcr.io/cloud-spanner-emulator/emulator:1.5.59` in a workflow
  job; create instance and database via the admin client with `SPANNER_EMULATOR_HOST`; run
  `test_node_int.py`, `test_edge_int.py`, `test_graphiti_mock.py` with `ENABLE_SPANNER=1`.
  Known emulator gaps: scoring differs from production, a single read-write transaction at a
  time (16 concurrent writers took 2.3 s with no `ABORTED` errors in the spike), mutation
  limits not enforced, quantified paths with lower bound 0 unsupported.
- **Real instance** (manual, gated by `SPANNER_PROJECT` etc., marked `integration`): same
  suites against a 90-day trial instance before opening PR-B, plus a latency note for one
  `add_episode` and one `search`, and the `{0,n}` path and GQL `ORDER BY` checks.
- **Lint and types**: ruff (single quotes, 100 cols, `E,F,UP,B,SIM,I`) and pyright basic,
  as enforced by `lint.yml` and `typecheck.yml`.

## 11. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Maintainers object to a driver that depends on the legacy hooks their Phase 2 deletes, or to skipping CONTRIBUTING step 5 (query variants) | High | Present the ops classes as the primary surface and `legacy.py` as a disposable shim; ask in the RFC whether the section 2.2 methods should move into the ops ABCs; note that the Oracle PR made the same deviation |
| No `rfc-approved`, or review stalls for months (ArcadeDB: 7 months) | High | Ship the driver as an external package once PR-A is in a release. Keep the PR free of core behaviour changes |
| Oracle PR #2086 lands first with provider-specific branches in `graphiti.py` | Medium | Coordinate on #2023/#2086: the PR-A methods replace Oracle's two branches |
| Interleaving rejects an edge whose source node is missing where Neo4j silently drops it | Medium | Explicit error type, README note, test. Bulk path is unaffected because nodes precede edges in one batch |
| Hub-node or group deletes exceed 80,000 mutations through cascaded index entries | Medium | Delete edges in chunks before nodes; PDML on edge tables before node tables; dedicated test |
| Ranking differences: Spanner `SCORE` vs Lucene BM25 change which candidates reach RRF | Medium | Results are fused by RRF with vector search, so ranking noise is dampened. Expose `idf_weight`/`bigram_weight`. Document |
| Emulator per-query latency and serialised writers make `test_graphiti_mock.py` slow | Medium | Spike measured about 10 ms per read and 145 ms per serialised write; run node/edge tests per PR and the mock suite nightly if needed |
| Brute-force KNN cost at scale | Medium | Same as FalkorDB/Kuzu/Neptune today; ANN follow-up with pre-filtered vector index on `group_id` |
| Experimental async client changes | Low | Not used in v1; thread offload isolated in one module |
| Dependency weight (grpc, protobuf, otel, monitoring) | Low | Optional extra; lock already contains compatible versions. Watch the Socket firewall report |
| Cost for users: Graph and FTS need Spanner Enterprise edition | Informational | State it plainly in README; the free trial covers evaluation |
| `DatetimeWithNanoseconds` and `JsonObject` leaking into Pydantic models | Low | Normalise in `codecs.py`; assert plain `datetime` and `dict` in parser tests |

## 12. Contribution process (from CONTRIBUTING.md and the intake workflows)

1. Comment on the existing issue #1198 rather than opening a new one. Include the design
   summary (appendix A) and ask for the `rfc-approved` label. CONTRIBUTING lists "a new
   database driver" among features that need design discussion first; `pr-intake.yml`
   flags PRs whose linked issue lacks the label.
2. Say openly that a non-Cypher backend cannot follow CONTRIBUTING step 5 (provider query
   variants in `node_db_queries.py` / `edge_db_queries.py`) and that the ops classes are the
   substitute. The Oracle PR is the precedent.
3. Sign the CLA by commenting on the PR exactly: `I have read the CLA Document and I hereby
   sign the CLA` (the bot records it, as in commit 2b5db9e).
4. Every PR body: `Fixes #1198`, the PR template checklist, `make check` output, docs
   updated, AI-assistance disclosure.
5. Keep PR-A and PR-B separate. PR-A is reviewable in minutes and is a prerequisite for
   out-of-tree use.
6. Expect `needs-tests`, `needs-rfc` or `needs-info` labels with a 14-day response window.
7. Run `NEO4J_PASSWORD=... make test` locally; `make test` does not disable Neo4j
   (see AGENTS.md).

## 13. Open questions

Resolved by the emulator spike (section 14): rquery syntax (`OR` and `|` both work), a
parameter is accepted as the quantified-path upper bound, `NOT ENFORCED` foreign keys and
`CREATE VECTOR INDEX` are accepted, per-query latency on the emulator is about 10 ms, and 16
concurrent read-write transactions completed with no `ABORTED` errors (about 145 ms each,
serialised by the emulator).

Still open:

1. Does a `TOKENLIST` over `CONCAT(name, summary)` give acceptable recall versus two
   TOKENLIST columns in one search index? (Affects DDL only; the CONCAT form works.)
2. Does plain SQL `APPROX_COSINE_DISTANCE` work on an interleaved edge table? (ANN for
   `fact_embedding`; not needed for v1.)
3. Exact mutation accounting for a row with generated TOKENLIST columns and three indexes
   (affects chunk size; the emulator does not enforce the limit).
4. Whether the GQL `ORDER BY` anomaly seen on edge KNN (section 14) is emulator-only. The
   SQL path is the default regardless.
5. Production ranking versus the emulator's partial scorer.
6. `TRAIL` and `ARRAY_INCLUDES_ANY` on TOKENLIST were not in the spike script; add them to
   the real-instance run.
7. Whether maintainers want the section 2.2 legacy-only methods promoted into the ops ABCs.

## 14. Emulator spike results (2026-10-28)

Environment: `gcr.io/cloud-spanner-emulator/emulator:1.5.59`, `google-cloud-spanner` 3.71.0,
Python 3.12. Scripts and raw output live in the session scratchpad
(`spike/spike_v2.py`, `spike/results_v2.txt`); they should be moved into
`tests/driver/spanner/` as the first conformance tests.

| Check | Result | Note |
|---|---|---|
| 10 DDL statements incl. interleaved edge tables, unique/secondary/search indexes, `CREATE PROPERTY GRAPH` | Pass | After renaming the leading child key to `uuid` (see section 5) |
| `ALTER TABLE ... FOREIGN KEY ... NOT ENFORCED` | Pass | |
| `CREATE VECTOR INDEX ... OPTIONS (distance_type='COSINE')` | Pass | |
| `insert_or_update` with JSON, `ARRAY<STRING>`, `ARRAY<FLOAT32>`, TIMESTAMP | Pass | Elements must be `float`; partial-column upsert keeps omitted columns |
| GQL `MATCH ... WHERE n.uuid IN UNNEST(@uuids)` | Pass, 9 ms | Returns `DatetimeWithNanoseconds`, `JsonObject`, `list` |
| Label filter `'Person' IN UNNEST(n.labels)` in GQL | Pass | |
| Quantified path `{1,2}` and `{1,@depth}` | Pass, 10 ms | Parameterised bound accepted |
| Undirected one-hop `-[:RELATES_TO]-` | Pass | |
| Two parallel RELATES_TO edges between the same pair | Pass | Both returned with distinct `edge_uuid` |
| SQL KNN with `group_id` filter and `min_score` | Pass, 11 ms | |
| GQL KNN on edge `fact_embedding` | Pass with caveat | Scores correct, `ORDER BY` not honoured on the emulator |
| SQL `SEARCH`/`SCORE` with `group_id IN UNNEST` | Pass, 8 ms | `alice`, `alice OR founder`, `alice \| founder` all matched |
| `SEARCH` inside GQL on nodes; SQL `SEARCH` on edge table | Pass | |
| Reverse traversal via target index; edge lookup by `edge_uuid` | Pass | |
| `JSON_VALUE(attributes, '$.city') = @city` | Pass | |
| Node delete cascades outgoing edges only | Pass | Incoming edges remain, as designed |
| Partitioned DML `DELETE FROM Entity WHERE group_id = @g` | Pass | Interleaved edges removed by cascade on this small dataset; production ordering per section 6 still applies |
| 200-row mutation batch with embeddings and JSON | Pass, 78 ms | |
| 16 concurrent read-write transactions on 8 threads | Pass, 2.3 s total | No `ABORTED`; emulator serialises writers |

## 15. Review log

An independent adversarial review of the first draft (same day) produced fifteen findings.
All are incorporated above. The material ones:

- Edge-table key naming (blocker) was already corrected by the spike; the property-graph
  `PROPERTIES (... AS ...)` aliasing was added so GQL sees Graphiti's names.
- The first PR-A design would have rerouted all four existing drivers; replaced with
  non-abstract ops methods that default to `NotImplementedError`.
- Test harness helpers and four mock tests use raw Cypher; added to PR-A. Test embeddings
  are 384-dimensional; `embedding_dim` is now a driver parameter the fixture sets.
- Six legacy-hook methods on the ingestion and search paths have no ops counterpart; listed
  in section 2.2 and in the mapping table.
- Interleaving enforces source-node existence; partitioned deletes must start with edge
  tables; hub-node deletes need chunking.
- `TRAIL` path mode and the same-group predicate added to BFS; cursor direction corrected;
  full-text group filtering moved into `TOKEN()` columns so the search index accelerates it;
  Episodic and Community search indexes made explicit.
- Estimates re-baselined from 5,000-7,000 lines / 3-5 weeks to 8,000-12,000 lines /
  6-9 weeks against the Oracle PR's measured size.

## 16. Implementation status (2026-10-28)

Implemented on branch `feat/spanner-graph-driver` (fork `rickragv/graphiti`):

- `875ec09` PR-A core hooks (`SagaNodeOperations.get_by_name`,
  `EpisodeNodeOperations.count_mentions_of_entity`) with routing tests.
- `128a21c` core fix: `label_propagation` never terminated for two entities joined only by
  parallel edges (found while implementing community clusters; affects every backend).
- `6800e27` the driver: `graphiti_core/driver/spanner/` (foundation, eleven ops classes,
  legacy adapters), `spanner` extra, test-harness registration, 146 foundation tests.
- `e0b8359` README, quickstart, this spec, emulator CI workflow.

Verified on emulator 1.5.59: the driver's own suite plus the repository's `test_node_int.py`,
`test_edge_int.py` and `test_graphiti_mock.py` (26 passed, 3 skipped: test-only Cypher
helpers) with `ENABLE_SPANNER=1`; repo-wide ruff and pyright clean. Deviations from the
design discovered during implementation are recorded in the relevant sections above
(source-column filters, `GRAPH_TABLE` traversals, Neo4j-style cosine scale, `{0,n}` bounds
accepted, `Database.close()` run on a daemon thread).

## Appendix A. Draft RFC comment for issue #1198

> I would like to contribute a Spanner Graph backend and am asking for `rfc-approved` before
> opening a PR. Design summary:
>
> - Native GoogleSQL/GQL driver under `graphiti_core/driver/spanner/`, opt-in via
>   `graphiti-core[spanner]`. No Cypher translation. Fixed tables per node and edge kind,
>   edges interleaved in their source table, `labels ARRAY<STRING>` and `attributes JSON`
>   as in the Oracle PR, `group_id` as a column.
> - Implements all eleven `driver/operations` ABCs as the primary surface, plus thin
>   `GraphOperationsInterface` / `SearchInterface` adapters so `add_episode` and `search`
>   run without touching Cypher until Phase 2 of the operations redesign lands. Full-text via
>   Spanner search indexes, vectors via `COSINE_DISTANCE` (ANN later), BFS via GQL quantified
>   paths. Core mappings verified on the Spanner emulator.
> - Needs two small provider-agnostic core additions, which I would send as a separate first
>   PR: non-abstract `SagaNodeOperations.get_by_name` and
>   `EpisodeNodeOperations.count_mentions_of_entity`, consulted before the two remaining
>   inline queries in `graphiti.py`, plus provider-neutral count helpers in the test harness.
>   Existing drivers keep their exact queries.
> - This deviates from CONTRIBUTING step 5 (provider query variants) because a non-Cypher
>   backend cannot share those templates; the Oracle PR (#2086) makes the same deviation.
> - Tests on the Spanner emulator in CI plus manual runs on a real instance.
>
> Questions: (1) is an in-tree driver acceptable, or would you prefer an external package
> until Phase 2 lands? (2) Six legacy-hook methods on the ingestion path have no ops
> counterpart (`node_get_by_names`, `node_get_by_attribute_values`,
> `saga_get_previous_episode_uuid`, `saga_get_episode_contents`,
> `get_embeddings_for_communities`, the two `*_load_embeddings_bulk`). Would you accept
> promoting them into the ops ABCs as non-abstract defaults so any ops-complete driver works
> on that path?
