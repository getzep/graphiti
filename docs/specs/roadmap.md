# Roadmap

<!-- Distilled from recent commits (post-0.29.0), open PRs, open issues, and spec/driver-operations-redesign.md.
     Not a forecast — a snapshot. Re-check before relying on it. -->

## Now

- **0.29.x hardening of the combined node+edge extraction prompt** that landed in 0.29.0 (`164030f`, `673902c`). Open PRs are closing temporal and dedup gaps it exposed: `#1492` (force `valid_at` fallback to `REFERENCE_TIME`), `#1494` (avoid FalkorDB edge rematch scans), `#1486` (unpack community results from `semaphore_gather` in `add_episode`).
- **Backend portability fixes for FalkorDB and Neo4j.** Group IDs and fulltext queries must behave identically across drivers: `#1493` (group BM25 filters for multiple group_ids), `#1475` / `#1483` (FalkorDB backticks and stopword-only inputs), `#1482` (Neo4j database parameter actually honored on `execute_query`).
- **MCP server reliability**: `#1490` (optional `reference_time` on `add_memory`), `#1491` (cascade `delete_episode` to extracted edges and orphan entities), `#1488` (bearer-token auth + configurable DNS-rebind hosts).

## Next

- **Embedder safety**: chunk + batch + average long inputs (`#1487`); the `gemini-embedding-2` family needs `batch_size=1` (`#1474`, issue `#1467`).
- **Pydantic v2 hygiene** before v3 hard-fails: migrate `SearchInterface` and friends off class-based `Config` (`#1477`, `#1478`).
- **MCP / non-OpenAI provider isolation** (`#1441`): `OpenAIRerankerClient` is the default cross-encoder and currently demands `OPENAI_API_KEY` even when the LLM/embedder is Azure, Anthropic, or Gemini.
- **`make test` scope fix** (`#1495`) so the core test command only runs core tests.

## Later

- **Driver operations redesign** (`spec/driver-operations-redesign.md`, Draft). Phase 1 is non-breaking: data models become pure Pydantic, namespace wrappers (`graphiti.nodes.entity.save(...)`, `graphiti.edges.entity.save(...)`) become the public surface, and per-driver operations ABCs replace logic threaded through `EntityNode` / `EntityEdge`. Adding a new backend should reduce to implementing one driver + its operations files.
- **Hybrid episode search via MCP** (RFC `#1427`): expose a `search_hybrid` and `get_episode` MCP tool so clients can retrieve episodes the same way they retrieve nodes/edges.
- **Historical-backfill temporal correctness** (`#1489`): three known gaps when episodes arrive out of chronological order.
- **`build_communities` performance** (`#1419`): replace the O(N) Cypher round-trips during Leiden projection on larger graphs.
- **Kuzu on Windows** (`#1469`): C-extension access violation on `add_episode` with `kuzu 0.11.3`.

## Parked

- _No publicly documented "considered and deferred" decisions at the moment._ If/when one is recorded (deprecate a backend, drop a provider, etc.), capture the one-line reason here so we don't relitigate it.
