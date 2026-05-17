# AI quality

## AI surface area

Every Graphiti ingest pass and search call invokes an LLM, embedder, or reranker. The specific surfaces are:

- **Entity extraction** from episode text (`graphiti_core/prompts/extract_nodes.py`).
- **Edge / relationship extraction** (`extract_edges.py`) and the **combined node+edge prompt** introduced in 0.29.0 (`extract_nodes_and_edges.py`, `utils/maintenance/combined_extraction.py`).
- **Node deduplication** against existing graph candidates (`dedupe_nodes.py`, `utils/maintenance/dedup_helpers.py`).
- **Edge deduplication and invalidation** — the LLM decides when a newly extracted edge supersedes an existing one and `expired_at` is set on the old edge (`dedupe_edges.py`, `utils/maintenance/edge_operations.resolve_extracted_edges`).
- **Node summary / attribute extraction** from the surrounding subgraph (`summarize_nodes.py`).
- **Saga incremental summarization** of new episodes since `last_summarized_at` (`summarize_sagas.py`, `Graphiti.summarize_saga`).
- **Community detection summaries** generated after Leiden/Louvain clustering (`utils/maintenance/community_operations.build_communities`).
- **Embedding generation** for node names and edge facts (`graphiti_core/embedder/{openai,azure_openai,gemini,voyage}.py`, plus `sentence-transformers`).
- **Cross-encoder reranking** at search time (`graphiti_core/cross_encoder/{openai,gemini,bge}_reranker_client.py`).

## Success criteria

- **Provenance is preserved**: every extracted node and edge is linked back to its source episode via `EpisodicEdge` (MENTIONS). An entity or fact with no traceable episode is a regression.
- **Bi-temporal correctness**: when a new episode contradicts an existing fact, the prior `EntityEdge` gets `expired_at` (and `invalid_at` when the new fact carries a `valid_at` boundary). Historical queries at time `T` still return the fact that was valid at `T`.
- **Deduplication**: semantically equivalent entities across episodes resolve to one `EntityNode`; duplicate edges between the same pair of entities collapse.
- **Custom ontology**: user-supplied Pydantic `entity_types` and `edge_types` are honored by extraction without prompt rewriting (`Graphiti.add_episode(entity_types=..., edge_types=...)`).
- **Retrieval**: `Graphiti.search()` returns relevant `EntityEdge` results at sub-second latency for typical graph sizes, and `search_()` returns the same hybrid result across nodes / edges / episodes / communities.

## Evaluation approach

- **End-to-end graph-building harness** lives in `tests/evals/`:
  - `eval_e2e_graph_building.py` (`build_subgraph`, `build_baseline_graph`, `eval_graph`) ingests multi-session conversations and scores graph quality.
  - `eval_cli.py` parameterizes `--multi-session-count`, `--session-length`, `--build-baseline`.
  - Datasets under `tests/evals/data/longmemeval_data/`.
- **Unit + mock suite**: `tests/test_graphiti_mock.py` is the regression baseline for `add_episode`, `search`, and dedup paths without a database. `tests/test_add_triplet.py` covers manual triplet ingest.
- **Maintenance behaviour**: `tests/utils/maintenance/` covers entity extraction, node ops, bulk utils, edge ops, and temporal operations (the invalidation contract).
- **Driver integration** runs in CI (`.github/workflows/unit_tests.yml`) with Neo4j and FalkorDB services; locally via `docker-compose.test.yml`.
- **Provider integration** tests are gated and live next to their clients: `tests/llm_client/test_anthropic_client_int.py`, `tests/cross_encoder/test_bge_reranker_client_int.py`, etc.
- Evals are not run on every PR — they're a manual / nightly check that protects the graph-building contract.

## Known failure modes

- **LLMs without structured output** (smaller local models, some chat APIs) produce malformed schemas and the extraction call fails. Use `OpenAIGenericClient` for OpenAI-compatible endpoints, but expect lower reliability on small models. Documented in `README.md` and `CLAUDE.md`.
- **Group-ID escaping** is not uniform across drivers: hyphens, backticks, and escaped underscores break FalkorDB RediSearch (`#1465`, `#1475`, `#1483`, `#1425`).
- **Gemini embedding-2 batching** silently returns the wrong number of vectors (`#1467`, fix in `#1474` forcing `batch_size=1`).
- **Long-input embeddings** crash without guard chunking (`#1487`).
- **MCP cross-encoder coupling**: `OpenAIRerankerClient` is the default even when the LLM/embedder is Azure / Anthropic / Gemini, so `OPENAI_API_KEY` is required regardless (`#1441`).
- **Neo4j `DateTime` serialization** in MCP responses fails for `search_memory_facts` (`#1438`).
- **Rate-limit pressure**: ingestion is concurrent by default (`SEMAPHORE_LIMIT=10`). Anthropic's lower RPM means ingest must drop to ~5–8; OpenAI Tier 4 can go to 20–50. Tune in `README.md`.
- **`add_episode_bulk` is not equivalent to looped `add_episode`** for edge invalidation — bulk path resolves dedup in-memory across the batch. Use single-episode mode if temporal contradictions across the batch must be detected episode-by-episode.

## Safety & privacy

- **Telemetry is opt-out, content-free, and silent on failure** (`graphiti_core/telemetry/telemetry.py`). It captures only anonymous UUID, OS / Python version, Graphiti version, and the provider/backend types selected. It never sees keys, queries, episode content, node/edge content, IPs, hostnames, or paths. Auto-disabled when `pytest` is detected. Disable globally with `GRAPHITI_TELEMETRY_ENABLED=false`.
- **Input hygiene**: all LLM inputs pass through `LLMClient._clean_input()` to strip invalid Unicode; `prompts/prompt_helpers.DO_NOT_ESCAPE_UNICODE` enforces the system-prompt convention.
- **Validators run before any DB write**: `validate_group_id`, `validate_node_labels`, `validate_entity_types`, `validate_excluded_entity_types` in `graphiti_core/helpers.py`. Label-injection coverage in `tests/test_node_label_security.py`.
- **No prompt-injection defense is documented**: episodic content is fed to the LLM verbatim by design (it's user data). Callers must treat extracted entities as potentially adversarial.
- **MCP server** currently uses hardcoded DNS-rebinding protection (`#1470`); bearer-token auth and configurable rebind hosts are in flight (`#1488`).
- **No credentials are read from telemetry, logs, or episode storage.** Set `store_raw_episode_content=False` on `Graphiti` if raw episodes contain data you do not want persisted.

## Regression checks

A change to a prompt, model, or extraction path must keep these green:

- `tests/test_graphiti_mock.py` — the large mock suite over `add_episode`, `search`, dedup, invalidation.
- `tests/utils/maintenance/` — entity extraction, node ops, bulk utils, edge ops, **temporal operations** (the bi-temporal invalidation contract is the most fragile thing in the codebase).
- `tests/test_add_triplet.py` — manual triplet ingestion (`Graphiti.add_triplet`).
- `tests/test_node_label_security.py` — label / property injection.
- Driver integration in CI (`unit_tests.yml`) with Neo4j and FalkorDB services up.
- `tests/evals/eval_e2e_graph_building.py` against a stable baseline before bumping default models or rewriting any extraction prompt.
