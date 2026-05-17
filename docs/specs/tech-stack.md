# Tech stack

## Runtime & language

- **Python `>=3.10,<4`**, managed with **uv**. Package name `graphiti-core` (currently `0.29.0`).
- Three deliverables in this repo: the core library (`graphiti_core/`), a FastAPI REST service (`server/`), and an MCP server (`mcp_server/`). Each has its own `pyproject.toml` and `Makefile`.
- Container images shipped via Dockerfile + docker-compose; CI builds run on Depot.

## Key libraries

- **Data**: `pydantic>=2.11.5` (all data models, structured-output schemas), `numpy` (vector ops).
- **Graph backends (pluggable)**: `neo4j>=5.26.0` (default), `falkordb>=1.1.2,<2.0.0`, `kuzu>=0.11.3`, Amazon Neptune (via `langchain-aws` + `boto3` + `opensearch-py`). Each backend lives under `graphiti_core/driver/{neo4j,falkordb,kuzu,neptune}/`.
- **LLM clients**: `openai>=1.91.0` (default + generic OpenAI-compatible path used by Ollama/LM Studio), `anthropic>=0.49.0`, `google-genai>=1.62.0`, `groq>=0.2.0`, plus a `GLiNER2` zero-shot NER path.
- **Embedders**: OpenAI, Azure OpenAI, Gemini, Voyage AI, sentence-transformers.
- **Cross-encoders / rerankers**: OpenAI reranker, Gemini reranker (log-prob ranking), BGE local.
- **Retry / resilience**: `tenacity>=9.0.0`. **Telemetry**: `posthog>=3.0.0` (opt-out). **Tracing**: optional `opentelemetry-{api,sdk}>=1.20.0`.
- **REST server**: `fastapi>=0.115.0`, `uvicorn>=0.44.0`, `pydantic-settings>=2.4.0`.
- **MCP server**: `mcp>=1.9.4`, served over stdio / HTTP (SSE deprecated), configured by `mcp_server/config/config.yaml` with `${VAR:default}` env expansion.
- **Tooling**: `ruff>=0.7.1`, `pyright>=1.1.404`, `pytest>=8.3.3`, `pytest-asyncio>=0.24.0`, `pytest-xdist>=3.6.1`.

## Conventions

- 4-space indent, 100-char lines, single quotes — enforced by `ruff` (`tool.ruff` in `pyproject.toml`).
- Modules and functions are `snake_case`; Pydantic models and classes are `PascalCase`.
- All ingest/search APIs are `async`. Concurrency is bounded by `SEMAPHORE_LIMIT` (default `10`) and `Graphiti(max_coroutines=...)`.
- All data is partitioned by `group_id` (alphanumeric, dash, underscore only — validated by `helpers.validate_group_id`). Driver `.clone(database=...)` switches the underlying DB without reopening connections.
- Bi-temporal fields are first-class on edges: `valid_at`, `invalid_at`, `expired_at`, `reference_time`. Nodes carry `valid_at` (episodic) and `last_summarized_at` (saga).
- Tests live in `tests/`. Integration tests use the `_int` suffix **and** `@pytest.mark.integration`; `make test` excludes them. `asyncio_mode = auto` is set in `pytest.ini`.
- LLM extraction prompts and structured-output schemas are colocated in `graphiti_core/prompts/`; each prompt is versioned (`prompt_library.extract_nodes.v1(context)`).

## Non-negotiables

- **Structured-output-only LLMs.** Extraction and dedup expect Pydantic schemas back. Other providers may work but are not supported.
- **Invalidation, never deletion.** Contradicted facts get `expired_at` / `invalid_at` set so historical queries remain answerable.
- **Pluggable backends.** Adding a graph database means implementing a `GraphDriver` plus per-operation files under `driver/<backend>/operations/`; no provider may leak into `graphiti.py` or the prompts.
- **Telemetry stays opt-out, content-free, and silent-on-failure.** Disabled automatically under `pytest`. Disable globally with `GRAPHITI_TELEMETRY_ENABLED=false`. Source of truth: `graphiti_core/telemetry/telemetry.py`.
- **Type checking and lint must pass on `graphiti_core/`.** `pyright` runs in `basic` mode against the core; `server/` uses `standard`.
- **Public-API and >500 LOC changes require an RFC issue before the PR.** All contributors must sign the Zep CLA (`Zep-CLA.md`).
- **MCP server and REST server may not import each other.** Each is independently versioned and released (`release-mcp-server.yml`, `release-server-container.yml`, `release-graphiti-core.yml`).
