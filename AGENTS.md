# Repository Guidelines

## Project Structure & Module Organization
Graphiti's core library lives under `graphiti_core/`, split into domain modules such as `nodes.py`, `edges.py`, `models/`, and `search/` for retrieval pipelines. Database drivers in `graphiti_core/driver/` support Neo4j, FalkorDB, and Neptune (plus a deprecated Kuzu driver). Additional core modules include `cross_encoder/` (reranking via BGE, OpenAI, and Gemini), `telemetry/` (OpenTelemetry tracing), `namespaces/` (namespace management), and `migrations/` (database migrations). Service adapters and API glue reside in `server/graph_service/`, while the MCP integration lives in `mcp_server/` (with its own `src/`, `tests/`, `config/`, and `docker/` subdirectories). Shared assets sit in `images/` and `examples/`. Tests cover the core package via `tests/`, with configuration in `conftest.py`, `pytest.ini`, and Docker compose files for optional services. Specifications live in `spec/` and type signatures in `signatures/`. Tooling manifests live at the repo root, including `pyproject.toml`, `Makefile`, and deployment compose files.

## Build, Test, and Development Commands
- `make install`: install the dev environment (`uv sync --extra dev`).
- `make format`: run `ruff` to sort imports and apply the canonical formatter.
- `make lint`: execute `ruff` plus `pyright` type checks against `graphiti_core`.
- `make test`: run unit tests only, excluding integration tests and disabling non-Neo4j drivers (`DISABLE_FALKORDB=1 DISABLE_KUZU=1 DISABLE_NEPTUNE=1 uv run pytest -m "not integration"`).
- `make check`: run format, lint, and test in sequence.
- `uv run pytest tests/path/test_file.py`: target a specific module or test selection.
- `docker-compose -f docker-compose.test.yml up`: provision local graph/search dependencies for integration flows.

## Coding Style & Naming Conventions
Python code uses 4-space indentation, 100-character lines, and prefers single quotes as configured in `pyproject.toml`. Modules, files, and functions stay snake_case; Pydantic models in `graphiti_core/models` use PascalCase with explicit type hints. Keep side-effectful code inside drivers or adapters (`graphiti_core/driver`, `graphiti_core/cross_encoder`, `graphiti_core/utils`) and rely on pure helpers elsewhere. Run `make format` before committing to normalize imports and docstring formatting.

## Testing Guidelines
Author tests alongside features under `tests/`, naming files `test_<feature>.py` and functions `test_<behavior>`. Integration test files use the `_int` suffix (e.g., `test_edge_int.py`, `test_node_int.py`). Use `@pytest.mark.integration` for database-reliant scenarios so CI can gate them; `make test` excludes these by default. Async tests run automatically via `asyncio_mode = auto` in `pytest.ini`. Reproduce regressions with a failing test first and validate fixes via `uv run pytest -k "pattern"`. Start required backing services through `docker-compose.test.yml` when running integration suites locally. The `mcp_server/` has its own separate test suite under `mcp_server/tests/`.

## Commit & Pull Request Guidelines
Commits use an imperative, present-tense summary (for example, `add async cache invalidation`) optionally suffixed with the PR number as seen in history (`(#927)`). Squash fixups and keep unrelated changes isolated. Pull requests should include: a concise description, linked tracking issue, notes about schema or API impacts, and screenshots or logs when behavior changes. Confirm `make lint` and `make test` pass locally, and update docs or examples when public interfaces shift.

## Cursor Cloud specific instructions

Dependencies are installed by the startup update script (`uv sync` in the repo root, `server/`, and `mcp_server/`). `uv` lives at `~/.local/bin`; if it is not on `PATH`, prefix commands with `PATH="$HOME/.local/bin:$PATH"`. Set `GRAPHITI_TELEMETRY_ENABLED=false` when running anything to avoid PostHog network calls.

### Graph databases (Neo4j + FalkorDB via Docker)
Docker has no systemd here — start the daemon manually once per VM: `sudo dockerd > /tmp/dockerd.log 2>&1 &`. Then start the DBs on the host network exactly as CI does (see `.github/workflows/unit_tests.yml`):
- `sudo docker run -d --name falkordb --network host falkordb/falkordb:latest` (port 6379)
- `sudo docker run -d --name neo4j --network host -e NEO4J_AUTH=neo4j/testpass -e NEO4J_PLUGINS='["apoc"]' neo4j:5.26-community` (bolt 7687, http 7474)

### Tests (non-obvious)
- `make test` does NOT set `DISABLE_NEO4J`, so the `graph_driver` fixture stays parametrized on Neo4j and the suite REQUIRES a reachable Neo4j at `bolt://localhost:7687`. With no Neo4j, those tests hang on the driver's connection retry (not an env bug). Run it as `NEO4J_PASSWORD=testpass make test`.
- `tests/test_add_triplet.py` has pre-existing failures (its mock embedder doesn't stub `create_batch`, so `zip(strict=True)` raises). CI never runs this file, so ignore those 11 failures — they are unrelated to environment setup.
- The authoritative, fully-green no-DB unit gate is the CI command in `.github/workflows/unit_tests.yml` (`DISABLE_NEO4J=1 DISABLE_FALKORDB=1 DISABLE_KUZU=1 DISABLE_NEPTUNE=1` plus the `--ignore` list). The DB-backed unit tests are the `database-integration-tests` job in the same file (needs Neo4j + FalkorDB, uses mock LLMs, no API key).
- `server/` and `mcp_server/` test suites are live end-to-end tests marked `integration`; they self-skip with exit code 5 when `OPENAI_API_KEY` is unset. This skip is expected, not a failure.

### Running the services (dev mode)
All three run without a valid OpenAI key for startup/health only; real ingest/search (LLM extraction + embeddings) needs a real `OPENAI_API_KEY`.
- REST API (`server/`): `OPENAI_API_KEY=<placeholder-or-real> DB_BACKEND=falkordb FALKORDB_HOST=localhost FALKORDB_PORT=6379 uv run uvicorn graph_service.main:app --reload --port 8000`. Health at `/healthcheck`, Swagger at `/docs`. `Settings` requires `OPENAI_API_KEY` to be present (any value) or startup fails.
- MCP server (`mcp_server/`): `OPENAI_API_KEY=<...> FALKORDB_URI=redis://localhost:6379 uv run python main.py --transport http --host 0.0.0.0 --port 8001 --database-provider falkordb`. Serves streamable HTTP MCP at `/mcp/` (use port 8001 if 8000 is taken by the REST server).

### Backend gotcha
The FalkorDB async driver drops the connection ("Connection closed by server") when Graphiti issues concurrent queries on one connection (e.g. the gather inside `Graphiti.search`). For local hybrid-search work, prefer the Neo4j backend, which handles concurrent queries reliably.
