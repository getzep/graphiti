# Repository Guidelines

## Project Structure & Module Organization
Graphiti's core library lives under `graphiti_core/`, split into domain modules such as `nodes.py`, `edges.py`, `models/`, and `search/` for retrieval pipelines. Service adapters and API glue reside in `server/graph_service/`, while the MCP integration lives in `mcp_server/`. Shared assets and collateral sit in `images/` and `examples/`. Tests cover the package via `tests/`, with configuration in `conftest.py`, `pytest.ini`, and Docker compose files for optional services. Tooling manifests live at the repo root, including `pyproject.toml`, `Makefile`, and deployment compose files.

## Build, Test, and Development Commands
- `uv sync --extra dev`: install the dev environment declared in `pyproject.toml`.
- `make format`: run `ruff` to sort imports and apply the canonical formatter.
- `make lint`: execute `ruff` plus `pyright` type checks against `graphiti_core`.
- `make test`: run the full `pytest` suite (`uv run pytest`).
- `uv run pytest tests/path/test_file.py`: target a specific module or test selection.
- `docker-compose -f docker-compose.test.yml up`: provision local graph/search dependencies for integration flows.

## Coding Style & Naming Conventions
Python code uses 4-space indentation, 100-character lines, and prefers single quotes as configured in `pyproject.toml`. Modules, files, and functions stay snake_case; Pydantic models in `graphiti_core/models` use PascalCase with explicit type hints. Keep side-effectful code inside drivers or adapters (`graphiti_core/driver`, `graphiti_core/utils`) and rely on pure helpers elsewhere. Run `make format` before committing to normalize imports and docstring formatting.

## Testing Guidelines
Author tests alongside features under `tests/`, naming files `test_<feature>.py` and functions `test_<behavior>`. Use `@pytest.mark.integration` for database-reliant scenarios so CI can gate them. Reproduce regressions with a failing test first and validate fixes via `uv run pytest -k "pattern"`. Start required backing services through `docker-compose.test.yml` when running integration suites locally.

## Commit & Pull Request Guidelines
Commits use an imperative, present-tense summary (for example, `add async cache invalidation`) optionally suffixed with the PR number as seen in history (`(#927)`). Squash fixups and keep unrelated changes isolated. Pull requests should include: a concise description, linked tracking issue, notes about schema or API impacts, and screenshots or logs when behavior changes. Confirm `make lint` and `make test` pass locally, and update docs or examples when public interfaces shift.
