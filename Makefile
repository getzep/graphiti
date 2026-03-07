.PHONY: install format lint test test-integration all check

# Define variables
PYTHON = python3
UV = uv
PYTEST = $(UV) run pytest
RUFF = $(UV) run ruff
PYRIGHT = $(UV) run pyright

# Default target
all: format lint test

# Install dependencies
install:
	$(UV) sync --extra dev

# Format code
format:
	$(RUFF) check --select I --fix
	$(RUFF) format

# Lint code
lint:
	$(RUFF) check
	$(PYRIGHT) ./graphiti_core 

# Run tests
test:
	DISABLE_NEO4J=1 DISABLE_FALKORDB=1 DISABLE_KUZU=1 DISABLE_NEPTUNE=1 $(PYTEST) -m "not integration" -n auto

# Run integration tests using Docker for Neo4j and FalkorDB
test-integration:
	docker compose up neo4j falkordb -d --wait --wait-timeout 300
	DISABLE_KUZU=1 DISABLE_NEPTUNE=1 $(PYTEST) -m "integration" || true
	docker compose down neo4j falkordb

# Run format, lint, and test
check: format lint test
