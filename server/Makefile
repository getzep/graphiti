.PHONY: install format lint test all check

# Define variables
PYTHON = python3
UV = uv
PYTEST = $(UV) run pytest
RUFF = $(UV) run ruff
MYPY = $(UV) run mypy

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
	$(MYPY) . --show-column-numbers --show-error-codes --pretty 

# Run tests
test:
	$(PYTEST)

# Run format, lint, and test
check: format lint test