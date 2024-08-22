.PHONY: install format lint test all check

# Define variables
PYTHON = python3
POETRY = poetry
PYTEST = $(POETRY) run pytest
RUFF = $(POETRY) run ruff

# Default target
all: format lint test

# Install dependencies
install:
	$(POETRY) install --with dev

# Format code
format:
	$(POETRY) run ruff check --select I --fix
	$(POETRY) run ruff format

# Lint code
lint:
	$(POETRY) run ruff check

# Run tests
test:
	$(POETRY) run pytest

# Run format, lint, and test
check: format lint test