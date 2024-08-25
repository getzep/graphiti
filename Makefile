.PHONY: install format lint test all check

# Define variables
PYTHON = python3
POETRY = poetry
PYTEST = $(POETRY) run pytest
RUFF = $(POETRY) run ruff
MYPY = $(POETRY) run mypy

# Default target
all: format lint test

# Install dependencies
install:
	$(POETRY) install --with dev

# Format code
format:
	$(RUFF) check --select I --fix
	$(RUFF) format

# Lint code
lint:
	$(RUFF) check
	$(MYPY) ./graphiti_core --show-column-numbers --show-error-codes --pretty 

# Run tests
test:
	$(PYTEST)

# Run format, lint, and test
check: format lint test