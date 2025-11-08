# Development Commands

## Package Manager
This project uses **uv** (https://docs.astral.sh/uv/) instead of pip or poetry.

## Main Project Commands (from project root)

### Installation
```bash
# Install all dependencies including dev tools
make install
# OR
uv sync --extra dev
```

### Code Formatting
```bash
# Format code (runs ruff import sorting + code formatting)
make format
# Equivalent to:
#   uv run ruff check --select I --fix
#   uv run ruff format
```

### Linting
```bash
# Lint code (runs ruff checks + pyright type checking)
make lint
# Equivalent to:
#   uv run ruff check
#   uv run pyright ./graphiti_core
```

### Testing
```bash
# Run unit tests only (excludes integration tests)
make test
# Equivalent to:
#   DISABLE_FALKORDB=1 DISABLE_KUZU=1 DISABLE_NEPTUNE=1 uv run pytest -m "not integration"

# Run all tests including integration tests
uv run pytest

# Run only integration tests
uv run pytest -k "_int"

# Run specific test file
uv run pytest tests/test_specific_file.py

# Run specific test method
uv run pytest tests/test_file.py::test_method_name

# Run tests in parallel (faster)
uv run pytest -n auto
```

### Combined Checks
```bash
# Run format, lint, and test in sequence
make check
# OR
make all
```

## Server Commands (from server/ directory)

```bash
cd server/

# Install server dependencies
uv sync --extra dev

# Run server in development mode with auto-reload
uvicorn graph_service.main:app --reload

# Format server code
make format

# Lint server code
make lint

# Test server code
make test
```

## MCP Server Commands (from mcp_server/ directory)

```bash
cd mcp_server/

# Install MCP server dependencies
uv sync

# Run with Docker Compose
docker-compose up

# Stop Docker Compose
docker-compose down
```

## Environment Variables for Testing

### Required for Integration Tests
```bash
export TEST_OPENAI_API_KEY=...
export TEST_OPENAI_MODEL=...
export TEST_ANTHROPIC_API_KEY=...

# For Neo4j
export TEST_URI=neo4j://...
export TEST_USER=...
export TEST_PASSWORD=...
```

### Optional Runtime Variables
```bash
export OPENAI_API_KEY=...              # For LLM inference
export USE_PARALLEL_RUNTIME=true       # Neo4j parallel runtime (enterprise only)
export ANTHROPIC_API_KEY=...           # For Claude models
export GOOGLE_API_KEY=...              # For Gemini models
export GROQ_API_KEY=...                # For Groq models
export VOYAGE_API_KEY=...              # For VoyageAI embeddings
```

## Git Workflow

```bash
# Create a new branch
git checkout -b feature/your-feature-name

# After making changes, run checks
make check

# Commit changes (ensure all checks pass first)
git add .
git commit -m "Your commit message"

# Push to your fork
git push origin feature/your-feature-name
```

## Common Development Tasks

### Before Submitting PR
1. `make check` - Ensures code is formatted, linted, and tested
2. Verify all tests pass including integration tests if applicable
3. Update documentation if needed

### Adding New Dependencies
Edit `pyproject.toml`:
- Core dependencies → `[project.dependencies]`
- Optional features → `[project.optional-dependencies]`
- Dev dependencies → `[project.optional-dependencies.dev]`

Then run:
```bash
uv sync --extra dev
```

### Database Setup
- **Neo4j**: Version 5.26+ required, use Neo4j Desktop
- **FalkorDB**: Version 1.1.2+ as alternative backend

## Tool Versions
- Python: 3.10+
- UV: Latest stable
- Pytest: 8.3.3+
- Ruff: 0.7.1+
- Pyright: 1.1.404+
