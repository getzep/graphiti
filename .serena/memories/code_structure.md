# Graphiti Codebase Structure

## Root Directory Layout
```
graphiti/
├── graphiti_core/          # Core library (main Python package)
├── server/                 # FastAPI REST API service
├── mcp_server/             # Model Context Protocol server for AI assistants
├── tests/                  # Test suite (unit and integration tests)
├── examples/               # Example implementations and use cases
├── images/                 # Documentation images and assets
├── signatures/             # CLA signatures
├── .github/                # GitHub Actions workflows
├── pyproject.toml          # Project configuration and dependencies
├── Makefile                # Development commands
├── README.md               # Main documentation
├── CLAUDE.md               # Claude Code assistant instructions
├── CONTRIBUTING.md         # Contribution guidelines
└── docker-compose.yml      # Docker configuration
```

## Core Library (`graphiti_core/`)

### Main Components
- **`graphiti.py`**: Main entry point containing the `Graphiti` class that orchestrates all functionality
- **`nodes.py`**: Core node/entity data structures
- **`edges.py`**: Core edge/relationship data structures
- **`graphiti_types.py`**: Type definitions
- **`errors.py`**: Custom exception classes
- **`helpers.py`**: Utility helper functions
- **`graph_queries.py`**: Graph query definitions
- **`decorators.py`**: Function decorators
- **`tracer.py`**: OpenTelemetry tracing support

### Subdirectories
- **`driver/`**: Database drivers for Neo4j, FalkorDB, Kuzu, Neptune
- **`llm_client/`**: LLM clients for OpenAI, Anthropic, Gemini, Groq
- **`embedder/`**: Embedding clients for various providers (OpenAI, Voyage, local models)
- **`cross_encoder/`**: Cross-encoder models for reranking
- **`search/`**: Hybrid search implementation with configurable strategies
- **`prompts/`**: LLM prompts for entity extraction, deduplication, summarization
- **`utils/`**: Maintenance operations, bulk processing, datetime handling
- **`models/`**: Pydantic models for data structures
- **`migrations/`**: Database migration scripts
- **`telemetry/`**: Analytics and telemetry code

## Server (`server/`)
- **`graph_service/main.py`**: FastAPI application entry point
- **`routers/`**: API endpoint definitions (ingestion, retrieval)
- **`dto/`**: Data Transfer Objects for API contracts
- Has its own `Makefile` for server-specific commands

## MCP Server (`mcp_server/`)
- **`graphiti_mcp_server.py`**: Model Context Protocol server implementation
- **`docker-compose.yml`**: Containerized deployment with Neo4j
- Has its own `pyproject.toml` and dependencies

## Tests (`tests/`)
- **Unit tests**: Standard pytest tests
- **Integration tests**: Files with `_int` suffix (require database connections)
- **`evals/`**: End-to-end evaluation scripts
- **`conftest.py`**: Pytest configuration and fixtures (at root level)

## Key Classes
From `graphiti_core/graphiti.py`:
- `Graphiti`: Main orchestrator class
- `AddEpisodeResults`: Results from adding episodes
- `AddBulkEpisodeResults`: Results from bulk episode operations
- `AddTripletResults`: Results from adding triplets

## Configuration Files
- **`pyproject.toml`**: Main project configuration (dependencies, build system, tool configs)
- **`pytest.ini`**: Pytest configuration
- **`.env.example`**: Example environment variables
- **`docker-compose.yml`**: Docker setup for development
- **`docker-compose.test.yml`**: Docker setup for testing
