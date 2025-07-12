# DETAILS.md

---


🔍 **Powered by [Detailer](https://detailer.ginylil.com)** - AI-driven contextual code analysis

# Project Overview

## Purpose & Domain

This project implements **Graphiti**, a modular, scalable, and extensible system for building, querying, and maintaining **real-time, temporal knowledge graphs** optimized for AI agents. It addresses the challenge of managing complex, evolving knowledge with rich temporal semantics, enabling AI systems to maintain episodic memories, perform hybrid semantic and symbolic search, and reason over graph-structured data.

### Problem Solved
- Incremental ingestion and update of knowledge graph episodes with **bi-temporal data modeling** (validity and transaction times).
- Support for **hybrid retrieval** combining vector embeddings and symbolic graph queries.
- Integration of **multiple LLM and embedding providers** for semantic extraction, ranking, and reasoning.
- Scalable graph storage with pluggable backends (Neo4j, FalkorDB).
- Automated **entity and relation extraction**, deduplication, and temporal contradiction detection using LLMs.
- Programmatic and REST API access for graph operations.

### Target Users & Use Cases
- AI researchers and developers building **agent memory systems** or **knowledge bases**.
- Enterprises requiring **temporal knowledge management** with AI-powered semantic search.
- Developers integrating **LLM-powered graph reasoning** into applications.
- Teams needing **incremental, real-time graph updates** with rich metadata and embeddings.

### Core Business Logic & Domain Models
- **Nodes**: Entities, episodic events, communities.
- **Edges**: Relations, mentions, duplicate-of links, temporal facts.
- **Episodes**: Time-stamped data units representing knowledge updates.
- **Embeddings**: Vector representations for semantic similarity.
- **Cross-Encoders**: Models for reranking and relevance scoring.
- **Temporal Reasoning**: Validity and contradiction detection over time.
- **Graph Drivers**: Abstractions over Neo4j and FalkorDB for persistence.

---

# Architecture and Structure

## High-Level Architecture

The system is architected as a **modular, layered platform** with clear separation of concerns:

- **Core Domain Layer (`graphiti_core/`)**  
  Implements domain models, graph drivers, embedding clients, LLM clients, prompt engineering, search algorithms, and maintenance utilities.

- **Server Layer (`server/`)**  
  Provides REST API services built with FastAPI, exposing graph operations and search endpoints.

- **MCP Server Layer (`mcp_server/`)**  
  Implements a Model Context Protocol server for programmatic AI agent interaction with the knowledge graph.

- **Examples (`examples/`)**  
  Demonstration scripts and datasets illustrating usage scenarios (podcast parsing, ecommerce, Wizard of Oz narrative).

- **Tests (`tests/`)**  
  Comprehensive unit and integration tests covering core components, embedding clients, LLM clients, drivers, and evaluation workflows.

- **DevOps & CI/CD (`.github/`, `Makefile`, `docker-compose.yml`)**  
  Automation workflows for testing, linting, building, and deployment.

---

## Complete Repository Structure

```
.
├── .github/ (15 items)
│   ├── ISSUE_TEMPLATE/
│   │   └── bug_report.md
│   ├── workflows/ (9 items)
│   │   ├── cla.yml
│   │   ├── claude-code-review.yml
│   │   ├── claude.yml
│   │   ├── codeql.yml
│   │   ├── lint.yml
│   │   ├── mcp-server-docker.yml
│   │   ├── release-graphiti-core.yml
│   │   ├── typecheck.yml
│   │   └── unit_tests.yml
│   ├── dependabot.yml
│   ├── pull_request_template.md
│   └── secret_scanning.yml
├── examples/ (21 items)
│   ├── data/
│   │   └── manybirds_products.json
│   ├── ecommerce/
│   │   ├── runner.ipynb
│   │   └── runner.py
│   ├── langgraph-agent/
│   │   ├── agent.ipynb
│   │   └── tinybirds-jess.png
│   ├── podcast/
│   │   ├── podcast_runner.py
│   │   ├── podcast_transcript.txt
│   │   └── transcript_parser.py
│   ├── quickstart/
│   │   ├── README.md
│   │   ├── quickstart_falkordb.py
│   │   ├── quickstart_neo4j.py
│   │   └── requirements.txt
│   └── wizard_of_oz/
│       ├── parser.py
│       ├── runner.py
│       └── woo.txt
├── graphiti_core/ (86 items)
│   ├── cross_encoder/
│   │   ├── __init__.py
│   │   ├── bge_reranker_client.py
│   │   ├── client.py
│   │   ├── gemini_reranker_client.py
│   │   └── openai_reranker_client.py
│   ├── driver/
│   │   ├── __init__.py
│   │   ├── driver.py
│   │   ├── falkordb_driver.py
│   │   └── neo4j_driver.py
│   ├── embedder/ (6 items)
│   │   ├── __init__.py
│   │   ├── azure_openai.py
│   │   ├── client.py
│   │   ├── gemini.py
│   │   ├── openai.py
│   │   └── voyage.py
│   ├── llm_client/ (12 items)
│   │   ├── __init__.py
│   │   ├── anthropic_client.py
│   │   ├── azure_openai_client.py
│   │   ├── client.py
│   │   ├── config.py
│   │   ├── errors.py
│   │   ├── gemini_client.py
│   │   ├── groq_client.py
│   │   ├── openai_base_client.py
│   │   ├── openai_client.py
│   │   ├── openai_generic_client.py
│   │   └── utils.py
│   ├── models/ (7 items)
│   │   ├── edges/
│   │   │   ├── __init__.py
│   │   │   └── edge_db_queries.py
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   └── node_db_queries.py
│   │   └── __init__.py
│   ├── prompts/ (12 items)
│   │   ├── __init__.py
│   │   ├── dedupe_edges.py
│   │   ├── dedupe_nodes.py
│   │   ├── eval.py
│   │   ├── extract_edge_dates.py
│   │   ├── extract_edges.py
│   │   ├── extract_nodes.py
│   │   ├── invalidate_edges.py
│   │   ├── lib.py
│   │   ├── models.py
│   │   ├── prompt_helpers.py
│   │   └── summarize_nodes.py
│   ├── search/ (7 items)
│   │   ├── __init__.py
│   │   ├── search.py
│   │   ├── search_config.py
│   │   ├── search_config_recipes.py
│   │   ├── search_filters.py
│   │   ├── search_helpers.py
│   │   └── search_utils.py
│   ├── telemetry/
│   │   ├── __init__.py
│   │   └── telemetry.py
│   ├── utils/ (13 items)
│   │   ├── maintenance/ (7 items)
│   │   │   ├── __init__.py
│   │   │   ├── community_operations.py
│   │   │   ├── edge_operations.py
│   │   │   ├── graph_data_operations.py
│   │   │   ├── node_operations.py
│   │   │   ├── temporal_operations.py
│   │   │   └── utils.py
│   │   ├── ontology_utils/
│   │   │   └── entity_types_utils.py
│   │   ├── __init__.py
│   │   ├── bulk_utils.py
│   │   └── datetime_utils.py
│   ├── __init__.py
│   ├── edges.py
│   ├── errors.py
│   ├── graph_queries.py
│   ├── graphiti.py
│   ├── graphiti_types.py
│   ├── helpers.py
│   ├── nodes.py
│   └── py.typed
├── images/
│   ├── arxiv-screenshot.png
│   ├── graphiti-graph-intro.gif
│   ├── graphiti-intro-slides-stock-2.gif
│   └── simple_graph.svg
├── mcp_server/ (11 items)
│   ├── .env.example
│   ├── .python-version
│   ├── Dockerfile
│   ├── README.md
│   ├── cursor_rules.md
│   ├── docker-compose.yml
│   ├── graphiti_mcp_server.py
│   ├── mcp_config_sse_example.json
│   ├── mcp_config_stdio_example.json
│   ├── pyproject.toml
│   └── uv.lock
├── server/ (19 items)
│   ├── graph_service/ (13 items)
│   │   ├── dto/
│   │   │   ├── __init__.py
│   │   │   ├── common.py
│   │   │   ├── ingest.py
│   │   │   └── retrieve.py
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   ├── ingest.py
│   │   │   └── retrieve.py
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── main.py
│   │   └── zep_graphiti.py
│   ├── .env.example
│   ├── Makefile
│   ├── README.md
│   ├── pyproject.toml
│   └── uv.lock
├── signatures/
│   └── version1/
│       └── cla.json
├── tests/ (38 items)
│   ├── cross_encoder/
│   │   ├── test_bge_reranker_client.py
│   │   └── test_gemini_reranker_client.py
│   ├── driver/
│   │   ├── __init__.py
│   │   └── test_falkordb_driver.py
│   ├── embedder/
│   │   ├── embedder_fixtures.py
│   │   ├── test_gemini.py
│   │   ├── test_openai.py
│   │   └── test_voyage.py
│   ├── evals/ (8 items)
│   │   ├── data/
│   │   │   └── longmemeval_data/
│   │   │       ...
│   │   ├── eval_cli.py
│   │   ├── eval_e2e_graph_building.py
│   │   ├── pytest.ini
│   │   └── utils.py
│   ├── llm_client/
│   │   ├── test_anthropic_client.py
│   │   ├── test_anthropic_client_int.py
│   │   ├── test_client.py
│   │   ├── test_errors.py
│   │   └── test_gemini_client.py
│   ├── utils/
│   │   ├── maintenance/
│   │   │   ├── test_edge_operations.py
│   │   │   └── test_temporal_operations_int.py
│   │   └── search/
│   │       └── search_utils_test.py
│   ├── helpers_test.py
│   ├── test_entity_exclusion_int.py
│   ├── test_graphiti_falkordb_int.py
│   ├── test_graphiti_int.py
│   ├── test_node_falkordb_int.py
│   └── test_node_int.py
├── .env.example
├── .gitignore
├── CLAUDE.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── Dockerfile
├── LICENSE
├── Makefile
├── README.md
├── SECURITY.md
├── Zep-CLA.md
├── conftest.py
├── depot.json
├── docker-compose.test.yml
├── docker-compose.yml
├── ellipsis.yaml
├── poetry.lock
├── py.typed
├── pyproject.toml
├── pytest.ini
└── uv.lock
```

---

# Technical Implementation Details

## Core Modules

### `graphiti_core/`

- **Domain Models:**  
  - `nodes.py`, `edges.py`, and `models/` define graph entities and relationships with persistence methods.
  - Use Pydantic for validation and serialization.
  - Support episodic, entity, and community nodes with embeddings and temporal metadata.

- **Graph Drivers:**  
  - Abstract base classes in `driver/driver.py`.
  - Concrete implementations for Neo4j (`neo4j_driver.py`) and FalkorDB (`falkordb_driver.py`).
  - Drivers provide async query execution, session management, and connection lifecycle.

- **Embedding Clients (`embedder/`):**  
  - Strategy pattern with `EmbedderClient` interface.
  - Implementations for OpenAI, Azure OpenAI, Google Gemini, VoyageAI.
  - Async methods for single and batch embedding generation.
  - Configurable via Pydantic config classes.

- **LLM Clients (`llm_client/`):**  
  - Abstract `LLMClient` interface with provider-specific subclasses (OpenAI, Anthropic, Gemini, Groq).
  - Handles retries, caching, input cleaning, and structured response parsing.
  - Uses async HTTP clients or SDKs.
  - Supports JSON extraction from text responses.

- **Cross-Encoder Clients (`cross_encoder/`):**  
  - For passage ranking and reranking.
  - Implementations for BGE, Gemini, OpenAI.
  - Use async concurrency and normalization of scores.

- **Prompt Engineering (`prompts/`):**  
  - Modular prompt templates for entity extraction, edge extraction, deduplication, evaluation, and temporal reasoning.
  - Uses Pydantic models and Protocols for prompt function interfaces.
  - Supports versioning and dynamic prompt selection.

- **Search (`search/`):**  
  - Implements fulltext, vector similarity, BFS, and hybrid search strategies.
  - Provides query builders, rerankers, and filter constructors.
  - Uses numpy for vector operations and MMR reranking.

- **Maintenance Utilities (`utils/maintenance/`):**  
  - Node and edge extraction, deduplication, attribute extraction.
  - Temporal operations for date inference and contradiction detection.
  - Graph data operations like index building and data clearing.

- **Bulk Utilities (`utils/bulk_utils.py`):**  
  - Batch extraction, deduplication, and bulk insertion helpers.
  - Uses concurrency control for scalable processing.

- **Telemetry (`telemetry/`):**  
  - Captures environment and usage data for analytics.

---

## Server Components

### `server/graph_service/`

- **REST API Service:**  
  - FastAPI-based web service exposing graph ingestion and retrieval endpoints.
  - Uses dependency injection for configuration and graph client (`ZepGraphiti`).
  - DTOs (`dto/`) define request and response schemas.
  - Routers (`routers/`) implement endpoints for ingest and retrieve operations.
  - `zep_graphiti.py` extends core `Graphiti` with domain-specific methods.

- **Configuration (`config.py`):**  
  - Uses Pydantic `BaseSettings` for environment variable management.
  - Singleton pattern via `lru_cache`.

- **Main Application (`main.py`):**  
  - Initializes FastAPI app, includes routers, manages lifecycle events.

---

### `mcp_server/`

- **MCP Protocol Server:**  
  - Implements a server for programmatic AI agent interaction with the knowledge graph.
  - Uses JSON configs to define transport protocols (`sse`, `stdio`).
  - Main application in `graphiti_mcp_server.py` manages configuration, client initialization, and API endpoints.
  - Supports async episode processing, search, and CRUD operations via MCP tools and resources.

---

## Examples

- **Podcast Parsing (`examples/podcast/`):**  
  - Parses transcripts, maps speakers, creates episodic nodes.
  - Demonstrates async ingestion and bulk loading.

- **Quickstart (`examples/quickstart/`):**  
  - Scripts for connecting to FalkorDB and Neo4j.
  - Demonstrates index building, episode addition, and hybrid search.

- **Wizard of Oz (`examples/wizard_of_oz/`):**  
  - Demo script loading Wizard of Oz text as episodes.
  - Uses Anthropic LLM client for semantic extraction.
  - Illustrates async orchestration and graph ingestion.

- **Ecommerce (`examples/ecommerce/`):**  
  - Ingests product data and simulates messaging episodes.

---

# Development Patterns and Standards

- **Asynchronous Programming:**  
  - Extensive use of `async/await` for I/O-bound operations (DB queries, API calls).
  - Concurrency control with semaphores for rate-limited APIs.

- **Strategy & Adapter Patterns:**  
  - Embedding and LLM clients implement common interfaces with provider-specific adapters.
  - Cross-encoder clients follow a strategy pattern for interchangeable ranking models.

- **Configuration Management:**  
  - Pydantic models for config validation.
  - Environment variables loaded via `dotenv` and injected via dependency injection.

- **Testing:**  
  - Pytest-based unit and integration tests.
  - Extensive mocking of external APIs.
  - Integration tests for Neo4j and FalkorDB drivers.
  - Test fixtures for domain models and clients.

- **Error Handling:**  
  - Custom exceptions for rate limits, refusals, and empty responses.
  - Retry logic with exponential backoff (`tenacity`).
  - Logging of warnings and errors.

- **Code Quality:**  
  - Linting with `ruff`.
  - Static typing with `pyright`.
  - Formatting and import sorting enforced.

- **Prompt Engineering:**  
  - Modular prompt templates with versioning.
  - Use of Pydantic for prompt response validation.

---

# Integration and Dependencies

## External Libraries

- **Graph Database:**  
  - `neo4j` Python driver for Neo4j connectivity.
  - Optional FalkorDB client.

- **AI/ML APIs:**  
  - `openai` SDK for OpenAI and Azure OpenAI.
  - `anthropic` SDK for Anthropic API.
  - `google.genai` for Google Gemini.
  - `groq` for Groq API.
  - `voyageai` for VoyageAI embeddings.

- **Data Validation & Serialization:**  
  - `pydantic` for models and config.

- **Async HTTP & Retry:**  
  - `httpx` for async HTTP requests.
  - `tenacity` for retry logic.

- **Numerical Processing:**  
  - `numpy` for vector operations and similarity calculations.

- **Web Framework:**  
  - `fastapi` for REST API.

- **Testing:**  
  - `pytest`, `pytest-asyncio` for async tests.
  - `unittest.mock` for mocking.

- **DevOps & CI/CD:**  
  - GitHub Actions workflows for testing, linting, and deployment.
  - Docker and Docker Compose for containerization.

---

# Usage and Operational Guidance

## Setup & Configuration

- **Environment Variables:**  
  - Database connection (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`).
  - API keys for LLM and embedding providers (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).
  - Optional FalkorDB connection parameters.
  - Runtime flags for concurrency and parallelism.

- **Dependency Installation:**  
  - Use `uv sync --extra dev` or `pip` with `pyproject.toml`.
  - Follow Makefile targets (`make install`, `make test`, `make lint`).

- **Database Setup:**  
  - Neo4j or FalkorDB must be running and accessible.
  - Use provided scripts or API calls to build indices and constraints.

## Running the System

- **MCP Server:**  
  - Configure transport via JSON config files (`mcp_config_sse_example.json` or `mcp_config_stdio_example.json`).
  - Run `graphiti_mcp_server.py` to start the MCP server.
  - Use MCP protocol clients to interact programmatically.

- **REST API Server:**  
  - Run FastAPI app in `server/graph_service/main.py`.
  - Access endpoints for ingestion, search, and retrieval.
  - Use OpenAPI docs (`/docs`) for API exploration.

- **Examples:**  
  - Run scripts in `examples/` for demos (podcast ingestion, ecommerce data, Wizard of Oz).
  - Modify environment variables or `.env` files for credentials.

## Development & Testing

- **Testing:**  
  - Run `pytest` with `pytest-asyncio` support.
  - Integration tests require running Neo4j or FalkorDB instances.
  - Use mocks for unit tests of embedding and LLM clients.

- **Code Quality:**  
  - Run `make format` and `make lint` regularly.
  - Use `make check` to run all quality checks and tests.

- **Extending the System:**  
  - Add new embedding or LLM providers by implementing `EmbedderClient` or `LLMClient` interfaces.
  - Add new prompt templates under `graphiti_core/prompts/`.
  - Extend graph drivers for new backends by subclassing `GraphDriver`.

## Monitoring & Observability

- **Logging:**  
  - Uses Python logging with configurable levels.
  - Logs API calls, errors, and operational info.

- **Telemetry:**  
  - Captures environment and usage data for analytics.
  - Can be disabled for privacy.

- **Performance:**  
  - Async design enables high throughput.
  - Bulk operations and concurrency controls optimize ingestion and search.

---

# Actionable Insights for Developers and AI Agents

- **To Understand What This Codebase Does:**  
  - Focus on `graphiti_core/graphiti.py` as the main orchestrator.
  - Explore `graphiti_core/embedder/` and `llm_client/` for AI integration.
  - Review `graphiti_core/driver/` for database abstraction.
  - Check `server/graph_service/` for REST API implementation.
  - Use `examples/` for practical usage patterns.

- **To Modify or Extend:**  
  - Implement new embedding or LLM clients by subclassing base interfaces.
  - Add prompt templates in `graphiti_core/prompts/` following existing patterns.
  - Extend or replace graph drivers for new databases.
  - Use provided DTOs and routers to add new API endpoints.
  - Follow async patterns and concurrency controls for scalability.

- **To Run & Test:**  
  - Use `.env` files and environment variables for configuration.
  - Run `make install` and `make check` for setup and validation.
  - Use Docker Compose for local environment setup.
  - Run integration tests with real database backends.
  - Use mocks for unit testing external API interactions.

- **To Navigate the Codebase:**  
  - The directory structure is modular and layered; start from `graphiti_core/` for core logic.
  - Use `tests/` for examples of usage and validation.
  - Configuration and deployment scripts are under `.github/` and root.

---

# Summary

This project is a **comprehensive AI-powered knowledge graph platform** with strong support for temporal reasoning, semantic search, and LLM integration. It is architected for **scalability, modularity, and extensibility**, with clear layering between domain models, AI clients, graph drivers, and API services. The codebase follows modern Python async programming practices, uses Pydantic for data validation, and integrates multiple AI providers via a strategy pattern. Extensive testing and CI/CD pipelines ensure code quality and reliability. The provided examples and configuration files facilitate onboarding and deployment.

---

*End of DETAILS.md*