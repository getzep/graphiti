# Graphiti Project Overview

## Purpose
Graphiti is a Python framework for building and querying temporally-aware knowledge graphs, specifically designed for AI agents operating in dynamic environments. It continuously integrates user interactions, structured/unstructured data, and external information into a coherent, queryable graph with incremental updates and efficient retrieval.

## Key Features
- **Bi-temporal data model**: Explicit tracking of event occurrence times
- **Hybrid retrieval**: Combining semantic embeddings, keyword search (BM25), and graph traversal
- **Custom entity definitions**: Support via Pydantic models
- **Real-time incremental updates**: No batch recomputation required
- **Multiple graph backends**: Neo4j and FalkorDB support
- **Optional OpenTelemetry tracing**: For distributed systems

## Use Cases
- Integrate and maintain dynamic user interactions and business data
- Facilitate state-based reasoning and task automation for agents
- Query complex, evolving data with semantic, keyword, and graph-based search methods

## Relationship to Zep
Graphiti powers the core of Zep, a turn-key context engineering platform for AI Agents. This is the open-source version that provides flexibility for custom implementations.

## Tech Stack
- **Language**: Python 3.10+
- **Package Manager**: uv (modern, fast Python package installer)
- **Core Dependencies**:
  - Pydantic 2.11.5+ (data validation and models)
  - Neo4j 5.26.0+ (primary graph database)
  - OpenAI 1.91.0+ (LLM inference and embeddings)
  - Tenacity 9.0.0+ (retry logic)
  - DiskCache 5.6.3+ (caching)
  
- **Optional Integrations**:
  - Anthropic (Claude models)
  - Google Gemini
  - Groq
  - FalkorDB (alternative graph database)
  - Kuzu (graph database)
  - Neptune (AWS graph database)
  - VoyageAI (embeddings)
  - Sentence Transformers (local embeddings)
  - OpenTelemetry (tracing)

- **Development Tools**:
  - Ruff (linting and formatting)
  - Pyright (type checking)
  - Pytest (testing framework with pytest-asyncio and pytest-xdist)

## Project Version
Current version: 0.22.1pre2 (pre-release)

## Repository
https://github.com/getzep/graphiti
