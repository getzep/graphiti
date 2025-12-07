# Graphiti MCP Server Integration Tests

This directory contains a comprehensive integration test suite for the Graphiti MCP Server using the official Python MCP SDK.

## Overview

The test suite is designed to thoroughly test all aspects of the Graphiti MCP server with special consideration for LLM inference latency and system performance.

## Test Organization

### Core Test Modules

- **`test_inmemory_example.py`** - **Recommended** in-memory tests using FastMCP Client (fast, reliable)
- **`test_comprehensive_integration.py`** - Integration test suite using subprocess (slower, may have env issues)
- **`test_async_operations.py`** - Tests for concurrent operations and async patterns
- **`test_stress_load.py`** - Stress testing and load testing scenarios
- **`test_fixtures.py`** - Shared fixtures and test utilities
- **`test_mcp_integration.py`** - Original MCP integration tests
- **`test_configuration.py`** - Configuration loading and validation tests

### Test Categories

Tests are organized with pytest markers:

- `unit` - Fast unit tests without external dependencies
- `integration` - Tests requiring database and services
- `slow` - Long-running tests (stress/load tests)
- `requires_neo4j` - Tests requiring Neo4j
- `requires_falkordb` - Tests requiring FalkorDB
- `requires_openai` - Tests requiring OpenAI API key

## Installation

```bash
# Install test dependencies
uv add --dev pytest pytest-asyncio pytest-timeout pytest-xdist faker psutil

# Install MCP SDK (fastmcp is already a dependency of graphiti-core)
uv add mcp
```

> **Note on FastMCP**: The `fastmcp` package (v2.13.3) is a dependency of `graphiti-core` and provides the `Client` class for testing. The MCP server uses `mcp.server.fastmcp.FastMCP` which is bundled in the official `mcp` package.

## Running Tests

### Quick Start (Recommended)

The fastest and most reliable way to test is using the in-memory tests:

```bash
# Run in-memory tests (fast, ~1 second)
uv run pytest tests/test_inmemory_example.py -v -s
```

This uses FastMCP's recommended testing pattern with in-memory transport, avoiding subprocess issues.

### Alternative: Subprocess-based Tests

The original test runner spawns subprocess servers. These tests may experience environment variable issues:

```bash
# Run smoke tests (may timeout due to subprocess issues)
python tests/run_tests.py smoke

# Run integration tests with mock LLM
python tests/run_tests.py integration --mock-llm

# Run all tests
python tests/run_tests.py all
```

> **Note**: The subprocess-based tests use `StdioServerParameters` which can have environment variable isolation issues. If you encounter `ValueError: invalid literal for int()` errors related to `SEMAPHORE_LIMIT` or `MAX_REFLEXION_ITERATIONS`, use the in-memory tests instead.

### Test Runner Options

```bash
python tests/run_tests.py [suite] [options]

Suites:
  unit          - Unit tests only
  integration   - Integration tests
  comprehensive - Comprehensive integration suite
  async         - Async operation tests
  stress        - Stress and load tests
  smoke         - Quick smoke tests
  all           - All tests

Options:
  --database    - Database backend (neo4j, falkordb)
  --mock-llm    - Use mock LLM for faster testing
  --parallel N  - Run tests in parallel with N workers
  --coverage    - Generate coverage report
  --skip-slow   - Skip slow tests
  --timeout N   - Test timeout in seconds
  --check-only  - Only check prerequisites
```

### Examples

```bash
# Quick smoke test with FalkorDB (default)
python tests/run_tests.py smoke

# Full integration test with Neo4j
python tests/run_tests.py integration --database neo4j

# Stress testing with parallel execution
python tests/run_tests.py stress --parallel 4

# Run with coverage
python tests/run_tests.py all --coverage

# Check prerequisites only
python tests/run_tests.py all --check-only
```

## Test Coverage

### Core Operations
- Server initialization and tool discovery
- Adding memories (text, JSON, message)
- Episode queue management
- Search operations (semantic, hybrid)
- Episode retrieval and deletion
- Entity and edge operations

### Async Operations
- Concurrent operations
- Queue management
- Sequential processing within groups
- Parallel processing across groups

### Performance Testing
- Latency measurement
- Throughput testing
- Batch processing
- Resource usage monitoring

### Stress Testing
- Sustained load scenarios
- Spike load handling
- Memory leak detection
- Connection pool exhaustion
- Rate limit handling

## Configuration

### Environment Variables

```bash
# Database configuration
export DATABASE_PROVIDER=falkordb  # or neo4j
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=graphiti
export FALKORDB_URI=redis://localhost:6379

# LLM configuration
export OPENAI_API_KEY=your_key_here  # or use --mock-llm

# Test configuration
export TEST_MODE=true
export LOG_LEVEL=INFO
```

### pytest.ini Configuration

The `pytest.ini` file configures:
- Test discovery patterns
- Async mode settings
- Test markers
- Timeout settings
- Output formatting

## In-Memory Testing Pattern (Recommended)

The `test_inmemory_example.py` file demonstrates FastMCP's recommended testing approach:

```python
import os
import sys
from pathlib import Path

import pytest
from fastmcp.client import Client

# Set env vars BEFORE importing graphiti modules
def set_env_if_empty(key: str, value: str):
    if not os.environ.get(key):
        os.environ[key] = value

set_env_if_empty('SEMAPHORE_LIMIT', '10')
set_env_if_empty('MAX_REFLEXION_ITERATIONS', '0')
set_env_if_empty('FALKORDB_URI', 'redis://localhost:6379')

# Import after env vars are set
from graphiti_mcp_server import mcp

@pytest.fixture
async def mcp_client():
    """In-memory MCP client - no subprocess needed."""
    async with Client(transport=mcp) as client:
        yield client

async def test_list_tools(mcp_client: Client):
    tools = await mcp_client.list_tools()
    assert len(tools) > 0
```

### Benefits of In-Memory Testing

| Aspect | In-Memory | Subprocess |
|--------|-----------|------------|
| Speed | ~1 second | 10+ minutes |
| Reliability | High | Environment issues |
| Debugging | Easy | Difficult |
| Resource Usage | Low | High |

### Available MCP Tools

The Graphiti MCP server exposes these tools:

- `add_memory` - Add episodes to the knowledge graph
- `search_nodes` - Search for entity nodes
- `search_memory_facts` - Search for facts/relationships
- `delete_entity_edge` - Delete an edge
- `delete_episode` - Delete an episode
- `get_entity_edge` - Get edge by UUID
- `get_episodes` - Get recent episodes
- `clear_graph` - Clear all data
- `get_status` - Get server status

## Test Fixtures

### Data Generation

The test suite includes comprehensive data generators:

```python
from test_fixtures import TestDataGenerator

# Generate test data
company = TestDataGenerator.generate_company_profile()
conversation = TestDataGenerator.generate_conversation()
document = TestDataGenerator.generate_technical_document()
```

### Test Client

Simplified client creation:

```python
from test_fixtures import graphiti_test_client

async with graphiti_test_client(database="falkordb") as (session, group_id):
    # Use session for testing
    result = await session.call_tool('add_memory', {...})
```

## Performance Considerations

### LLM Latency Management

The tests account for LLM inference latency through:

1. **Configurable timeouts** - Different timeouts for different operations
2. **Mock LLM option** - Fast testing without API calls
3. **Intelligent polling** - Adaptive waiting for episode processing
4. **Batch operations** - Testing efficiency of batched requests

### Resource Management

- Memory leak detection
- Connection pool monitoring
- Resource usage tracking
- Graceful degradation testing

## CI/CD Integration

### GitHub Actions

```yaml
name: MCP Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      neo4j:
        image: neo4j:5.26
        env:
          NEO4J_AUTH: neo4j/graphiti
        ports:
          - 7687:7687

    steps:
      - uses: actions/checkout@v2

      - name: Install dependencies
        run: |
          pip install uv
          uv sync --extra dev

      - name: Run smoke tests
        run: python tests/run_tests.py smoke --mock-llm

      - name: Run integration tests
        run: python tests/run_tests.py integration --database neo4j
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Troubleshooting

### Common Issues

1. **Database connection failures**
   ```bash
   # Check Neo4j
   curl http://localhost:7474

   # Check FalkorDB
   redis-cli ping
   ```

2. **API key issues**
   ```bash
   # Use mock LLM for testing without API key
   python tests/run_tests.py all --mock-llm
   ```

3. **Timeout errors**
   ```bash
   # Increase timeout for slow systems
   python tests/run_tests.py integration --timeout 600
   ```

4. **Memory issues**
   ```bash
   # Skip stress tests on low-memory systems
   python tests/run_tests.py all --skip-slow
   ```

5. **Environment variable errors** (`ValueError: invalid literal for int()`)
   ```bash
   # This occurs when SEMAPHORE_LIMIT or MAX_REFLEXION_ITERATIONS is set to empty string
   # Solution 1: Use in-memory tests (recommended)
   uv run pytest tests/test_inmemory_example.py -v

   # Solution 2: Set env vars explicitly
   SEMAPHORE_LIMIT=10 MAX_REFLEXION_ITERATIONS=0 python tests/run_tests.py smoke
   ```

   **Root cause**: The graphiti_core/helpers.py module parses environment variables at import time. If these are set to empty strings (not unset), `int('')` fails.

## Test Reports

### Performance Report

After running performance tests:

```python
from test_fixtures import PerformanceBenchmark

benchmark = PerformanceBenchmark()
# ... run tests ...
print(benchmark.report())
```

### Load Test Report

Stress tests generate detailed reports:

```
LOAD TEST REPORT
================
Test Run 1:
  Total Operations: 100
  Success Rate: 95.0%
  Throughput: 12.5 ops/s
  Latency (avg/p50/p95/p99/max): 0.8/0.7/1.5/2.1/3.2s
```

## Contributing

When adding new tests:

1. Use appropriate pytest markers
2. Include docstrings explaining test purpose
3. Use fixtures for common operations
4. Consider LLM latency in test design
5. Add timeout handling for long operations
6. Include performance metrics where relevant

## License

See main project LICENSE file.