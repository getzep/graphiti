# Graphiti MCP Server Integration Tests

This directory contains a comprehensive integration test suite for the Graphiti MCP Server using the official Python MCP SDK.

## Overview

The test suite is designed to thoroughly test all aspects of the Graphiti MCP server with special consideration for LLM inference latency and system performance.

## Test Organization

### Core Test Modules

- **`test_comprehensive_integration.py`** - Main integration test suite covering all MCP tools
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

# Install MCP SDK
uv add mcp
```

## Running Tests

### Quick Start

```bash
# Run smoke tests (quick validation)
python tests/run_tests.py smoke

# Run integration tests with mock LLM
python tests/run_tests.py integration --mock-llm

# Run all tests
python tests/run_tests.py all
```

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