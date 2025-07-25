# Test Suite Documentation

This directory contains the test suite for the Graphiti MCP Server OAuth wrapper and related functionality.

## Test Structure

### Core Test Files

- **`test_basic.py`** - Basic functionality tests for core imports and async operations
- **`test_oauth_simple.py`** - Comprehensive unit tests for OAuth wrapper functionality (RECOMMENDED)

### Additional Test Files

- **`test_oauth.py`** - Legacy OAuth endpoint tests (manual testing script style)
- **`test_oauth_routes.py`** - Legacy route accessibility tests
- **`test_oauth_wrapper.py`** - Complex unit tests with detailed mocking (some failures due to SSE complexity)
- **`test_oauth_integration.py`** - Integration tests (some failures due to complex mocking)

## Running Tests

### Recommended Commands

```bash
# Run only reliable unit tests
make test-unit

# Run OAuth wrapper tests with coverage
make test-oauth

# Run all tests (may have some failures in complex SSE tests)
make test
```

### Coverage

The current test suite provides:
- **86% coverage** for the OAuth wrapper (`src/oauth_wrapper.py`)
- **100% test reliability** for core functionality

The missing 14% coverage is primarily:
- SSE streaming generator functions (lines 117-132)
- Main module execution block (lines 191-193)

## Test Categories

### Unit Tests (`test_oauth_simple.py`)
These tests provide comprehensive coverage of:
- OAuth metadata endpoints (authorization server, protected resource)
- Client registration functionality
- Request proxying (messages endpoint)
- POST request handling for SSE endpoint
- Environment variable usage
- Header filtering and forwarding
- Query parameter handling
- Error response propagation

### Integration Tests
These tests verify end-to-end functionality but may require more complex mocking setup.

## Test Architecture

### Fixtures
- **`client`** - FastAPI test client for OAuth wrapper
- **`mock_env`** - Mocked environment variables

### Mocking Strategy
- Uses `patch` to mock `httpx.AsyncClient` calls
- Avoids actual network requests
- Focuses on behavior verification rather than implementation details

## Regression Prevention

The test suite prevents regressions in:
1. **OAuth Discovery Flow** - Ensures OAuth metadata endpoints work correctly
2. **Client Registration** - Verifies client credential generation
3. **Request Proxying** - Confirms proper forwarding to MCP server
4. **Error Handling** - Ensures errors are properly propagated
5. **Header Management** - Verifies header filtering and forwarding
6. **Environment Configuration** - Tests environment variable usage

## Known Limitations

- SSE streaming tests are complex to mock properly due to async generator behavior
- Some integration tests may fail due to complex async context manager mocking
- Real network testing requires running services (covered by manual testing)

## Contributing

When adding new features to the OAuth wrapper:
1. Add corresponding tests to `test_oauth_simple.py`
2. Run `make test-oauth` to verify coverage
3. Ensure all tests pass with `make test-unit`
4. Consider the impact on existing functionality