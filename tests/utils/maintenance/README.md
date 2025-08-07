# Field System Testing Guide

This README provides comprehensive guidance for running and understanding the field system integration tests in the Graphiti project. These tests cover the MongoDB-Neo4j hybrid architecture for field and cluster management.

## Test Files Overview

### 1. `test_field_nodes_int.py`

**Purpose**: Integration tests for field and cluster node operations with MongoDB synchronization.

**Key Test Areas**:

- FieldNode creation, update, and MongoDB synchronization
- ClusterNode operations with MongoDB cluster metadata integration
- Error handling when MongoDB services are unavailable
- Field analytics (distribution analysis, similarity detection)
- Field constraint validation and temporal consistency

**MongoDB Integration Points**:

- Cluster existence validation
- Field count tracking and synchronization
- Graceful error handling when MongoDB is unavailable

### 2. `test_field_edges_int.py`

**Purpose**: Integration tests for field edge operations with MongoDB cluster validation.

**Key Test Areas**:

- BelongsToEdge creation with cluster validation
- FieldRelationshipEdge operations with MongoDB checks
- Edge constraint validation (cluster consistency, confidence bounds)
- Temporal ordering validation for relationships
- Error handling for MongoDB validation failures

**MongoDB Integration Points**:

- Cluster existence validation before edge creation
- Graceful degradation when MongoDB clusters don't exist
- Error handling for MongoDB connection failures

### 3. `test_field_nodes_operations_int.py`

**Purpose**: Integration tests for bulk field and cluster node operations.

**Key Test Areas**:

- Bulk field creation and MongoDB synchronization
- Cluster creation with MongoDB metadata integration
- Field analytics on bulk datasets
- MongoDB synchronization helpers (`_sync_field_counts_bulk`, `_sync_clusters_bulk`)
- Constraint validation across bulk operations

**MongoDB Integration Points**:

- Bulk cluster validation and creation
- Efficient field count synchronization
- Error handling in bulk operations

### 4. `test_field_edges_operations_int.py`

**Purpose**: Integration tests for bulk field edge operations and relationship analysis.

**Key Test Areas**:

- Bulk BELONGS_TO edge operations with MongoDB validation
- Bulk field relationship edge operations
- Field relationship pattern analysis
- Bidirectional relationship detection
- Network analysis of field relationships
- Constraint validation across bulk edge operations

**MongoDB Integration Points**:

- Bulk cluster validation before edge operations
- MongoDB error handling in bulk operations
- Cluster consistency validation

## Prerequisites

### Required Services

1. **Neo4j Database**

   - Version: 4.4 or higher
   - Default URI: `bolt://localhost:7687`
   - Default credentials: `neo4j/test`

2. **MongoDB (Optional but Recommended)**

   - Version: 4.4 or higher
   - Used for cluster metadata management
   - Tests include graceful error handling when unavailable

3. **Python Environment**
   - Python 3.8 or higher
   - Virtual environment recommended

### Environment Variables

Set the following environment variables for custom configurations:

```bash
# Neo4j Configuration
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="test"

# MongoDB Configuration (optional)
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DATABASE="graphiti_metadata"
```

### Required Python Packages

Install dependencies using one of the following methods:

#### Using Poetry (Recommended)

```bash
poetry install
```

#### Using UV

```bash
uv sync
```

#### Using pip

```bash
pip install -r requirements.txt
# Or install test dependencies separately
pip install pytest pytest-asyncio neo4j motor
```

## Running the Tests

### Full Test Suite

Run all field system integration tests:

```bash
# Using pytest directly
pytest tests/utils/maintenance/test_field_*_int.py -v

# Using poetry
poetry run pytest tests/utils/maintenance/test_field_*_int.py -v

# Using UV
uv run pytest tests/utils/maintenance/test_field_*_int.py -v
```

### Individual Test Files

Run specific test files:

```bash
# Field nodes integration tests
pytest tests/utils/maintenance/test_field_nodes_int.py -v

# Field edges integration tests
pytest tests/utils/maintenance/test_field_edges_int.py -v

# Field nodes operations (bulk) tests
pytest tests/utils/maintenance/test_field_nodes_operations_int.py -v

# Field edges operations (bulk) tests
pytest tests/utils/maintenance/test_field_edges_operations_int.py -v
```

### Test Categories

Run tests by specific categories using markers:

```bash
# Integration tests only
pytest -m integration tests/utils/maintenance/test_field_*_int.py

# MongoDB-related tests
pytest -k "mongodb" tests/utils/maintenance/test_field_*_int.py

# Bulk operation tests
pytest -k "bulk" tests/utils/maintenance/test_field_*_int.py

# Analytics tests
pytest -k "analytics" tests/utils/maintenance/test_field_*_int.py
```

### Specific Test Classes or Methods

Run specific test classes or methods:

```bash
# Run a specific test class
pytest tests/utils/maintenance/test_field_nodes_int.py::TestFieldNodeIntegration -v

# Run a specific test method
pytest tests/utils/maintenance/test_field_nodes_int.py::TestFieldNodeIntegration::test_field_node_mongodb_sync -v

# Run tests matching a pattern
pytest tests/utils/maintenance/test_field_edges_int.py -k "mongodb_validation" -v
```

## Test Configuration

### pytest.ini Configuration

The tests use the following pytest configuration (defined in `pytest.ini`):

```ini
[tool:pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    integration: marks tests as integration tests
    slow: marks tests as slow running
```

### Neo4j Test Database

Tests automatically handle Neo4j database cleanup, but you can manually reset the test database:

```bash
# Connect to Neo4j and clear all data (BE CAREFUL - THIS DELETES ALL DATA)
docker exec -it neo4j cypher-shell -u neo4j -p test
> MATCH (n) DETACH DELETE n;
```

### MongoDB Test Collections

MongoDB tests use mock services by default. If running with real MongoDB:

```bash
# Connect to MongoDB and drop test collections
mongo graphiti_metadata
> db.clusters.drop()
> db.field_metadata.drop()
```

## Test Structure and Patterns

### Fixture Pattern

All tests follow a consistent fixture pattern:

```python
@pytest.fixture
async def driver():
    """Neo4j driver fixture for integration tests"""
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    yield driver
    await driver.close()

@pytest.fixture
def mock_embedder():
    """Mock embedder client fixture"""
    embedder = MagicMock()
    embedder.create.return_value = [0.1, 0.2, 0.3, 0.4, 0.5] * 200  # 1000-dim
    return embedder
```

### Test Data Fixtures

Sample data fixtures provide realistic test scenarios:

```python
@pytest.fixture
def sample_field_nodes():
    """Sample field nodes for testing"""
    return [
        FieldNode(
            uuid=str(uuid.uuid4()),
            name="timestamp",
            description="Log entry timestamp",
            cluster_id="linux_audit_batelco",
            # ... additional fields
        ),
        # ... more sample nodes
    ]
```

### MongoDB Mocking Pattern

MongoDB operations are mocked for reliable testing:

```python
@patch('graphiti_core.field_nodes.ClusterMetadataService')
async def test_field_node_mongodb_sync(self, mock_cluster_service, driver):
    # Mock cluster service
    mock_service_instance = AsyncMock()
    mock_service_instance.validate_cluster_exists.return_value = True
    mock_cluster_service.return_value = mock_service_instance

    # Test operations...

    # Verify MongoDB integration
    mock_service_instance.validate_cluster_exists.assert_called()
```

## Understanding Test Results

### Test Output Interpretation

- ✅ **PASSED**: Test completed successfully with all assertions met
- ❌ **FAILED**: Test failed due to assertion error or unexpected exception
- ⚠️ **SKIPPED**: Test was skipped (usually due to missing dependencies)
- 🔄 **XFAIL**: Expected failure (test is known to fail under certain conditions)

### Common Test Scenarios

1. **MongoDB Available**: Tests run with full MongoDB integration
2. **MongoDB Unavailable**: Tests run with mocked MongoDB services and graceful error handling
3. **Neo4j Connection Issues**: Tests will fail if Neo4j is not accessible
4. **Missing Dependencies**: Tests will be skipped if required packages are missing

### Performance Expectations

- **Field Node Tests**: ~30-60 seconds for full suite
- **Field Edge Tests**: ~20-40 seconds for full suite
- **Bulk Operations Tests**: ~60-120 seconds due to larger datasets
- **Analytics Tests**: ~30-90 seconds depending on analysis complexity

## Troubleshooting

### Common Issues and Solutions

#### 1. Neo4j Connection Errors

```
neo4j.exceptions.ServiceUnavailable: Could not connect to bolt://localhost:7687
```

**Solution**:

- Ensure Neo4j is running: `docker ps | grep neo4j`
- Check Neo4j logs: `docker logs neo4j`
- Verify connection settings in environment variables

#### 2. Import Resolution Errors

```
Import "pytest" could not be resolved
```

**Solution**:

- This is a common IDE issue and doesn't affect test execution
- Ensure pytest is installed: `pip install pytest`
- Run tests from command line to verify they work

#### 3. Async Test Issues

```
RuntimeError: Event loop is closed
```

**Solution**:

- Ensure `asyncio_mode = auto` is set in `pytest.ini`
- Use `@pytest.mark.asyncio` decorator on async test functions
- Check that all async resources are properly cleaned up

#### 4. MongoDB Mock Issues

```
AttributeError: 'MagicMock' object has no attribute 'validate_cluster_exists'
```

**Solution**:

- Ensure MongoDB mocking is properly configured
- Use `AsyncMock` for async methods
- Verify mock service instance creation

#### 5. Memory Issues with Large Datasets

```
MemoryError: Unable to allocate array
```

**Solution**:

- Reduce test dataset sizes
- Run tests individually instead of full suite
- Increase available memory or use smaller batch sizes

### Debug Mode

Run tests with debug output:

```bash
# Enable debug logging
pytest tests/utils/maintenance/test_field_nodes_int.py -v -s --log-level=DEBUG

# Capture stdout
pytest tests/utils/maintenance/test_field_nodes_int.py -v -s --capture=no

# Drop into debugger on failure
pytest tests/utils/maintenance/test_field_nodes_int.py --pdb
```

### Test Database Cleanup

If tests fail due to leftover data:

```bash
# Clean Neo4j test database
docker exec -it neo4j cypher-shell -u neo4j -p test -c "MATCH (n) DETACH DELETE n;"

# Or restart Neo4j container
docker restart neo4j
```

## Contributing

### Adding New Tests

When adding new tests to the field system:

1. **Follow Naming Conventions**:

   - Integration tests: `test_*_int.py`
   - Unit tests: `test_*.py`
   - Test classes: `Test*`
   - Test methods: `test_*`

2. **Use Consistent Fixtures**:

   - Always use the `driver` fixture for Neo4j access
   - Use `mock_embedder` for embedding operations
   - Create sample data fixtures for reusable test data

3. **Include MongoDB Mocking**:

   - Mock `ClusterMetadataService` for MongoDB operations
   - Test both success and error scenarios
   - Include graceful degradation tests

4. **Add Proper Cleanup**:

   - Ensure all created nodes/edges are deleted after tests
   - Use try/finally blocks or fixtures for cleanup
   - Don't rely on database state between tests

5. **Document Test Purpose**:
   - Include docstrings explaining test objectives
   - Document expected MongoDB integration points
   - Explain any complex test scenarios

### Test Coverage Requirements

- **Minimum Coverage**: 80% for new field system code
- **Critical Paths**: 95% coverage for MongoDB integration
- **Error Handling**: 100% coverage for exception scenarios
- **Edge Cases**: Include boundary condition tests

## Support

For issues with the field system tests:

1. **Check Test Logs**: Review detailed test output for specific error messages
2. **Verify Setup**: Ensure all prerequisites are properly configured
3. **Run Individual Tests**: Isolate failing tests to understand root cause
4. **Check Dependencies**: Verify all required packages are installed
5. **Database State**: Ensure clean database state before running tests

For additional support, consult the main project documentation or open an issue in the project repository.
