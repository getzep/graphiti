# Gremlin Query Language Support for Neptune Database

## Overview

This PR adds experimental support for the **Gremlin query language** to Graphiti's Neptune Database driver, enabling users to choose between openCypher and Gremlin when working with AWS Neptune Database.

## Motivation

While Graphiti currently supports AWS Neptune Database using openCypher, Neptune also natively supports **Apache TinkerPop Gremlin**, which:

- Is Neptune's native query language with potentially better performance for certain traversal patterns
- Opens the door for future support of other Gremlin-compatible databases (Azure Cosmos DB, JanusGraph, DataStax Graph, etc.)
- Provides an alternative query paradigm for users who prefer imperative traversal syntax

## Implementation Summary

### 1. Core Infrastructure (`graphiti_core/driver/driver.py`)

- Added `QueryLanguage` enum with `CYPHER` and `GREMLIN` options
- Added `query_language` field to `GraphDriver` base class (defaults to `CYPHER` for backward compatibility)

### 2. Query Generation (`graphiti_core/graph_queries.py`)

Added Gremlin query generation functions:

- `gremlin_match_node_by_property()` - Query nodes by label and property
- `gremlin_match_nodes_by_uuids()` - Batch node retrieval
- `gremlin_match_edge_by_property()` - Query edges by label and property
- `gremlin_get_outgoing_edges()` - Traverse relationships
- `gremlin_bfs_traversal()` - Breadth-first graph traversal
- `gremlin_delete_all_nodes()` - Bulk deletion
- `gremlin_delete_nodes_by_group_id()` - Filtered deletion
- `gremlin_retrieve_episodes()` - Time-filtered episode retrieval

### 3. Neptune Driver Updates (`graphiti_core/driver/neptune_driver.py`)

- Added optional `query_language` parameter to `NeptuneDriver.__init__()`
- Conditional import of `gremlinpython` (graceful degradation if not installed)
- Dual client initialization (Cypher via langchain-aws, Gremlin via gremlinpython)
- Query routing based on selected language
- Separate `_run_cypher_query()` and `_run_gremlin_query()` methods
- Gremlin result set conversion to dictionary format for consistency

### 4. Maintenance Operations (`graphiti_core/utils/maintenance/graph_data_operations.py`)

Updated `clear_data()` function to:
- Detect query language and route to appropriate query generation
- Support Gremlin-based node deletion with group_id filtering

### 5. Dependencies (`pyproject.toml`)

- Added `gremlinpython>=3.7.0` to `neptune` and `dev` optional dependencies
- Maintains backward compatibility - Gremlin is optional

## Usage

### Basic Example

```python
from graphiti_core import Graphiti
from graphiti_core.driver.driver import QueryLanguage
from graphiti_core.driver.neptune_driver import NeptuneDriver
from graphiti_core.llm_client import OpenAIClient

# Create Neptune driver with Gremlin query language
driver = NeptuneDriver(
    host='neptune-db://your-cluster.amazonaws.com',
    aoss_host='your-aoss-cluster.amazonaws.com',
    port=8182,
    query_language=QueryLanguage.GREMLIN  # Use Gremlin instead of Cypher
)

llm_client = OpenAIClient()
graphiti = Graphiti(driver, llm_client)

# The high-level Graphiti API remains unchanged
await graphiti.build_indices_and_constraints()
await graphiti.add_episode(...)
results = await graphiti.search(...)
```

### Installation

```bash
# Install with Neptune and Gremlin support
pip install graphiti-core[neptune]

# Or install gremlinpython separately
pip install gremlinpython
```

## Important Limitations

### Supported

✅ Basic graph operations (CRUD on nodes/edges)
✅ Graph traversal and BFS
✅ Maintenance operations (clear_data, delete by group_id)
✅ Neptune Database clusters

### Not Yet Supported

❌ Neptune Analytics (only supports Cypher)
❌ Direct Gremlin-based fulltext search (still uses OpenSearch)
❌ Direct Gremlin-based vector similarity (still uses OpenSearch)
❌ Complete search_utils.py Gremlin implementation (marked as pending)

### Why OpenSearch is Still Used

Neptune's Gremlin implementation doesn't include native fulltext search or vector similarity functions. These operations continue to use the existing OpenSearch (AOSS) integration, which provides:

- BM25 fulltext search across node/edge properties
- Vector similarity search via k-NN
- Hybrid search capabilities

This hybrid approach (Gremlin for graph traversal + OpenSearch for search) is a standard pattern for production Neptune applications.

## Files Changed

### Core Implementation
- `graphiti_core/driver/driver.py` - QueryLanguage enum
- `graphiti_core/driver/neptune_driver.py` - Dual-language support
- `graphiti_core/driver/__init__.py` - Export QueryLanguage
- `graphiti_core/graph_queries.py` - Gremlin query functions
- `graphiti_core/utils/maintenance/graph_data_operations.py` - Gremlin maintenance ops

### Testing & Documentation
- `tests/test_neptune_gremlin_int.py` - Integration tests (NEW)
- `examples/quickstart/quickstart_neptune_gremlin.py` - Example (NEW)
- `examples/quickstart/README.md` - Updated with Gremlin info

### Dependencies
- `pyproject.toml` - Added gremlinpython dependency

## Testing

### Unit Tests

All existing unit tests pass (103/103). The implementation maintains full backward compatibility.

```bash
uv run pytest tests/ -k "not _int"
```

### Integration Tests

New integration test suite `test_neptune_gremlin_int.py` includes:

- Driver initialization with Gremlin
- Basic CRUD operations
- Error handling (e.g., Gremlin + Neptune Analytics = error)
- Dual-mode compatibility (Cypher and Gremlin on same cluster)

**Note:** Integration tests require actual Neptune Database and OpenSearch clusters.

## Backward Compatibility

✅ **100% backward compatible**

- Default query language is `CYPHER` (existing behavior)
- `gremlinpython` is an optional dependency
- Existing code continues to work without any changes
- If Gremlin is requested but not installed, a clear error message guides installation

## Future Work

The following enhancements are planned for future iterations:

1. **Complete search_utils.py Gremlin Support**
   - Implement Gremlin-specific versions of hybrid search functions
   - May require custom Gremlin steps or continued OpenSearch integration

2. **Broader Database Support**
   - Azure Cosmos DB (Gremlin API)
   - JanusGraph
   - DataStax Graph
   - Any Apache TinkerPop 3.x compatible database

3. **Performance Benchmarking**
   - Compare Cypher vs Gremlin performance on Neptune
   - Identify optimal use cases for each language

4. **Enhanced Error Handling**
   - Gremlin-specific error messages and debugging info
   - Query validation before execution

## References

- [AWS Neptune Documentation](https://docs.aws.amazon.com/neptune/)
- [Apache TinkerPop Gremlin](https://tinkerpop.apache.org/gremlin.html)
- [gremlinpython Documentation](https://tinkerpop.apache.org/docs/current/reference/#gremlin-python)

---

**Status:** ✅ Ready for review
**Breaking Changes:** None
**Requires Migration:** No
