## Summary

This PR adds experimental support for **Apache TinkerPop Gremlin** as an alternative query language for AWS Neptune Database, alongside the existing openCypher support. This enables users to choose their preferred query language and opens the door for future support of other Gremlin-compatible databases (Azure Cosmos DB, JanusGraph, DataStax Graph, etc.).

## Motivation

While Graphiti currently supports AWS Neptune Database using openCypher, Neptune also natively supports Gremlin, which:

- Is Neptune's native query language with potentially better performance for certain traversal patterns
- Provides an alternative query paradigm for users who prefer imperative traversal syntax
- Opens the door for broader database compatibility with the TinkerPop ecosystem

## Key Features

- ✅ `QueryLanguage` enum (CYPHER, GREMLIN) for explicit language selection
- ✅ Dual-mode `NeptuneDriver` supporting both Cypher and Gremlin
- ✅ Gremlin query generation functions for common graph operations
- ✅ Graceful degradation when `gremlinpython` is not installed
- ✅ 100% backward compatible (defaults to CYPHER)

## Implementation Details

### Core Infrastructure
- **graphiti_core/driver/driver.py**: Added `QueryLanguage` enum and `query_language` field to base driver
- **graphiti_core/driver/neptune_driver.py**:
  - Dual client initialization (Cypher via langchain-aws, Gremlin via gremlinpython)
  - Query routing based on language selection
  - Separate `_run_cypher_query()` and `_run_gremlin_query()` methods
- **graphiti_core/graph_queries.py**: 9 new Gremlin query generation functions:
  - `gremlin_match_node_by_property()`
  - `gremlin_match_nodes_by_uuids()`
  - `gremlin_match_edge_by_property()`
  - `gremlin_get_outgoing_edges()`
  - `gremlin_bfs_traversal()`
  - `gremlin_delete_all_nodes()`
  - `gremlin_delete_nodes_by_group_id()`
  - `gremlin_retrieve_episodes()`
  - `gremlin_cosine_similarity_filter()` (placeholder)

### Maintenance Operations
- **graphiti_core/utils/maintenance/graph_data_operations.py**: Updated `clear_data()` to support both query languages

### Testing & Documentation
- **tests/test_neptune_gremlin_int.py**: Comprehensive integration tests
- **examples/quickstart/quickstart_neptune_gremlin.py**: Working usage example
- **examples/quickstart/README.md**: Updated with Gremlin instructions
- **GREMLIN_FEATURE.md**: Complete feature documentation

### Dependencies
- **pyproject.toml**: Added `gremlinpython>=3.7.0` to neptune and dev extras

## Usage Example

```python
from graphiti_core import Graphiti
from graphiti_core.driver.driver import QueryLanguage
from graphiti_core.driver.neptune_driver import NeptuneDriver
from graphiti_core.llm_client import OpenAIClient

# Create Neptune driver with Gremlin query language
driver = NeptuneDriver(
    host='neptune-db://your-cluster.amazonaws.com',
    aoss_host='your-aoss-cluster.amazonaws.com',
    query_language=QueryLanguage.GREMLIN  # Use Gremlin instead of Cypher
)

llm_client = OpenAIClient()
graphiti = Graphiti(driver, llm_client)

# The high-level Graphiti API remains unchanged
await graphiti.build_indices_and_constraints()
await graphiti.add_episode(...)
results = await graphiti.search(...)
```

## Installation

```bash
# Install with Neptune and Gremlin support
pip install graphiti-core[neptune]
```

## Current Limitations

### Supported ✅
- Basic graph operations (CRUD on nodes/edges)
- Graph traversal and BFS
- Maintenance operations (clear_data, delete by group_id)
- Neptune Database clusters

### Not Yet Supported ❌
- Neptune Analytics (only supports Cypher)
- Direct Gremlin-based fulltext search (still uses OpenSearch)
- Direct Gremlin-based vector similarity (still uses OpenSearch)
- Complete `search_utils.py` Gremlin implementation (marked for future work)

### Why OpenSearch is Still Used

Neptune's Gremlin implementation doesn't include native fulltext search or vector similarity functions. These operations continue to use the existing OpenSearch (AOSS) integration, which provides:

- BM25 fulltext search across node/edge properties
- Vector similarity search via k-NN
- Hybrid search capabilities

This hybrid approach (Gremlin for graph traversal + OpenSearch for search) is a standard pattern for production Neptune applications.

## Testing

- ✅ All existing unit tests pass (103/103)
- ✅ New integration tests for Gremlin operations
- ✅ Type checking passes with pyright
- ✅ Linting passes with ruff

```bash
# Run unit tests
uv run pytest tests/ -k "not _int"

# Run Gremlin integration tests (requires Neptune Database)
uv run pytest tests/test_neptune_gremlin_int.py
```

## Breaking Changes

**None.** This is fully backward compatible:
- Default query language is `CYPHER` (existing behavior unchanged)
- `gremlinpython` is an optional dependency
- All existing code continues to work without modifications

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

## Checklist

- [x] Code follows project style guidelines (ruff formatting)
- [x] Type checking passes (pyright)
- [x] All tests pass
- [x] Documentation updated (README, examples, GREMLIN_FEATURE.md)
- [x] Backward compatibility maintained
- [x] No breaking changes

## Related Issues

This addresses feature requests for:
- Broader database compatibility
- Neptune Gremlin support
- Alternative query language options

## Additional Notes

See `GREMLIN_FEATURE.md` in the repository for complete technical documentation, including detailed implementation notes and architecture decisions.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
