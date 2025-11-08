# MCP Server Tools Documentation

## Overview
The Graphiti MCP Server exposes Graphiti functionality through the Model Context Protocol (MCP) for AI assistants (like those in LibreChat). Each tool is decorated with `@mcp.tool()` and provides a specific capability.

## Tool Naming Convention
All tools follow MCP best practices:
- **snake_case naming**: All lowercase with underscores
- **Action-oriented**: Start with verbs (add, search, get, compare, delete)
- **Concise descriptions**: First line describes core action
- **Clear parameters**: Descriptions specify format and provide examples

Reference: https://modelcontextprotocol.io/specification/2025-06-18/server/tools

## Recent Changes

### 2025-11-08 - UUID Parameter Documentation Enhanced
**Problem**: LLMs were attempting to generate and provide UUIDs when adding NEW memories, which should never happen - UUIDs must be auto-generated for new episodes.

**Solution**: Enhanced the `uuid` parameter documentation in `add_memory` to be very explicit: "NEVER provide a UUID for new episodes - UUIDs are auto-generated. This parameter can ONLY be used for updating an existing episode by providing its existing UUID."

**Impact**: Clear guidance for LLMs to prevent them from trying to generate UUIDs for new memories while preserving the ability to update existing episodes.

## Tool List

### Core Memory Management
1. **add_memory** - Add episodes to the knowledge graph ✨ IMPROVED DOCS
2. **clear_graph** - Clear all data for specified group IDs
3. **get_status** - Get server and database connection status

### Search and Retrieval Tools
4. **search_nodes** - Search for nodes/entities using semantic search
5. **search_memory_facts** - Search for facts/relationships using semantic search
6. **get_entities_by_type** ⭐ NEW - Retrieve entities by their type classification
7. **compare_facts_over_time** ⭐ NEW - Compare facts between two time periods

### Entity and Episode Management
8. **get_entity_edge** - Retrieve a specific entity edge by UUID
9. **delete_entity_edge** - Delete an entity edge from the graph
10. **get_episodes** - Retrieve episodes from the graph
11. **delete_episode** - Delete an episode from the graph

## Tool Details

### add_memory (Updated Documentation)
**Purpose**: Add episodes to the knowledge graph

**MCP-Compliant Description**: "Add an episode to memory. This is the primary way to add information to the graph."

**Parameters**:
- `name`: str - Name of the episode
- `episode_body`: str - Content to persist (JSON string for source='json')
- `group_id`: Optional[str] - Group ID for this graph (uses default if not provided)
- `source`: str = 'text' - Source type ('text', 'json', or 'message')
- `source_description`: str = '' - Optional description of the source
- `uuid`: Optional[str] = None - **NEVER provide for NEW episodes**. Can ONLY be used to update an existing episode by providing its UUID.

**UUID Parameter Behavior**:
- **For NEW episodes**: Do NOT provide - auto-generated
- **For UPDATING episodes**: Provide the existing episode's UUID to replace/update it
- **Other uses**: Idempotent operations or external system integration (advanced)

**Implementation Notes**:
- Returns immediately, processes in background
- Episodes for same group_id processed sequentially
- Providing a UUID updates the episode with that UUID if it exists

### get_entities_by_type
**Added**: 2025-11-08
**Purpose**: Essential for PKM (Personal Knowledge Management) - enables browsing entities by their type classification

**MCP-Compliant Description**: "Retrieve entities by their type classification."

**Parameters**:
- `entity_types`: List[str] - Entity types to retrieve (e.g., ["Pattern", "Insight", "Preference"])
- `group_ids`: Optional[List[str]] - Filter by group IDs
- `max_entities`: int = 20 - Maximum entities to return
- `query`: Optional[str] - Optional search query to filter entities

**Implementation Notes**:
- Uses `SearchFilters(node_labels=entity_types)` from graphiti_core
- Uses `NODE_HYBRID_SEARCH_RRF` search config
- When query is provided: semantic search with type filter
- When query is empty: uses space (' ') as generic query to retrieve all of the type
- Returns `NodeSearchResponse` (same format as search_nodes)

**Use Cases**:
- "Show me all my Preferences"
- "List Patterns I've identified"
- "Get Insights about productivity"
- "Find all documented Procedures"

**Example**:
```python
# Get all preferences
get_entities_by_type(entity_types=["Preference"])

# Get patterns and insights about productivity
get_entities_by_type(
    entity_types=["Pattern", "Insight"],
    query="productivity"
)
```

### compare_facts_over_time
**Added**: 2025-11-08
**Purpose**: Track how knowledge/understanding evolved over time - critical for seeing how Patterns, Insights, and understanding changed

**MCP-Compliant Description**: "Compare facts between two time periods."

**Parameters**:
- `query`: str - Search query for facts to compare
- `start_time`: str - ISO 8601 timestamp (e.g., "2024-01-01" or "2024-01-01T10:30:00Z")
- `end_time`: str - ISO 8601 timestamp
- `group_ids`: Optional[List[str]] - Filter by group IDs
- `max_facts_per_period`: int = 10 - Max facts per time category

**Returns**: Dictionary with:
- `facts_from_start`: Facts valid at start_time
- `facts_at_end`: Facts valid at end_time
- `facts_invalidated`: Facts that were invalidated between start and end
- `facts_added`: Facts that became valid between start and end
- `summary`: Count statistics

**Implementation Notes**:
- Uses `DateFilter` and `ComparisonOperator` from graphiti_core.search.search_filters
- Uses `EDGE_HYBRID_SEARCH_RRF` search config
- Makes 4 separate searches with temporal filters:
  1. Facts valid at start (valid_at <= start AND (invalid_at > start OR invalid_at IS NULL))
  2. Facts valid at end (valid_at <= end AND (invalid_at > end OR invalid_at IS NULL))
  3. Facts invalidated (invalid_at > start AND invalid_at <= end)
  4. Facts added (created_at > start AND created_at <= end)
- Uses `format_fact_result()` helper for consistent formatting

**Use Cases**:
- "How did my understanding of sleep patterns change this month?"
- "What productivity insights were replaced?"
- "Show me how my procedures evolved"
- "Track changes in my preferences over time"

**Example**:
```python
compare_facts_over_time(
    query="productivity patterns",
    start_time="2024-01-01",
    end_time="2024-03-01"
)
```

## Implementation Constraints

### Safe Design Principles
All tools follow strict constraints to maintain upstream compatibility:
1. **Only use public Graphiti APIs** - No custom Cypher queries, no internal methods
2. **MCP server only changes** - No modifications to graphiti_core/
3. **Existing patterns** - Follow same structure as existing tools
4. **Standard imports** - Only use imports already in the file or from stable public APIs
5. **MCP compliance** - Follow MCP specification for tool naming and descriptions
6. **LLM-friendly documentation** - Clear guidance to prevent LLM confusion (e.g., UUID usage)

### Dependencies
All required imports are either:
- Already present in the file (SearchFilters, format_fact_result)
- From stable public APIs (DateFilter, ComparisonOperator, search configs)

No new dependencies added to pyproject.toml.

## Testing Notes

### Validation Tests Passed
- ✅ Python syntax check (py_compile)
- ✅ Ruff formatting (auto-formatted)
- ✅ Ruff linting (all checks passed)
- ✅ No custom Cypher or internal APIs used
- ✅ Follows project code style conventions
- ✅ MCP specification compliance verified
- ✅ UUID documentation enhanced to prevent LLM misuse

### Manual Testing Required
Before production use, test:
1. add_memory without LLM trying to provide UUIDs for NEW episodes
2. add_memory with UUID for UPDATING existing episodes
3. get_entities_by_type with various entity type combinations
4. get_entities_by_type with and without query parameter
5. compare_facts_over_time with various date ranges
6. Error handling for invalid inputs (empty types, bad dates, etc.)

## File Location
`mcp_server/src/graphiti_mcp_server.py`

- `add_memory`: Updated documentation (lines 320-403)
- `get_entities_by_type`: Inserted after `search_nodes` function (lines 486-583)
- `compare_facts_over_time`: Inserted after `search_memory_facts` function (lines 585-766)
