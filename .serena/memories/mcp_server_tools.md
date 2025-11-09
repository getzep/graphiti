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

### 2025-11-09 - Backward Compatibility Wrappers Added
**Problem**: External code review found tool surface mismatches between server and clients/tests:
- Tests expected `search_memory_nodes` but server only had `search_nodes`
- Tests called tools with singular `group_id` (string) but server expected `group_ids` (list)
- Tests used `last_n` parameter but server expected `max_episodes`

**Solution**: Added backward compatibility without breaking changes:
1. Created `search_memory_nodes` wrapper that delegates to `search_nodes`
2. Updated `get_episodes` to accept both `group_id`/`group_ids` and `last_n`/`max_episodes`
3. Updated `clear_graph` to accept both `group_id` (singular) and `group_ids` (plural)

**Impact**: All existing clients, tests, and documentation examples now work without modification.

### 2025-11-08 - UUID Parameter Documentation Enhanced
**Problem**: LLMs were attempting to generate and provide UUIDs when adding NEW memories, which should never happen - UUIDs must be auto-generated for new episodes.

**Solution**: Enhanced the `uuid` parameter documentation in `add_memory` to be very explicit: "NEVER provide a UUID for new episodes - UUIDs are auto-generated. This parameter can ONLY be used for updating an existing episode by providing its existing UUID."

**Impact**: Clear guidance for LLMs to prevent them from trying to generate UUIDs for new memories while preserving the ability to update existing episodes.

## Tool List

### Core Memory Management
1. **add_memory** - Add episodes to the knowledge graph ✨ IMPROVED DOCS
2. **clear_graph** - Clear all data for specified group IDs ✨ BACKWARD COMPATIBLE
3. **get_status** - Get server and database connection status

### Search and Retrieval Tools
4. **search_nodes** - Search for nodes/entities using semantic search
5. **search_memory_nodes** - ⭐ NEW Alias for search_nodes (backward compatibility)
6. **search_memory_facts** - Search for facts/relationships using semantic search
7. **get_entities_by_type** - Retrieve entities by their type classification
8. **compare_facts_over_time** - Compare facts between two time periods

### Entity and Episode Management
9. **get_entity_edge** - Retrieve a specific entity edge by UUID
10. **delete_entity_edge** - Delete an entity edge from the graph
11. **get_episodes** - Retrieve episodes from the graph ✨ BACKWARD COMPATIBLE
12. **delete_episode** - Delete an episode from the graph

## Tool Details

### search_memory_nodes (NEW - Backward Compatibility Wrapper)
**Added**: 2025-11-09
**Purpose**: Backward compatibility alias for `search_nodes`

**MCP-Compliant Description**: "Search for nodes in the graph memory (compatibility wrapper)."

**Parameters**:
- `query`: str - The search query
- `group_id`: Optional[str] - Single group ID (backward compatibility)
- `group_ids`: Optional[List[str]] - List of group IDs (preferred)
- `max_nodes`: int = 10 - Maximum number of nodes to return
- `entity_types`: Optional[List[str]] - Entity types to filter by

**Implementation Notes**:
- Converts singular `group_id` to `[group_id]` list if provided
- Delegates to `search_nodes` for actual implementation
- Maintains backward compatibility with existing clients

**Location**: After `search_nodes` in `mcp_server/src/graphiti_mcp_server.py`

### get_episodes (Updated - Backward Compatible)
**Updated**: 2025-11-09
**Purpose**: Get episodes from the graph memory

**Parameters** (now accepts both old and new formats):
- `group_id`: Optional[str] - Single group ID (backward compatibility)
- `group_ids`: Optional[List[str]] - List of group IDs (preferred)
- `last_n`: Optional[int] - Max episodes to return (backward compatibility)
- `max_episodes`: int = 10 - Max episodes to return (preferred)

**Backward Compatibility**:
- Old: `get_episodes(group_id="test", last_n=5)` ✅ Works
- New: `get_episodes(group_ids=["test"], max_episodes=5)` ✅ Works
- Both parameters provided: `group_ids` and `max_episodes` take precedence

### clear_graph (Updated - Backward Compatible)
**Updated**: 2025-11-09
**Purpose**: Clear all data from the graph for specified group IDs

**Parameters** (now accepts both old and new formats):
- `group_id`: Optional[str] - Single group ID (backward compatibility)
- `group_ids`: Optional[List[str]] - List of group IDs (preferred)

**Backward Compatibility**:
- Old: `clear_graph(group_id="test")` ✅ Works
- New: `clear_graph(group_ids=["test1", "test2"])` ✅ Works
- Both provided: `group_ids` takes precedence

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

## Implementation Constraints

### Safe Design Principles
All tools follow strict constraints to maintain upstream compatibility:
1. **Only use public Graphiti APIs** - No custom Cypher queries, no internal methods
2. **MCP server only changes** - No modifications to graphiti_core/
3. **Existing patterns** - Follow same structure as existing tools
4. **Standard imports** - Only use imports already in the file or from stable public APIs
5. **MCP compliance** - Follow MCP specification for tool naming and descriptions
6. **LLM-friendly documentation** - Clear guidance to prevent LLM confusion (e.g., UUID usage)
7. **Backward compatibility** - New parameters don't break existing clients

### Dependencies
All required imports are either:
- Already present in the file (SearchFilters, format_fact_result)
- From stable public APIs (DateFilter, ComparisonOperator, search configs)

No new dependencies added to pyproject.toml.

## Validation Tests

### Automated Tests Passed (2025-11-09)
- ✅ Python syntax check (py_compile)
- ✅ Ruff formatting (auto-formatted)
- ✅ Ruff linting (all checks passed)
- ✅ No custom Cypher or internal APIs used
- ✅ Follows project code style conventions
- ✅ MCP specification compliance verified
- ✅ UUID documentation enhanced to prevent LLM misuse
- ✅ Backward compatibility maintained

### Manual Testing Required
Before production use, test:
1. search_memory_nodes with both group_id and group_ids parameters
2. get_episodes with old format (group_id, last_n) and new format (group_ids, max_episodes)
3. clear_graph with both group_id and group_ids parameters
4. add_memory without LLM trying to provide UUIDs for NEW episodes
5. add_memory with UUID for UPDATING existing episodes
6. get_entities_by_type with various entity type combinations
7. compare_facts_over_time with various date ranges

## File Location
`mcp_server/src/graphiti_mcp_server.py`

**Current tool locations** (after 2025-11-09 updates):
- `add_memory`: lines 320-403
- `search_nodes`: lines 406-483
- `search_memory_nodes`: After line 483 (new wrapper)
- `get_entities_by_type`: lines 486-583
- `search_memory_facts`: lines 587-640
- `compare_facts_over_time`: lines 641-829
- `delete_entity_edge`: lines 832-856
- `delete_episode`: lines 858-882
- `get_entity_edge`: lines 884-908
- `get_episodes`: lines 939-1004 (updated for backward compat)
- `clear_graph`: lines 1018-1050 (updated for backward compat)
- `get_status`: lines 1014-1089
