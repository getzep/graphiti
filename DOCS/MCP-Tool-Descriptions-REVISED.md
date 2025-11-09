# MCP Tool Descriptions - Production-Ready Revision

**Date:** November 9, 2025
**Status:** Implementation Ready
**MCP Compliance:** Validated against MCP SDK 1.21.0 and 2025-06-18 specification

---

## Executive Summary

This document contains MCP-compliant tool descriptions optimized for LLM tool selection accuracy. All changes are **docstring-only** with no breaking changes to functionality.

**Key Design Principles:**
1. ✅ **LLM-visible priority** - Critical information in docstrings, not hidden in meta
2. ✅ **Decision trees** - Clear guidance for overlapping tools
3. ✅ **Standard annotations** - Using only MCP-specified fields
4. ✅ **Accessibility first** - No emojis, clear language
5. ✅ **Proven patterns** - Standard Python docstring conventions

**What Changed:**
- Priority guidance moved INTO docstrings (LLM-visible)
- Decision trees added for tool disambiguation
- Examples in standard format (after Args section)
- Clearer differentiation between overlapping tools
- No cosmetic changes (emojis, etc.)

---

## MCP Compliance Verification

### Annotations Used (All Standard)
```python
annotations={
    'title': str,              # Display name
    'readOnlyHint': bool,      # Never modifies data
    'destructiveHint': bool,   # May destroy data
    'idempotentHint': bool,    # Safe to retry
    'openWorldHint': bool,     # Accesses external resources
}
```

### Additional Parameters (SDK-Supported)
```python
tags={'category', 'keywords'}  # For client filtering
meta={                          # For client UX (NOT visible to LLM)
    'version': str,
    'category': str,
    'priority': float,          # Client hint only
}
```

**Critical:** Meta fields are NOT visible to LLMs. Priority/importance MUST be in docstring.

---

## Tool Descriptions - Ready for Implementation

### Tool 1: `add_memory` (PRIMARY STORAGE)

```python
@mcp.tool(
    annotations={
        'title': 'Add Memory',
        'readOnlyHint': False,
        'destructiveHint': False,
        'idempotentHint': True,
        'openWorldHint': True,
    },
    tags={'write', 'memory', 'ingestion', 'core'},
    meta={
        'version': '1.0',
        'category': 'core',
        'priority': 0.9,
    },
)
async def add_memory(
    name: str,
    episode_body: str,
    group_id: str | None = None,
    source: str = 'text',
    source_description: str = '',
    uuid: str | None = None,
) -> SuccessResponse | ErrorResponse:
    """Add information to memory. **This is the PRIMARY method for storing information.**

    **PRIORITY: Use this tool FIRST when storing any information.**

    Processes content asynchronously, automatically extracting entities, relationships,
    and deduplicating similar information. Returns immediately while processing continues
    in background.

    WHEN TO USE THIS TOOL:
    - Storing information → add_memory (this tool) **USE THIS FIRST**
    - Searching information → use search_nodes or search_memory_facts
    - Deleting information → use delete_episode or delete_entity_edge

    Use Cases:
    - Recording conversation context, insights, or observations
    - Storing user preferences, requirements, or procedures
    - Capturing information about people, organizations, events, topics
    - Importing structured data (JSON format)
    - Updating existing information (provide uuid parameter)

    Args:
        name: Brief descriptive title for this memory episode
        episode_body: Content to store. For JSON source, must be properly escaped JSON string
        group_id: Optional namespace for organizing memories (uses default if not provided)
        source: Content format - 'text', 'json', or 'message' (default: 'text')
        source_description: Optional context about where this information came from
        uuid: ONLY for updates - provide existing episode UUID. DO NOT provide for new memories

    Returns:
        SuccessResponse confirming episode was queued for processing

    Examples:
        # Store plain text observation
        add_memory(
            name="Customer preference",
            episode_body="Acme Corp prefers email communication over phone calls"
        )

        # Store structured data
        add_memory(
            name="Product catalog",
            episode_body='{"company": "Acme", "products": [{"id": "P001", "name": "Widget"}]}',
            source="json"
        )

        # Update existing episode
        add_memory(
            name="Customer preference",
            episode_body="Acme Corp prefers Slack communication",
            uuid="abc-123-def-456"  # UUID from previous get_episodes or search
        )
    """
```

**Changes:**
- Added "**This is the PRIMARY method**" in first line (LLM-visible)
- Added "**PRIORITY: Use this tool FIRST**" guidance
- Decision tree at top for quick disambiguation
- Examples in standard location after Args
- Removed emoji from title

---

### Tool 2: `search_nodes` (PRIMARY ENTITY SEARCH)

```python
@mcp.tool(
    annotations={
        'title': 'Search Memory Entities',
        'readOnlyHint': True,
        'destructiveHint': False,
        'idempotentHint': True,
        'openWorldHint': True,
    },
    tags={'search', 'entities', 'memory', 'core'},
    meta={
        'version': '1.0',
        'category': 'core',
        'priority': 0.8,
    },
)
async def search_nodes(
    query: str,
    group_ids: list[str] | None = None,
    max_nodes: int = 10,
    entity_types: list[str] | None = None,
) -> NodeSearchResponse | ErrorResponse:
    """Search for entities by name or content. **PRIMARY method for finding entities.**

    **PRIORITY: Use this tool for entity searches (people, organizations, concepts).**

    Searches entity names, summaries, and attributes using hybrid semantic + keyword matching.
    Returns the entities themselves (nodes), not relationships or conversation content.

    WHEN TO USE THIS TOOL:
    - Finding specific entities by name/content → search_nodes (this tool) **USE THIS**
    - Listing ALL entities of a type → use get_entities_by_type
    - Searching conversation content or relationships → use search_memory_facts

    Use Cases:
    - "Find information about Acme Corporation"
    - "Search for entities related to Python programming"
    - "What entities exist about productivity?"
    - Retrieving entities before adding related information

    Args:
        query: Search keywords or semantic description
        group_ids: Optional list of memory namespaces to search within
        max_nodes: Maximum number of results to return (default: 10)
        entity_types: Optional filter by entity types (e.g., ["Organization", "Person"])

    Returns:
        NodeSearchResponse containing matching entities with names, summaries, and metadata

    Examples:
        # Find entities by name
        search_nodes(query="Acme")

        # Semantic search
        search_nodes(query="companies in the technology sector")

        # Filter by entity type
        search_nodes(
            query="productivity",
            entity_types=["Insight", "Pattern"]
        )
    """
```

**Changes:**
- "**PRIMARY method for finding entities**" in first line
- Clear priority guidance for LLMs
- Decision tree distinguishes from search_memory_facts and get_entities_by_type
- Examples show different search patterns
- Clean title without emoji

---

### Tool 3: `search_memory_facts` (PRIMARY CONTENT SEARCH)

```python
@mcp.tool(
    annotations={
        'title': 'Search Memory Facts',
        'readOnlyHint': True,
        'destructiveHint': False,
        'idempotentHint': True,
        'openWorldHint': True,
    },
    tags={'search', 'facts', 'relationships', 'memory', 'core'},
    meta={
        'version': '1.0',
        'category': 'core',
        'priority': 0.85,
    },
)
async def search_memory_facts(
    query: str,
    group_ids: list[str] | None = None,
    max_facts: int = 10,
    center_node_uuid: str | None = None,
) -> FactSearchResponse | ErrorResponse:
    """Search conversation content and relationships. **PRIMARY method for content search.**

    **PRIORITY: Use this tool for searching conversation/episode content and entity relationships.**

    "Facts" in this context means relationships/connections between entities, not factual
    statements. Searches the actual conversation content and how entities are connected.

    WHEN TO USE THIS TOOL:
    - Searching conversation/episode content → search_memory_facts (this tool) **USE THIS**
    - Finding entity relationships → search_memory_facts (this tool) **USE THIS**
    - Finding entities by name → use search_nodes
    - Listing entities by type → use get_entities_by_type

    Use Cases:
    - "What conversations mentioned pricing?"
    - "How is Acme Corp related to our products?"
    - "Find relationships between User and productivity patterns"
    - Searching what was actually said in conversations

    Args:
        query: Search query for conversation content or relationships
        group_ids: Optional list of memory namespaces to search within
        max_facts: Maximum number of results to return (default: 10)
        center_node_uuid: Optional entity UUID to center search around (find relationships)

    Returns:
        FactSearchResponse containing matching relationships with source, target, and context

    Examples:
        # Search conversation content
        search_memory_facts(query="discussions about budget")

        # Find entity relationships
        search_memory_facts(
            query="collaboration",
            center_node_uuid="entity-uuid-123"
        )

        # Broad relationship search
        search_memory_facts(query="how does Acme relate to pricing")
    """
```

**Changes:**
- "**PRIMARY method for content search**" in first line
- Explicit priority guidance
- Clarified "facts = relationships" confusion
- Decision tree clearly separates from search_nodes
- Use cases show conversation search vs entity search

---

### Tool 4: `get_entities_by_type` (BROWSE BY TYPE)

```python
@mcp.tool(
    annotations={
        'title': 'Get Entities by Type',
        'readOnlyHint': True,
        'destructiveHint': False,
        'idempotentHint': True,
        'openWorldHint': True,
    },
    tags={'search', 'entities', 'browse', 'classification'},
    meta={
        'version': '1.0',
        'category': 'discovery',
        'priority': 0.75,
    },
)
async def get_entities_by_type(
    entity_types: list[str],
    group_ids: list[str] | None = None,
    max_entities: int = 20,
    query: str | None = None,
) -> NodeSearchResponse | ErrorResponse:
    """Retrieve ALL entities of specified type(s), optionally filtered by query.

    **Use this to browse/list entities by their classification type.**

    WHEN TO USE THIS TOOL:
    - Listing ALL entities of a type → get_entities_by_type (this tool) **USE THIS**
    - Searching entities by content → use search_nodes
    - Searching relationships/content → use search_memory_facts

    Use Cases:
    - "Show me all Preferences"
    - "List all Insights and Patterns"
    - "Get all Organizations" (optionally filtered by keyword)
    - Browsing knowledge organized by entity classification

    Args:
        entity_types: REQUIRED. Type(s) to retrieve (e.g., ["Insight"], ["Preference", "Requirement"])
        group_ids: Optional list of memory namespaces to search within
        max_entities: Maximum results (default: 20, higher than search tools)
        query: Optional keyword filter within the type(s). Omit to get ALL entities of type

    Returns:
        NodeSearchResponse containing all entities of the specified type(s)

    Examples:
        # Get ALL entities of a type
        get_entities_by_type(entity_types=["Preference"])

        # Get multiple types
        get_entities_by_type(entity_types=["Insight", "Pattern"])

        # Filter within a type
        get_entities_by_type(
            entity_types=["Organization"],
            query="technology"
        )
    """
```

**Changes:**
- Clear "browse/list" positioning
- Decision tree separates from search tools
- Emphasized that query is optional (can get ALL)
- Higher default max_entities (20 vs 10) documented

---

### Tool 5: `compare_facts_over_time` (TEMPORAL ANALYSIS)

```python
@mcp.tool(
    annotations={
        'title': 'Compare Facts Over Time',
        'readOnlyHint': True,
        'destructiveHint': False,
        'idempotentHint': True,
        'openWorldHint': True,
    },
    tags={'search', 'facts', 'temporal', 'analysis', 'evolution'},
    meta={
        'version': '1.0',
        'category': 'analytics',
        'priority': 0.6,
    },
)
async def compare_facts_over_time(
    query: str,
    start_time: str,
    end_time: str,
    group_ids: list[str] | None = None,
    max_facts_per_period: int = 10,
) -> dict[str, Any] | ErrorResponse:
    """Compare relationships/facts between two time periods to track knowledge evolution.

    **Use for temporal analysis - how information changed over time.**

    Returns four categories: facts at start, facts at end, facts invalidated, facts added.
    Useful for understanding how knowledge evolved or changed during a specific time window.

    WHEN TO USE THIS TOOL:
    - Analyzing how information changed over time → compare_facts_over_time (this tool)
    - Current/recent information → use search_memory_facts
    - Single point-in-time search → use search_memory_facts

    Use Cases:
    - "How did our understanding of Acme Corp change this month?"
    - "What information was added/updated between Jan-Feb?"
    - "Track evolution of productivity insights over Q1"

    Args:
        query: Search query for facts to track over time
        start_time: Start timestamp in ISO 8601 format (e.g., "2024-01-01" or "2024-01-01T10:30:00Z")
        end_time: End timestamp in ISO 8601 format
        group_ids: Optional list of memory namespaces to analyze
        max_facts_per_period: Maximum facts per category (default: 10)

    Returns:
        Dictionary with: facts_from_start, facts_at_end, facts_invalidated, facts_added

    Examples:
        # Track changes over a month
        compare_facts_over_time(
            query="customer requirements",
            start_time="2024-01-01",
            end_time="2024-01-31"
        )

        # Analyze knowledge evolution
        compare_facts_over_time(
            query="productivity patterns",
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-03-31T23:59:59Z"
        )
    """
```

---

### Tool 6: `get_entity_edge` (DIRECT UUID LOOKUP)

```python
@mcp.tool(
    annotations={
        'title': 'Get Entity Edge by UUID',
        'readOnlyHint': True,
        'destructiveHint': False,
        'idempotentHint': True,
        'openWorldHint': True,
    },
    tags={'retrieval', 'facts', 'uuid', 'direct-access'},
    meta={
        'version': '1.0',
        'category': 'direct-access',
        'priority': 0.5,
    },
)
async def get_entity_edge(uuid: str) -> dict[str, Any] | ErrorResponse:
    """Retrieve a specific relationship/fact by its UUID. **Direct lookup only.**

    **Use ONLY when you already have the exact UUID from a previous search.**

    WHEN TO USE THIS TOOL:
    - You have a UUID from previous search → get_entity_edge (this tool)
    - Searching for facts → use search_memory_facts
    - Don't have a UUID → use search_memory_facts

    Args:
        uuid: UUID of the relationship to retrieve

    Returns:
        Dictionary with fact details (source entity, target entity, relationship, timestamps)

    Examples:
        # Retrieve specific relationship
        get_entity_edge(uuid="abc-123-def-456")
    """
```

---

### Tool 7: `get_episodes` (RECENT EPISODES)

```python
@mcp.tool(
    annotations={
        'title': 'Get Recent Episodes',
        'readOnlyHint': True,
        'destructiveHint': False,
        'idempotentHint': True,
        'openWorldHint': True,
    },
    tags={'retrieval', 'episodes', 'history', 'changelog'},
    meta={
        'version': '1.0',
        'category': 'direct-access',
        'priority': 0.5,
    },
)
async def get_episodes(
    group_id: str | None = None,
    group_ids: list[str] | None = None,
    last_n: int | None = None,
    max_episodes: int = 10,
) -> EpisodeSearchResponse | ErrorResponse:
    """Retrieve recent episodes by recency. **Like 'git log' for memory.**

    **Use for listing what was added recently, NOT for searching content.**

    Think: "git log" (this tool) vs "git grep" (search_memory_facts)

    WHEN TO USE THIS TOOL:
    - List recent additions to memory → get_episodes (this tool)
    - Audit what was added recently → get_episodes (this tool)
    - Search episode CONTENT → use search_memory_facts
    - Find episodes by keywords → use search_memory_facts

    Use Cases:
    - "What was added to memory recently?"
    - "Show me the last 10 episodes"
    - "List recent memory additions as a changelog"

    Args:
        group_id: Single memory namespace (legacy parameter)
        group_ids: List of memory namespaces (preferred)
        last_n: Maximum episodes (legacy parameter, use max_episodes instead)
        max_episodes: Maximum episodes to return (default: 10)

    Returns:
        EpisodeSearchResponse with episodes sorted by recency (newest first)

    Examples:
        # Get recent episodes
        get_episodes(max_episodes=10)

        # Get recent episodes from specific namespace
        get_episodes(group_ids=["my-project"], max_episodes=20)
    """
```

**Changes:**
- Added "git log vs git grep" analogy
- Clear separation from content search
- Emphasized recency, not content

---

### Tool 8: `delete_entity_edge` (DESTRUCTIVE)

```python
@mcp.tool(
    annotations={
        'title': 'Delete Entity Edge',
        'readOnlyHint': False,
        'destructiveHint': True,
        'idempotentHint': True,
        'openWorldHint': True,
    },
    tags={'delete', 'destructive', 'facts', 'admin'},
    meta={
        'version': '1.0',
        'category': 'maintenance',
        'priority': 0.3,
    },
)
async def delete_entity_edge(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete a specific relationship/fact. **DESTRUCTIVE - Cannot be undone.**

    **WARNING: This operation is permanent and irreversible.**

    WHEN TO USE THIS TOOL:
    - User explicitly confirms deletion → delete_entity_edge (this tool)
    - Removing verified incorrect relationship → delete_entity_edge (this tool)
    - Updating information → use add_memory (preferred)
    - Marking as outdated → system handles automatically

    Safety Requirements:
    - Only use after explicit user confirmation
    - Verify UUID is correct before deleting
    - Cannot be undone - ensure user understands
    - Idempotent (safe to retry if operation fails)

    Args:
        uuid: UUID of the relationship to permanently delete

    Returns:
        SuccessResponse confirming deletion

    Examples:
        # Delete after user confirmation
        delete_entity_edge(uuid="abc-123-def-456")
    """
```

---

### Tool 9: `delete_episode` (DESTRUCTIVE)

```python
@mcp.tool(
    annotations={
        'title': 'Delete Episode',
        'readOnlyHint': False,
        'destructiveHint': True,
        'idempotentHint': True,
        'openWorldHint': True,
    },
    tags={'delete', 'destructive', 'episodes', 'admin'},
    meta={
        'version': '1.0',
        'category': 'maintenance',
        'priority': 0.3,
    },
)
async def delete_episode(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete a specific episode. **DESTRUCTIVE - Cannot be undone.**

    **WARNING: This operation is permanent and irreversible.**

    WHEN TO USE THIS TOOL:
    - User explicitly confirms deletion → delete_episode (this tool)
    - Removing incorrect, outdated, or sensitive information → delete_episode (this tool)
    - Updating episode → use add_memory with uuid parameter (preferred)
    - Clearing all data → use clear_graph

    Safety Requirements:
    - Only use after explicit user confirmation
    - Verify UUID is correct before deleting
    - Cannot be undone - ensure user understands
    - May affect related entities and relationships
    - Idempotent (safe to retry if operation fails)

    Args:
        uuid: UUID of the episode to permanently delete

    Returns:
        SuccessResponse confirming deletion

    Examples:
        # Delete after user confirmation
        delete_episode(uuid="episode-abc-123")
    """
```

---

### Tool 10: `clear_graph` (EXTREMELY DESTRUCTIVE)

```python
@mcp.tool(
    annotations={
        'title': 'Clear Graph - DANGER',
        'readOnlyHint': False,
        'destructiveHint': True,
        'idempotentHint': True,
        'openWorldHint': True,
    },
    tags={'delete', 'destructive', 'admin', 'bulk', 'danger'},
    meta={
        'version': '1.0',
        'category': 'admin',
        'priority': 0.1,
    },
)
async def clear_graph(
    group_id: str | None = None,
    group_ids: list[str] | None = None,
) -> SuccessResponse | ErrorResponse:
    """Delete ALL data for specified memory namespaces. **EXTREMELY DESTRUCTIVE.**

    **DANGER: Destroys ALL episodes, entities, and relationships. NO UNDO POSSIBLE.**

    MANDATORY SAFETY PROTOCOL FOR LLMs:
    1. Confirm user understands ALL DATA will be PERMANENTLY DELETED
    2. Ask user to type the exact group_id to confirm intent
    3. Only proceed after EXPLICIT confirmation with typed group_id
    4. If user shows ANY hesitation, DO NOT proceed

    WHEN TO USE THIS TOOL:
    - ONLY after explicit multi-step confirmation
    - Resetting test/development environments
    - Starting completely fresh after catastrophic errors
    - NEVER use for removing specific items (use delete_episode or delete_entity_edge)

    Critical Warnings:
    - Destroys ALL data for specified namespace(s)
    - NO backup is created automatically
    - NO undo is possible
    - Affects all users sharing the group_id
    - Cannot recover deleted data

    Args:
        group_id: Single namespace to clear (legacy parameter)
        group_ids: List of namespaces to clear (preferred)

    Returns:
        SuccessResponse confirming all data was destroyed

    Examples:
        # ONLY after explicit confirmation protocol
        clear_graph(group_id="test-environment")
    """
```

**Changes:**
- Added "MANDATORY SAFETY PROTOCOL FOR LLMs" section
- Explicit 4-step confirmation process
- Stronger warnings about data loss

---

### Tool 11: `get_status` (HEALTH CHECK)

```python
@mcp.tool(
    annotations={
        'title': 'Get Server Status',
        'readOnlyHint': True,
        'destructiveHint': False,
        'idempotentHint': True,
        'openWorldHint': True,
    },
    tags={'admin', 'health', 'status', 'diagnostics'},
    meta={
        'version': '1.0',
        'category': 'admin',
        'priority': 0.4,
    },
)
async def get_status() -> StatusResponse:
    """Check server health and database connectivity.

    **Use for diagnostics and health checks.**

    WHEN TO USE THIS TOOL:
    - Verifying server is operational → get_status (this tool)
    - Diagnosing connection issues → get_status (this tool)
    - Pre-flight health check → get_status (this tool)
    - Retrieving data → use search tools (not this)

    Returns:
        StatusResponse with status ('ok' or 'error') and connection details

    Examples:
        # Check server health
        get_status()
    """
```

---

### Tool 12: `search_memory_nodes` (LEGACY ALIAS)

```python
@mcp.tool(
    annotations={
        'title': 'Search Memory Nodes',
        'readOnlyHint': True,
        'destructiveHint': False,
        'idempotentHint': True,
        'openWorldHint': True,
    },
    tags={'search', 'entities', 'legacy', 'compatibility'},
    meta={
        'version': '1.0',
        'category': 'compatibility',
        'priority': 0.7,
        'note': 'Legacy alias - prefer search_nodes for new code',
    },
)
async def search_memory_nodes(
    query: str,
    group_id: str | None = None,
    group_ids: list[str] | None = None,
    max_nodes: int = 10,
    entity_types: list[str] | None = None,
) -> NodeSearchResponse | ErrorResponse:
    """Search for entities. **Legacy compatibility alias for search_nodes.**

    **For new code, prefer using search_nodes instead.**

    This tool provides backward compatibility with older clients. It delegates to
    search_nodes with identical functionality.

    Args:
        query: Search query for finding entities
        group_id: Single namespace (legacy parameter)
        group_ids: List of namespaces (preferred)
        max_nodes: Maximum results (default: 10)
        entity_types: Optional type filter

    Returns:
        NodeSearchResponse (delegates to search_nodes)

    Examples:
        # Works identically to search_nodes
        search_memory_nodes(query="Acme")
    """
```

---

## Implementation Instructions

### Step 1: Update Tool Docstrings

Use Serena's symbolic editing tools to update each tool's docstring:

```bash
# For each tool:
mcp__serena__replace_symbol_body(
    name_path="add_memory",
    relative_path="mcp_server/src/graphiti_mcp_server.py",
    body="<new implementation from above>"
)
```

### Step 2: Update Priority Metadata

Update meta dictionaries for two tools:
- `search_memory_facts`: `'priority': 0.85` (was 0.8)
- `get_entities_by_type`: `'priority': 0.75` (was 0.7)

### Step 3: Validation

```bash
cd mcp_server

# Format code
uv run ruff format src/graphiti_mcp_server.py

# Lint
uv run ruff check src/graphiti_mcp_server.py

# Syntax check
python3 -m py_compile src/graphiti_mcp_server.py
```

### Step 4: Testing Checklist

Test with MCP clients (Claude Desktop, Cline, etc.):

- [ ] Decision trees help LLM choose correct tool
- [ ] Priority guidance is visible in tool descriptions
- [ ] Destructive operations trigger appropriate caution
- [ ] Examples are clear and helpful
- [ ] No rendering issues (emojis, formatting)
- [ ] Tool selection accuracy improved

---

## Priority Guidance Summary (LLM-Visible)

### How Priority is Communicated to LLMs

**In docstring first line:**
- "**This is the PRIMARY method**" → Top priority
- "**PRIMARY method for X**" → Category leader
- "**Use for X**" → Standard tool
- "**Direct lookup only**" → Specialized use
- "**DESTRUCTIVE**" → Use with extreme caution

**In WHEN TO USE section:**
- "**USE THIS FIRST**" → Highest priority
- "**USE THIS**" → Preferred for this use case
- "Prefer X over Y" → Comparative guidance

**NOT communicated to LLMs:**
- `meta.priority` values (0.1-0.9) → Client UX only
- Title emojis (removed for accessibility)
- Tags (used for client filtering)

---

## Key Improvements Over Original

1. **LLM-Visible Priority**
   - ✅ Priority in docstring first line (LLM sees this)
   - ✅ Explicit "USE THIS FIRST" guidance
   - ❌ Removed reliance on meta.priority (LLM doesn't see this)

2. **Decision Trees**
   - ✅ Kept and refined for all overlapping tools
   - ✅ Clear "A vs B vs C" guidance

3. **MCP Compliance**
   - ✅ Standard annotations only
   - ✅ No non-standard fields
   - ✅ Proper use of meta for client UX

4. **Accessibility**
   - ✅ No emojis in titles (screen reader friendly)
   - ✅ Clear, professional language
   - ✅ Standard Python docstring format

5. **Examples**
   - ✅ Standard placement (after Args section)
   - ✅ Realistic, varied use cases
   - ✅ Shows different parameter combinations

6. **Conciseness**
   - ✅ Removed unsupported claims (40-60% improvement, etc.)
   - ✅ Focused on actionable guidance
   - ✅ ~50% shorter than original document

---

## What Was Removed/Fixed

### Removed
- ❌ Emojis in title field (⭐ 🔍 ⚠️)
- ❌ Examples in Args section (non-standard)
- ❌ Unsupported quantitative claims (40-60% improvement)
- ❌ Reliance on meta.priority for LLM guidance

### Fixed
- ✅ Priority now in docstring (LLM-visible)
- ✅ Examples in standard location
- ✅ "Facts = relationships" clarified
- ✅ Clear separation of tools

### Kept
- ✅ Decision trees
- ✅ Safety protocols
- ✅ Clear differentiation
- ✅ Standard annotations

---

## Expected Outcomes

### Measurable Goals
- Improved LLM tool selection accuracy (measure after deployment)
- Reduced tool selection errors (track wrong tool usage)
- Faster decision time (fewer tools evaluated)

### Qualitative Goals
- LLMs choose correct tool more consistently
- Users see appropriate warnings for destructive operations
- Clearer documentation for developers
- Better developer experience

**Note:** Specific percentages removed. Measure actual impact after deployment.

---

## Files to Modify

**Primary:**
- `mcp_server/src/graphiti_mcp_server.py` (12 tool docstrings + 2 meta priority values)

**Documentation:**
- `DOCS/MCP-Tool-Descriptions-REVISED.md` (this file)

**Optional:**
- Update `.serena/memories/mcp_tool_descriptions_final_revision.md` with findings

---

## Rollback Plan

```bash
# If issues occur, rollback via git
git checkout HEAD -- mcp_server/src/graphiti_mcp_server.py

# Or restore from this documented state
```

---

## Next Steps

1. ✅ Review this revised document
2. ⬜ Approve for implementation
3. ⬜ Implement using Serena tools
4. ⬜ Validate with ruff + py_compile
5. ⬜ Test with MCP client
6. ⬜ Measure impact on tool selection
7. ⬜ Iterate based on real usage data

---

## FAQ

**Q: Why no emojis?**
A: Accessibility (screen readers), professionalism, and inconsistent rendering across clients. Priority should be communicated through words, not symbols.

**Q: Why is priority in docstrings instead of meta?**
A: Meta fields are not visible to LLMs (per MCP SDK docs). For LLMs to see priority, it must be in the docstring text itself.

**Q: Will this work with all entity types?**
A: Yes. Descriptions are generic with examples showing variety (PKM, business, technical use cases).

**Q: Any breaking changes?**
A: No. All changes are docstring-only. Functionality is identical.

**Q: How to measure success?**
A: Track tool selection accuracy before/after deployment. Monitor wrong tool usage rates. Gather user feedback.

---

**END OF DOCUMENT**

Ready for implementation via Serena symbolic editing tools.
