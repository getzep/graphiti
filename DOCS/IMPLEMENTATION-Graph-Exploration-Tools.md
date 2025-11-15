# Implementation Plan: Graph Exploration Tools for General PKM

**Status:** Ready for Implementation
**Priority:** High
**Estimated Effort:** 2-3 hours
**Created:** 2025-11-15
**Branch:** `claude/read-documentation-01J1sX9uuuitPsuoqWR6H9pA`

## Executive Summary

Add two new MCP tools and significantly enhance workflow instructions to enable effective Personal Knowledge Management (PKM) across all use cases. This implementation addresses the core issue of disconnected/duplicate entities by providing the LLM with complete graph visibility before adding new information.

## Use Cases (All Equally Important)

This implementation serves **general-purpose PKM**, including:

- 🏗️ **Architecture & Technical Decisions** - Database choices, API design patterns, technology evaluations
- 💼 **Work Projects** - Project constraints, team decisions, deployment considerations
- 🧠 **AI Coaching Sessions** - Personal insights, behavioral patterns, goal tracking
- 📚 **Learning & Research** - Technical concepts, article summaries, connected ideas
- 🔧 **Problem Solving** - Troubleshooting notes, solution patterns, lessons learned
- 💭 **General Knowledge** - Any information worth remembering and connecting

**Key Insight:** All use cases benefit from complete graph exploration and temporal tracking, whether tracking architectural decisions over time or personal coaching insights.

## Problem Analysis

### Current State
- Existing tools: `search_nodes`, `search_memory_facts`, `add_memory`
- Issue: LLM creates disconnected/duplicate entities instead of forming proper relations
- Root cause: Workflow problem, not architecture limitation

### Why Semantic Search Isn't Sufficient

**Critical Limitations of `search_memory_facts` for exploration:**

1. **Requires query formulation** - Must guess what queries to run
2. **No completeness guarantee** - Returns "relevant" results, may miss connections
3. **Expensive for systematic exploration** - Multiple speculative semantic searches
4. **Poor temporal tracking** - Doesn't return chronological episode history

**Why graph traversal is needed:**

1. **Pattern recognition** - Requires COMPLETE connection data
2. **Proactive exploration** - Direct traversal vs guessing semantic queries
3. **Cost efficiency** - Single graph query vs multiple semantic searches
4. **Temporal analysis** - Chronological episode history for evolution tracking
5. **Reliability** - Guaranteed complete results vs probabilistic semantic matching

## Solution: Two New Tools + Enhanced Instructions

### Tool 1: `get_entity_connections`

**Purpose:** Direct graph traversal to show ALL relationships connected to an entity

**Use Cases:**
- "Show me everything connected to PostgreSQL decision"
- "What constraints are linked to the deployment pipeline?"
- "What's connected to my stress mentions?" (coaching)
- "All considerations related to authentication approach"

**Leverages:** `EntityEdge.get_by_node_uuid()` from `graphiti_core/edges.py:456`

### Tool 2: `get_entity_timeline`

**Purpose:** Chronological episode history showing when/how an entity was discussed

**Use Cases:**
- "When did we first discuss microservices architecture?"
- "Show project status updates over time"
- "How has my perspective on work-life balance evolved?" (coaching)
- "Timeline of GraphQL research and decisions"

**Leverages:** `EpisodicNode.get_by_entity_node_uuid()` from `graphiti_core/nodes.py:415`

## Implementation Details

### File to Modify

**`mcp_server/src/graphiti_mcp_server.py`**

✅ Allowed per CLAUDE.md (mcp_server modifications permitted)
❌ No changes to `graphiti_core/` (uses existing features only)

### Changes Required

#### 1. Import Verification (Line ~18)

Verify `EpisodicNode` is imported:
```python
from graphiti_core.nodes import EpisodeType, EpisodicNode
```
(Already present - no change needed)

#### 2. Enhanced MCP Instructions (Lines 117-145)

**Replace entire `GRAPHITI_MCP_INSTRUCTIONS` with:**

```python
GRAPHITI_MCP_INSTRUCTIONS = """
Graphiti is a memory service for AI agents built on a knowledge graph. It transforms information
into a richly connected knowledge network of entities and relationships.

The system organizes data as:
- **Episodes**: Content snippets (conversations, notes, documents)
- **Nodes**: Entities (people, projects, concepts, decisions, anything)
- **Facts**: Relationships between entities with temporal metadata

## Core Workflow: SEARCH FIRST, THEN ADD

**Always explore existing knowledge before adding new information.**

### WHEN ADDING INFORMATION:

```
User provides information
    ↓
1. search_nodes(extract key entities/concepts)
    ↓
2. IF entities found:
   → get_entity_connections(entity_uuid) - See what's already linked
   → Optional: get_entity_timeline(entity_uuid) - Understand history
    ↓
3. add_memory(episode_body with explicit references to found entities)
    ↓
Result: New episode automatically connects to existing knowledge
```

### WHEN RETRIEVING INFORMATION:

```
User asks question
    ↓
1. search_nodes(extract keywords from question)
    ↓
2. For relevant entities:
   → get_entity_connections(uuid) - Explore neighborhood
   → get_entity_timeline(uuid) - See evolution/history
    ↓
3. Optional: search_memory_facts(semantic query) - Find specific relationships
    ↓
Result: Comprehensive answer from complete graph context
```

## Tool Selection Guide

**Finding entities:**
- `search_nodes` - Semantic search for entities by keywords/description

**Exploring connections:**
- `get_entity_connections` - **ALL** relationships for an entity (complete, direct graph traversal)
- `search_memory_facts` - Semantic search for relationships (query-driven, may miss some)

**Understanding history:**
- `get_entity_timeline` - **ALL** episodes mentioning an entity (chronological, complete)

**Adding information:**
- `add_memory` - Store new episodes (AFTER searching existing knowledge)

**Retrieval vs Graph Traversal:**
- Use `get_entity_connections` when you need COMPLETE data (pattern detection, exploration)
- Use `search_memory_facts` when you have a specific semantic query

## Examples

### Example 1: Adding Technical Decision

```python
# ❌ BAD: Creates disconnected node
add_memory(
    name="Database choice",
    episode_body="Chose PostgreSQL for new service"
)

# ✅ GOOD: Connects to existing knowledge
nodes = search_nodes(query="database architecture microservices")
# Found: MySQL (main db), Redis (cache), microservices pattern

connections = get_entity_connections(entity_uuid=nodes[0]['uuid'])
# Sees: MySQL connects to user-service, payment-service

add_memory(
    name="Database choice",
    episode_body="Chose PostgreSQL for new notification-service. Different from
                  existing MySQL used by user-service and payment-service because
                  we need better JSON support for notification templates."
)
# Result: Rich connections created automatically
```

### Example 2: Exploring Project Context

```python
# User asks: "What database considerations do we have?"

nodes = search_nodes(query="database")
# Returns: PostgreSQL, MySQL, Redis entities

for node in nodes:
    connections = get_entity_connections(entity_uuid=node['uuid'])
    # Shows ALL services, constraints, decisions connected to each database

    timeline = get_entity_timeline(entity_uuid=node['uuid'])
    # Shows when each database was discussed, decisions made over time

# Synthesize comprehensive answer from COMPLETE data
```

### Example 3: Pattern Recognition (Coaching Context)

```python
# User: "I'm feeling stressed today"

nodes = search_nodes(query="stress")
connections = get_entity_connections(entity_uuid=nodes[0]['uuid'])
# Discovers: stress ↔ work, sleep, project-deadline, coffee-intake

timeline = get_entity_timeline(entity_uuid=nodes[0]['uuid'])
# Shows: First mentioned 3 months ago, frequency increasing

# Can now make informed observations:
# "I notice stress is connected to work deadlines, sleep patterns, and
#  increased coffee. Mentions have tripled since the new project started."

add_memory(
    name="Stress discussion",
    episode_body="Discussed stress today. User recognizes connection to
                  project deadlines and sleep quality from our past conversations."
)
```

## Key Principles

1. **Always search before adding** - Even for trivial data
2. **Reference found entities by name** - Creates automatic connections
3. **Use complete data for patterns** - Graph traversal, not semantic guessing
4. **Track evolution over time** - Timeline shows how understanding changed

## Technical Notes

- All tools respect `group_id` filtering for namespace isolation
- Temporal metadata tracks both creation time and domain time
- Facts can be invalidated (superseded) without deletion
- Processing is asynchronous - episodes queued for background processing
"""
```

#### 3. Add Tool: `get_entity_connections` (After existing tools, ~line 850)

```python
@mcp.tool(
    annotations={
        'title': 'Get Entity Connections',
        'readOnlyHint': True,
        'destructiveHint': False,
        'idempotentHint': True,
        'openWorldHint': True,
    },
)
async def get_entity_connections(
    entity_uuid: str,
    group_ids: list[str] | None = None,
    max_connections: int = 50,
) -> FactSearchResponse | ErrorResponse:
    """Get ALL relationships connected to a specific entity. **Complete graph traversal.**

    **Use this to explore what's already known about an entity before adding new information.**

    Unlike search_memory_facts which requires a semantic query, this performs direct graph
    traversal to return EVERY relationship where the entity is involved. Guarantees completeness.

    WHEN TO USE THIS TOOL:
    - Exploring entity neighborhood → get_entity_connections **USE THIS**
    - Before adding related info → get_entity_connections **USE THIS**
    - Pattern detection (need complete data) → get_entity_connections **USE THIS**
    - Understanding full context without formulating queries

    WHEN NOT to use:
    - Specific semantic query → use search_memory_facts instead
    - Finding entities → use search_nodes instead

    Use Cases:
    - "Show everything connected to PostgreSQL decision"
    - "What's linked to the authentication service?"
    - "All relationships for Django project"
    - Before adding: "Let me see what's already known about X"

    Args:
        entity_uuid: UUID of entity to explore (from search_nodes or previous results)
        group_ids: Optional list of namespaces to filter connections
        max_connections: Maximum relationships to return (default: 50)

    Returns:
        FactSearchResponse with all connected relationships and temporal metadata

    Examples:
        # After finding entity
        nodes = search_nodes(query="Django project")
        django_uuid = nodes[0]['uuid']

        # Get all connections
        connections = get_entity_connections(entity_uuid=django_uuid)

        # Limited results for specific namespace
        connections = get_entity_connections(
            entity_uuid=django_uuid,
            group_ids=["work"],
            max_connections=20
        )
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Use existing EntityEdge.get_by_node_uuid() method
        edges = await EntityEdge.get_by_node_uuid(client.driver, entity_uuid)

        # Filter by group_ids if provided
        if group_ids:
            edges = [e for e in edges if e.group_id in group_ids]

        # Limit results
        edges = edges[:max_connections]

        if not edges:
            return FactSearchResponse(
                message=f'No connections found for entity {entity_uuid}',
                facts=[]
            )

        # Format using existing formatter
        facts = [format_fact_result(edge) for edge in edges]

        return FactSearchResponse(
            message=f'Found {len(facts)} connection(s) for entity',
            facts=facts
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting entity connections: {error_msg}')
        return ErrorResponse(error=f'Error getting entity connections: {error_msg}')
```

#### 4. Add Tool: `get_entity_timeline` (After get_entity_connections)

```python
@mcp.tool(
    annotations={
        'title': 'Get Entity Timeline',
        'readOnlyHint': True,
        'destructiveHint': False,
        'idempotentHint': True,
        'openWorldHint': True,
    },
)
async def get_entity_timeline(
    entity_uuid: str,
    group_ids: list[str] | None = None,
    max_episodes: int = 20,
) -> EpisodeSearchResponse | ErrorResponse:
    """Get chronological episode history for an entity. **Shows evolution over time.**

    **Use this to understand how an entity was discussed across conversations.**

    Returns ALL episodes where this entity was mentioned, in chronological order.
    Shows the full conversational history and temporal evolution of concepts, decisions,
    or any tracked entity.

    WHEN TO USE THIS TOOL:
    - Understanding entity history → get_entity_timeline **USE THIS**
    - "When did we discuss X?" → get_entity_timeline **USE THIS**
    - Tracking evolution over time → get_entity_timeline **USE THIS**
    - Seeing context from original sources

    WHEN NOT to use:
    - Semantic search across all episodes → use search_episodes instead
    - Finding entities → use search_nodes instead

    Use Cases:
    - "When did we first discuss microservices architecture?"
    - "Show all mentions of the deployment pipeline"
    - "Timeline of stress mentions" (coaching context)
    - "How did our understanding of GraphQL evolve?"

    Args:
        entity_uuid: UUID of entity (from search_nodes or previous results)
        group_ids: Optional list of namespaces to filter episodes
        max_episodes: Maximum episodes to return (default: 20, chronological)

    Returns:
        EpisodeSearchResponse with episodes ordered chronologically

    Examples:
        # After finding entity
        nodes = search_nodes(query="microservices")
        arch_uuid = nodes[0]['uuid']

        # Get conversation history
        timeline = get_entity_timeline(entity_uuid=arch_uuid)

        # Recent episodes only for specific namespace
        timeline = get_entity_timeline(
            entity_uuid=arch_uuid,
            group_ids=["architecture"],
            max_episodes=10
        )
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Use existing EpisodicNode.get_by_entity_node_uuid() method
        episodes = await EpisodicNode.get_by_entity_node_uuid(
            client.driver,
            entity_uuid
        )

        # Filter by group_ids if provided
        if group_ids:
            episodes = [e for e in episodes if e.group_id in group_ids]

        # Sort by valid_at (chronological order)
        episodes.sort(key=lambda e: e.valid_at)

        # Limit results
        episodes = episodes[:max_episodes]

        if not episodes:
            return EpisodeSearchResponse(
                message=f'No episodes found mentioning entity {entity_uuid}',
                episodes=[]
            )

        # Format episodes
        episode_results = []
        for ep in episodes:
            episode_results.append({
                'uuid': ep.uuid,
                'name': ep.name,
                'content': ep.content,
                'valid_at': ep.valid_at.isoformat() if ep.valid_at else None,
                'created_at': ep.created_at.isoformat() if ep.created_at else None,
                'source': ep.source.value if ep.source else None,
                'group_id': ep.group_id,
            })

        return EpisodeSearchResponse(
            message=f'Found {len(episode_results)} episode(s) mentioning entity',
            episodes=episode_results
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting entity timeline: {error_msg}')
        return ErrorResponse(error=f'Error getting entity timeline: {error_msg}')
```

#### 5. Update `add_memory` Docstring (Line ~385)

**Add after line 387 (`"""Add information to memory...`):**

```python
    """Add information to memory. **This is the PRIMARY method for storing information.**

    **⚠️ IMPORTANT: SEARCH FIRST, THEN ADD**

    Before using this tool, always:
    1. Search for related entities: search_nodes(query="relevant keywords")
    2. Explore their connections: get_entity_connections(entity_uuid=found_uuid)
    3. Then add with context: Reference found entities by name in episode_body

    This creates rich automatic connections instead of isolated nodes.

    **PRIORITY: Use this AFTER exploring existing knowledge.**
```

### Testing Plan

#### Manual Testing Scenarios

**Test 1: Connection Exploration**
```python
# Setup
add_memory(name="Tech stack", episode_body="Using Django with PostgreSQL")
nodes = search_nodes(query="Django")
django_uuid = nodes[0]['uuid']

# Test
connections = get_entity_connections(entity_uuid=django_uuid)

# Verify
assert len(connections['facts']) > 0
assert any('PostgreSQL' in str(fact) for fact in connections['facts'])
```

**Test 2: Timeline Tracking**
```python
# Setup: Add multiple episodes over time
add_memory(name="Week 1", episode_body="Started Django project evaluation")
add_memory(name="Week 2", episode_body="Django project approved, beginning development")
add_memory(name="Week 3", episode_body="Django project hit first milestone")

nodes = search_nodes(query="Django project")
project_uuid = nodes[0]['uuid']

# Test
timeline = get_entity_timeline(entity_uuid=project_uuid)

# Verify
assert len(timeline['episodes']) >= 3
assert timeline['episodes'][0]['name'] == "Week 1"  # chronological
```

**Test 3: Search-First Workflow**
```python
# Step 1: Search
nodes = search_nodes(query="database architecture")
# Found: PostgreSQL, MySQL entities

# Step 2: Explore
postgres_uuid = nodes[0]['uuid']
connections = get_entity_connections(entity_uuid=postgres_uuid)
timeline = get_entity_timeline(entity_uuid=postgres_uuid)

# Step 3: Add with context
add_memory(
    name="New service database",
    episode_body="New notification service will use PostgreSQL like our
                  existing user service, chosen for JSON support"
)

# Verify: New episode should connect to existing PostgreSQL entity
new_connections = get_entity_connections(entity_uuid=postgres_uuid)
assert len(new_connections['facts']) > len(connections['facts'])
```

**Test 4: Cross-Domain Usage**

```python
# Architecture decision
add_memory(name="API Design", episode_body="Using REST for public API, GraphQL internal")

# Coaching note
add_memory(name="Stress check-in", episode_body="Feeling stressed about API deadlines")

# Research note
add_memory(name="GraphQL Learning", episode_body="Studied GraphQL federation patterns")

# Test: All should work with same tools
for query in ["API", "stress", "GraphQL"]:
    nodes = search_nodes(query=query)
    if nodes:
        connections = get_entity_connections(entity_uuid=nodes[0]['uuid'])
        timeline = get_entity_timeline(entity_uuid=nodes[0]['uuid'])
        assert connections is not None
        assert timeline is not None
```

#### Integration Testing

- Test with LibreChat MCP integration
- Verify tool discovery in MCP inspector
- Test error handling (invalid UUIDs, missing entities)
- Verify group_id filtering works correctly
- Test with various entity types

#### Error Handling Tests

```python
# Invalid UUID
result = get_entity_connections(entity_uuid="invalid-uuid")
assert 'error' in result

# Empty results
result = get_entity_connections(entity_uuid="non-existent-uuid")
assert result['message'].contains('No connections')

# Group filtering
result = get_entity_connections(
    entity_uuid=valid_uuid,
    group_ids=["non-existent-group"]
)
assert len(result['facts']) == 0
```

## Deployment Strategy

### Phase 1: Implementation (Current Branch)
1. Make code changes on `claude/read-documentation-01J1sX9uuuitPsuoqWR6H9pA`
2. Manual testing with local MCP server
3. Verify with MCP inspector

### Phase 2: Docker Build
1. Update version in `mcp_server/pyproject.toml`
2. Trigger custom Docker build workflow
3. Test in containerized environment

### Phase 3: Documentation
1. Update `mcp_server/README.md` with new tools
2. Add workflow examples
3. Document best practices

## Success Metrics

### Qualitative
- ✅ LLM can explore complete entity context before adding data
- ✅ Users can ask "what do I know about X?" and get comprehensive view
- ✅ Reduced duplicate/disconnected entities
- ✅ Temporal tracking enables evolution analysis
- ✅ Works equally well for technical decisions, projects, and personal notes

### Quantitative
- Implementation: ~150 lines of code (tools + enhanced instructions)
- No new dependencies required
- 100% reuses existing Graphiti features
- Implementation time: 2-3 hours
- Backward compatible: adds tools, doesn't modify existing

## Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| Code Implementation | 45-60 min | Add two tools, update instructions, update add_memory docstring |
| Manual Testing | 30-45 min | Run test scenarios, verify functionality |
| Integration Testing | 30-45 min | Test with MCP server, error handling, edge cases |
| Documentation | 15-30 min | Code comments, commit messages |
| **Total** | **2-3 hours** | |

## Rejected Alternatives

### Alternative 1: Enhanced Instructions Only
**Why Rejected:**
- `search_memory_facts` requires semantic query formulation (guessing)
- No completeness guarantee for pattern detection
- No chronological episode history for temporal tracking
- More expensive (multiple semantic searches vs single graph query)

### Alternative 2: Build Episode Reprocessing Tool
**Why Rejected:**
- Episodes are immutable - reprocessing yields identical results
- Expensive (re-runs LLM on all episodes)
- Doesn't solve root problem (LLM needs visibility, not reprocessing)

### Alternative 3: Community Clustering Tool
**Why Rejected:**
- No clear trigger for when to call it
- Expensive with unclear value
- Current search doesn't use communities
- Better as manual admin operation

## References

### Code References
- `graphiti_core/edges.py:456` - `EntityEdge.get_by_node_uuid()`
- `graphiti_core/nodes.py:415` - `EpisodicNode.get_by_entity_node_uuid()`
- `mcp_server/src/graphiti_mcp_server.py` - MCP tool definitions
- `mcp_server/src/utils/formatting.py` - Result formatters

### Related Documents
- `CLAUDE.md` - Project modification guidelines
- `DOCS/MCP-Tool-Descriptions-Final-Revision.md` - MCP tool patterns
- Investigation documents (from other branch) - PKM use case analysis

## Risk Assessment

**LOW RISK:**
- ✅ No changes to `graphiti_core/` (respects CLAUDE.md)
- ✅ Only exposes existing, tested functionality
- ✅ Backward compatible (adds tools, doesn't modify)
- ✅ Standard FastMCP patterns and annotations
- ✅ Error handling follows existing patterns

**POTENTIAL ISSUES:**
- Large result sets (mitigated by `max_connections`/`max_episodes` limits)
- Invalid UUIDs (handled with try/except and error responses)
- Missing entities (returns empty results with clear message)

## Post-Implementation

### Monitoring
- Watch for usage patterns in logs
- Monitor performance (graph queries should be fast)
- Track user feedback

### Future Enhancements (Optional)
- Path finding between two entities
- Subgraph extraction (entity + N-hop neighborhood)
- Temporal filters (episodes/connections within time range)
- Bidirectional entity influence tracking

## Notes

- General-purpose design works for ALL PKM use cases
- Coaching analysis proved need for complete data, but tools stay domain-agnostic
- Instructions use mixed examples (technical, project, personal) to emphasize versatility
- Tools leverage existing Graphiti features - zero core changes needed

---

**Ready for implementation upon approval.**
