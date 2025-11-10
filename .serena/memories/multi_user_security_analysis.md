# Multi-User Security Analysis - Group ID Isolation

## Analysis Date: November 9, 2025

## Question: Should LLMs be able to specify group_id in multi-user LibreChat?

**Answer: NO - This creates a security vulnerability**

## Security Issue

**Current Risk:**
- Multiple users → Separate MCP instances → Shared database (Neo4j/FalkorDB)
- If LLM can specify `group_id` parameter, User A can access User B's data
- group_id is just a database filter, not a security boundary

**Example Attack:**
```python
# User A's LLM could run:
search_nodes(query="passwords", group_ids=["user_b_456"])
# This would search User B's graph!
```

## Recommended Solution

**Option 3: Security Flag (RECOMMENDED)**

Add configurable enforcement of session isolation:

```yaml
# config.yaml
graphiti:
  group_id: ${GRAPHITI_GROUP_ID:main}
  enforce_session_isolation: ${ENFORCE_SESSION_ISOLATION:false}
```

For LibreChat multi-user:
```yaml
env:
  GRAPHITI_GROUP_ID: "{{LIBRECHAT_USER_ID}}"
  ENFORCE_SESSION_ISOLATION: "true"  # NEW: Force isolation
```

**Tool Implementation:**
```python
@mcp.tool()
async def search_nodes(
    query: str,
    group_ids: list[str] | None = None,
    ...
):
    if config.graphiti.enforce_session_isolation:
        # Security: Always use session group_id
        effective_group_ids = [config.graphiti.group_id]
        if group_ids and group_ids != [config.graphiti.group_id]:
            logger.warning(
                f"Security: Ignoring group_ids {group_ids}. "
                f"Using session group_id: {config.graphiti.group_id}"
            )
    else:
        # Backward compat: Allow group_id override
        effective_group_ids = group_ids or [config.graphiti.group_id]
```

## Benefits

1. **Secure by default for LibreChat**: Set flag = true
2. **Backward compatible**: Single-user deployments can disable flag
3. **Explicit security**: Logged warnings show attempted breaches
4. **Flexible**: Supports both single-user and multi-user use cases

## Implementation Scope

**7 tools need security enforcement:**
1. add_memory
2. search_nodes (+ search_memory_nodes wrapper)
3. get_entities_by_type
4. search_memory_facts
5. compare_facts_over_time
6. get_episodes
7. clear_graph

**5 tools don't need changes:**
- get_entity_edge (UUID-based, already isolated)
- delete_entity_edge (UUID-based)
- delete_episode (UUID-based)
- get_status (global status, no data access)

## Security Properties After Fix

✅ Users cannot access other users' data  
✅ LLM hallucinations/errors can't breach isolation  
✅ Prompt injection attacks can't steal data  
✅ Configurable for different deployment scenarios  
✅ Logged warnings for security monitoring  

## Related Documentation

- LibreChat Setup: DOCS/Librechat.setup.md
- Verification: .serena/memories/librechat_integration_verification.md
- Implementation: mcp_server/src/graphiti_mcp_server.py
