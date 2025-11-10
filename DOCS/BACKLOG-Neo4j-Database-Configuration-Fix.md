# BACKLOG: Neo4j Database Configuration Fix

**Status:** Ready for Implementation
**Priority:** Medium
**Type:** Bug Fix + Architecture Improvement
**Date:** 2025-11-09

## Problem Statement

The MCP server does not pass the `database` parameter when initializing the Graphiti client with Neo4j, causing unexpected database behavior and user confusion.

### Current Behavior

1. **Configuration Issue:**
   - User configures `NEO4J_DATABASE=graphiti` in environment
   - MCP server reads this value into config but **does not pass it** to Graphiti constructor
   - Neo4jDriver defaults to `database='neo4j'` (hardcoded default)

2. **Runtime Behavior:**
   - graphiti-core tries to switch databases when `group_id != driver._database` (line 698-700)
   - Calls `driver.clone(database=group_id)` to create new driver
   - **Neo4jDriver does not implement clone()** - inherits no-op base implementation
   - Database switching silently fails, continues using 'neo4j' database
   - Data saved with `group_id` property in 'neo4j' database (not 'graphiti')

3. **User Experience:**
   - User expects data in 'graphiti' database (configured in env)
   - Neo4j Browser shows 'graphiti' database as empty
   - Data actually exists in 'neo4j' database with proper group_id filtering
   - Queries still work (property-based filtering) but confusing architecture

### Root Causes

1. **Incomplete Implementation in graphiti-core:**
   - Base `GraphDriver.clone()` returns `self` (no-op)
   - `FalkorDriver` implements clone() properly
   - `Neo4jDriver` does not implement clone()
   - Database switching only works for FalkorDB, not Neo4j

2. **Missing Parameter in MCP Server:**
   - `mcp_server/src/graphiti_mcp_server.py:233-240`
   - Neo4j initialization does not pass `database` parameter
   - FalkorDB initialization correctly passes `database` parameter

3. **Architectural Mismatch:**
   - Code comments suggest intent to use `group_id` as database name
   - Neo4j best practices recommend property-based multi-tenancy
   - Neo4j databases are heavyweight (not suitable for per-user isolation)

## Solution: Option 2 (Recommended)

**Architecture:** Single database with property-based multi-tenancy

### Design Principles

1. **ONE database** named via configuration (default: 'graphiti')
2. **MULTIPLE users** each with unique `group_id`
3. **Property-based isolation** using `WHERE n.group_id = 'user_id'`
4. **Neo4j best practices** for multi-tenant SaaS applications

### Why This Approach?

- **Performance:** Neo4j databases are heavyweight; property filtering is efficient
- **Operational:** Simpler backup, monitoring, index management
- **Scalability:** Proven pattern for multi-tenant Neo4j applications
- **Current State:** Already working this way (by accident), just needs cleanup

### Implementation Changes

#### File: `mcp_server/src/graphiti_mcp_server.py`

**Location:** Lines 233-240 (Neo4j initialization)

**Current Code:**
```python
# For Neo4j (default), use the original approach
self.client = Graphiti(
    uri=db_config['uri'],
    user=db_config['user'],
    password=db_config['password'],
    llm_client=llm_client,
    embedder=embedder_client,
    max_coroutines=self.semaphore_limit,
    # ❌ MISSING: database parameter not passed!
)
```

**Fixed Code:**
```python
# For Neo4j (default), use configured database with property-based multi-tenancy
database_name = (
    config.database.providers.neo4j.database
    if config.database.providers.neo4j
    else 'graphiti'
)

self.client = Graphiti(
    uri=db_config['uri'],
    user=db_config['user'],
    password=db_config['password'],
    llm_client=llm_client,
    embedder=embedder_client,
    max_coroutines=self.semaphore_limit,
    database=database_name,  # ✅ Pass configured database name
)
```

**Why this works:**
- Sets `driver._database = database_name` (e.g., 'graphiti')
- Prevents clone attempt at line 698: `if 'lvarming73' != 'graphiti'` → True, attempts clone
- Clone returns same driver (no-op), continues using 'graphiti' database
- **Wait, this still has the problem!** Let me reconsider...

**Actually, we need a different approach:**

The issue is graphiti-core's line 698-700 logic assumes `group_id == database`. For property-based multi-tenancy, we need to bypass this check.

**Better Fix (requires graphiti-core understanding):**

Since Neo4jDriver.clone() is a no-op, the current behavior is:
1. Line 698: `if group_id != driver._database` → True (user_id != 'graphiti')
2. Line 700: `driver.clone(database=group_id)` → Returns same driver
3. Data saved with `group_id` property in current database

**This actually works!** The problem is just initialization. Let's fix it properly:

```python
# For Neo4j (default), use configured database with property-based multi-tenancy
# Pass database parameter to ensure correct initial database selection
neo4j_database = (
    config.database.providers.neo4j.database
    if config.database.providers.neo4j
    else 'neo4j'
)

self.client = Graphiti(
    uri=db_config['uri'],
    user=db_config['user'],
    password=db_config['password'],
    llm_client=llm_client,
    embedder=embedder_client,
    max_coroutines=self.semaphore_limit,
    database=neo4j_database,  # ✅ Use configured database (from NEO4J_DATABASE env var)
)
```

**Note:** This ensures the driver starts with the correct database. The clone() call will be a no-op, but data will be in the right database from the start.

#### File: `mcp_server/src/services/factories.py`

**Location:** Lines 393-399

**Current Code:**
```python
return {
    'uri': uri,
    'user': username,
    'password': password,
    # Note: database and use_parallel_runtime would need to be passed
    # to the driver after initialization if supported
}
```

**Fixed Code:**
```python
return {
    'uri': uri,
    'user': username,
    'password': password,
    'database': neo4j_config.database,  # ✅ Include database in config
}
```

This ensures the database parameter is available in the config dictionary.

### Testing Plan

1. **Unit Test:** Verify database parameter is passed correctly
2. **Integration Test:** Verify data saved to configured database
3. **Multi-User Test:** Create episodes with different group_ids, verify isolation
4. **Query Test:** Verify hybrid search respects group_id filtering

## Cleanup Steps

### Prerequisites

- Backup current Neo4j data before any operations
- Note current data location: `neo4j` database with `group_id='lvarming73'`

### Step 1: Verify Current Data Location

```cypher
// In Neo4j Browser
:use neo4j

// Count nodes by group_id
MATCH (n)
WHERE n.group_id IS NOT NULL
RETURN n.group_id, count(*) as node_count

// Verify data exists
MATCH (n:Entity {group_id: 'lvarming73'})
RETURN count(n) as entity_count
```

### Step 2: Implement Code Fix

1. Update `mcp_server/src/services/factories.py` (add database to config)
2. Update `mcp_server/src/graphiti_mcp_server.py` (pass database parameter)
3. Test with unit tests

### Step 3: Create Target Database

```cypher
// In Neo4j Browser or Neo4j Desktop
CREATE DATABASE graphiti
```

### Step 4: Migrate Data (Option A - Manual Copy)

```cypher
// Switch to source database
:use neo4j

// Export data to temporary storage (if needed)
MATCH (n) WHERE n.group_id IS NOT NULL
WITH collect(n) as nodes
// Copy to graphiti database using APOC or manual approach
```

**Note:** This requires APOC procedures or manual export/import. See Option B for easier approach.

### Step 4: Migrate Data (Option B - Restart Fresh)

**Recommended if data is test/development data:**

1. Stop MCP server
2. Delete 'graphiti' database if exists: `DROP DATABASE graphiti IF EXISTS`
3. Create fresh 'graphiti' database: `CREATE DATABASE graphiti`
4. Deploy code fix
5. Restart MCP server (will use 'graphiti' database)
6. Let users re-add their data naturally

### Step 5: Configuration Update

Verify environment configuration in LibreChat:

```yaml
# In LibreChat MCP configuration
env:
  NEO4J_DATABASE: "graphiti"  # ✅ Already configured
  GRAPHITI_GROUP_ID: "lvarming73"  # User's group ID
  # ... other vars
```

### Step 6: Verify Fix

```cypher
// In Neo4j Browser
:use graphiti

// Verify data is in correct database
MATCH (n:Entity {group_id: 'lvarming73'})
RETURN count(n) as entity_count

// Check relationships
MATCH (n:Entity)-[r]->(m:Entity)
WHERE n.group_id = 'lvarming73'
RETURN count(r) as relationship_count
```

### Step 7: Cleanup Old Database (Optional)

**Only after confirming everything works:**

```cypher
// Delete data from old location
:use neo4j
MATCH (n) WHERE n.group_id = 'lvarming73'
DETACH DELETE n
```

## Expected Outcomes

### After Implementation

1. **Correct Database Usage:**
   - MCP server uses database from `NEO4J_DATABASE` env var
   - Default: 'graphiti' (or 'neo4j' if not configured)
   - Data appears in expected location

2. **Multi-Tenant Architecture:**
   - Single database shared across users
   - Each user has unique `group_id`
   - Property-based isolation via Cypher queries
   - Follows Neo4j best practices

3. **Operational Clarity:**
   - Neo4j Browser shows data in expected database
   - Configuration matches runtime behavior
   - Easier to monitor and backup

4. **Code Consistency:**
   - Neo4j initialization matches FalkorDB pattern
   - Database parameter explicitly passed
   - Clear architectural intent

## References

### Code Locations

- **Bug Location:** `mcp_server/src/graphiti_mcp_server.py:233-240`
- **Factory Fix:** `mcp_server/src/services/factories.py:393-399`
- **Neo4j Driver:** `graphiti_core/driver/neo4j_driver.py:34-47`
- **Database Switching:** `graphiti_core/graphiti.py:698-700`
- **Property Storage:** `graphiti_core/nodes.py:491`
- **Query Pattern:** `graphiti_core/nodes.py:566-568`

### Related Issues

- SEMAPHORE_LIMIT configuration (resolved - commit ba938c9)
- Rate limiting with OpenAI Tier 1 (resolved via SEMAPHORE_LIMIT=3)
- Database visibility confusion (this issue)

### Neo4j Multi-Tenancy Resources

- [Neo4j Multi-Tenancy Guide](https://neo4j.com/developer/multi-tenancy-worked-example/)
- [Property-based isolation](https://neo4j.com/docs/operations-manual/current/database-administration/multi-tenancy/)
- FalkorDB uses Redis databases (lightweight, per-user databases make sense)
- Neo4j databases are heavyweight (property-based filtering recommended)

## Implementation Checklist

- [ ] Update `factories.py` to include database in config dict
- [ ] Update `graphiti_mcp_server.py` to pass database parameter
- [ ] Add unit test verifying database parameter is passed
- [ ] Create 'graphiti' database in Neo4j
- [ ] Migrate or recreate data in correct database
- [ ] Verify queries work with correct database
- [ ] Update documentation/README with correct architecture
- [ ] Remove temporary test data from 'neo4j' database
- [ ] Commit changes with descriptive message
- [ ] Update Serena memory with architectural decisions

## Notes

- The graphiti-core library's database switching logic (lines 698-700) is partially implemented
- FalkorDriver has full clone() implementation (multi-database isolation)
- Neo4jDriver inherits no-op clone() (property-based isolation by default)
- This "accidental" architecture is actually the correct Neo4j pattern
- Fix makes the implicit behavior explicit and configurable
