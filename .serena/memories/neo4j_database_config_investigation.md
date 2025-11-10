# Neo4j Database Configuration Investigation Results

**Date:** 2025-11-10
**Status:** Investigation Complete - Problem Confirmed

## Executive Summary

The problem described in BACKLOG-Neo4j-Database-Configuration-Fix.md is **confirmed and partially understood**. However, the actual implementation challenge is **more complex than described** because:

1. The Graphiti constructor does NOT accept a `database` parameter
2. The database parameter must be passed directly to Neo4jDriver
3. The MCP server needs to create a Neo4jDriver instance and pass it to Graphiti

---

## Investigation Findings

### 1. Neo4j Initialization (MCP Server)

**File:** `mcp_server/src/graphiti_mcp_server.py`
**Lines:** 233-240

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
)
```

**Problem:** Database parameter is NOT passed. This results in Neo4jDriver using hardcoded default `database='neo4j'`.

**Comparison with FalkorDB (lines 220-223):**
```python
falkor_driver = FalkorDriver(
    host=db_config['host'],
    port=db_config['port'],
    password=db_config['password'],
    database=db_config['database'],  # ✅ Database IS passed!
)

self.client = Graphiti(
    graph_driver=falkor_driver,
    llm_client=llm_client,
    embedder=embedder_client,
    max_coroutines=self.semaphore_limit,
)
```

**Key Difference:** FalkorDB creates the driver separately and passes it to Graphiti. This is the correct pattern!

---

### 2. Database Config in Factories

**File:** `mcp_server/src/services/factories.py`
**Lines:** 393-399 (Neo4j), 428-434 (FalkorDB)

**Neo4j Config (Current):**
```python
return {
    'uri': uri,
    'user': username,
    'password': password,
    # Note: database and use_parallel_runtime would need to be passed
    # to the driver after initialization if supported
}
```

**FalkorDB Config (Working):**
```python
return {
    'driver': 'falkordb',
    'host': host,
    'port': port,
    'password': password,
    'database': falkor_config.database,  # ✅ Included!
}
```

**Finding:** FalkorDB correctly includes database in config, Neo4j does not.

---

### 3. Graphiti Constructor Analysis

**File:** `graphiti_core/graphiti.py`
**Lines:** 128-142 (constructor signature)
**Lines:** 198-203 (Neo4jDriver initialization)

**Constructor Signature:**
```python
def __init__(
    self,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    llm_client: LLMClient | None = None,
    embedder: EmbedderClient | None = None,
    cross_encoder: CrossEncoderClient | None = None,
    store_raw_episode_content: bool = True,
    graph_driver: GraphDriver | None = None,
    max_coroutines: int | None = None,
    tracer: Tracer | None = None,
    trace_span_prefix: str = 'graphiti',
):
```

**CRITICAL FINDING:** The Graphiti constructor does NOT have a `database` parameter!

**Driver Initialization (line 203):**
```python
self.driver = Neo4jDriver(uri, user, password)
```

**Issue:** Neo4jDriver is created without the database parameter, so it uses the hardcoded default:
- `Neo4jDriver.__init__(uri, user, password, database='neo4j')`
- The database defaults to 'neo4j'

---

### 4. Neo4jDriver Implementation

**File:** `graphiti_core/driver/neo4j_driver.py`
**Lines:** 35-47 (constructor)

**Constructor:**
```python
def __init__(
    self,
    uri: str,
    user: str | None,
    password: str | None,
    database: str = 'neo4j',
):
    super().__init__()
    self.client = AsyncGraphDatabase.driver(
        uri=uri,
        auth=(user or '', password or ''),
    )
    self._database = database
```

**Finding:** Neo4jDriver accepts and stores the database parameter correctly. Default is `'neo4j'`.

---

### 5. Clone Method Implementation

**File:** `graphiti_core/driver/driver.py`
**Lines:** 113-115 (base class - no-op)

**Base Class (GraphDriver):**
```python
def clone(self, database: str) -> 'GraphDriver':
    """Clone the driver with a different database or graph name."""
    return self
```

**FalkorDriver Implementation (falkordb_driver.py, lines 251-264):**
```python
def clone(self, database: str) -> 'GraphDriver':
    """
    Returns a shallow copy of this driver with a different default database.
    Reuses the same connection (e.g. FalkorDB, Neo4j).
    """
    if database == self._database:
        cloned = self
    elif database == self.default_group_id:
        cloned = FalkorDriver(falkor_db=self.client)
    else:
        # Create a new instance of FalkorDriver with the same connection but a different database
        cloned = FalkorDriver(falkor_db=self.client, database=database)

    return cloned
```

**Neo4jDriver Implementation:** Does NOT override clone() - inherits no-op base implementation.

**Finding:** Neo4jDriver.clone() returns `self` (no-op), so database switching fails silently.

---

### 6. Database Switching Logic in Graphiti

**File:** `graphiti_core/graphiti.py`
**Lines:** 698-700 (in add_episode method)

**Current Code:**
```python
if group_id != self.driver._database:
    # if group_id is provided, use it as the database name
    self.driver = self.driver.clone(database=group_id)
    self.clients.driver = self.driver
```

**Behavior:**
- Compares `group_id` (e.g., 'lvarming73') with `self.driver._database` (e.g., 'neo4j')
- If different, calls `clone(database=group_id)`
- For Neo4jDriver, clone() returns `self` unchanged
- Database stays as 'neo4j', not switched to 'lvarming73'

---

## Root Cause Analysis

| Issue | Root Cause | Severity |
|-------|-----------|----------|
| MCP server doesn't pass database to Neo4jDriver | Graphiti constructor doesn't support database parameter | HIGH |
| Neo4jDriver uses hardcoded 'neo4j' default | No database parameter passed during initialization | HIGH |
| Database switching fails silently | Neo4jDriver doesn't implement clone() method | HIGH |
| Config doesn't include database | Factories.py Neo4j case doesn't extract database | MEDIUM |

---

## Implementation Challenge

The backlog document suggests:
```python
self.client = Graphiti(
    uri=db_config['uri'],
    user=db_config['user'],
    password=db_config['password'],
    database=database_name,  # ❌ This parameter doesn't exist!
)
```

**BUT:** The Graphiti constructor does NOT have a `database` parameter!

**Correct Implementation (FalkorDB Pattern):**
```python
# Must create the driver separately with database parameter
neo4j_driver = Neo4jDriver(
    uri=db_config['uri'],
    user=db_config['user'],
    password=db_config['password'],
    database=db_config['database'],  # ✅ Pass to driver constructor
)

# Then pass driver to Graphiti
self.client = Graphiti(
    graph_driver=neo4j_driver,  # ✅ Pass pre-configured driver
    llm_client=llm_client,
    embedder=embedder_client,
    max_coroutines=self.semaphore_limit,
)
```

---

## Configuration Flow

### Current (Broken) Flow:
```
Neo4j env var (NEO4J_DATABASE)
    ↓
factories.py - returns {uri, user, password} ❌ database missing
    ↓
graphiti_mcp_server.py - Graphiti(uri, user, password)
    ↓
Graphiti.__init__ - Neo4jDriver(uri, user, password)
    ↓
Neo4jDriver - database='neo4j' (hardcoded default)
```

### Correct Flow (Should Be):
```
Neo4j env var (NEO4J_DATABASE)
    ↓
factories.py - returns {uri, user, password, database}
    ↓
graphiti_mcp_server.py - Neo4jDriver(uri, user, password, database)
    ↓
graphiti_mcp_server.py - Graphiti(graph_driver=neo4j_driver)
    ↓
Graphiti - uses driver with correct database
```

---

## Verification of Default Database

**Neo4jDriver default (line 40):** `database: str = 'neo4j'`

When initialized without database parameter:
```python
Neo4jDriver(uri, user, password)  # ← database defaults to 'neo4j'
```

This is stored in:
- `self._database = database` (line 47)
- Used in all queries via `params.setdefault('database_', self._database)` (line 69)

---

## Implementation Requirements

To fix this issue:

1. **Update factories.py (lines 393-399):**
   - Add `'database': neo4j_config.database` to returned config dict
   - Extract database from config object like FalkorDB does

2. **Update graphiti_mcp_server.py (lines 216-240):**
   - Create Neo4jDriver instance separately with database parameter
   - Pass driver to Graphiti via `graph_driver` parameter
   - Match FalkorDB pattern

3. **Optional: Add clone() to Neo4jDriver:**
   - Currently inherits no-op base implementation
   - Could be left as-is if using property-based multi-tenancy
   - Or implement proper database switching if needed

---

## Notes

- The backlog document's suggested fix won't work as-is because Graphiti constructor doesn't support database parameter
- The correct pattern is already demonstrated by FalkorDB implementation
- The solution requires restructuring Neo4j initialization to create driver separately
- FalkorDB already implements this correctly and can serve as a template
