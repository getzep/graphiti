# Database Parameter Fix - November 2025

## Summary

Fixed critical bug in graphiti_core where the `database` parameter was not being passed correctly to the Neo4j Python driver, causing all queries to execute against the default `neo4j` database instead of the configured database.

## Root Cause

In `graphiti_core/driver/neo4j_driver.py`, the `execute_query` method was incorrectly adding `database_` to the query parameters dict instead of passing it as a keyword argument to the Neo4j driver's `execute_query` method.

**Incorrect code (before fix):**
```python
params.setdefault('database_', self._database)  # Wrong - adds to params dict
result = await self.client.execute_query(cypher_query_, parameters_=params, **kwargs)
```

**Correct code (after fix):**
```python
kwargs.setdefault('database_', self._database)  # Correct - adds to kwargs
result = await self.client.execute_query(cypher_query_, parameters_=params, **kwargs)
```

## Impact

- **Before fix:** All Neo4j queries executed against the default `neo4j` database, regardless of the `database` parameter passed to `Neo4jDriver.__init__`
- **After fix:** Queries execute against the configured database (e.g., `graphiti`)

## Neo4j Driver API

According to Neo4j Python driver documentation, `database_` must be a keyword argument to `execute_query()`, not a query parameter:

```python
driver.execute_query(
    "MATCH (n) RETURN n",
    {"name": "Alice"},      # parameters_ - query params
    database_="graphiti"    # database_ - kwarg (NOT in parameters dict)
)
```

## Additional Fix: Index Creation Error Handling

Added graceful error handling in MCP server for Neo4j's known `IF NOT EXISTS` bug where fulltext and relationship indices throw `EquivalentSchemaRuleAlreadyExists` errors instead of being idempotent.

This prevents MCP server crashes when indices already exist.

## Files Modified

1. `graphiti_core/driver/neo4j_driver.py` - Fixed database_ parameter handling
2. `mcp_server/src/graphiti_mcp_server.py` - Added index error handling

## Testing

- ✅ Python syntax validation passed
- ✅ Ruff formatting applied
- ✅ Ruff linting passed with no errors
- Manual testing required:
  - Verify indices created in configured database (not default)
  - Verify data stored in configured database
  - Verify MCP server starts successfully with existing indices

## Version

This fix will be released as v1.0.5
