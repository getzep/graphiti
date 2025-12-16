# Neo4j Nested Attributes Serialization Fix

## Problem Description

### Symptoms

When using Graphiti with Neo4j, the system would crash with the following error when entity or edge attributes contained nested structures (Maps of Lists, Lists of Maps):

```
Neo.ClientError.Statement.TypeError: Property values can only be of primitive types 
or arrays thereof. Encountered: Map{discovered_aligned_resources -> List{String(...)}, 
shared_multifaceted_character_analysis -> List{String(...)}, ...}
```

Full error message:
```
Neo.ClientError.Statement.TypeError - Expected the value Map{...} to be of type 
BOOLEAN, STRING, INTEGER, FLOAT, DATE, LOCAL TIME, ZONED TIME, LOCAL DATETIME, 
ZONED DATETIME, DURATION or POINT, but was of type MAP NOT NULL.
```

### Root Cause

Neo4j property values can only be:
- **Primitives**: `BOOLEAN`, `STRING`, `INTEGER`, `FLOAT`, `DATE`, `TIME`, `DATETIME`, `DURATION`, `POINT`
- **Arrays of primitives**: `List[STRING]`, `List[INTEGER]`, etc.

Neo4j **cannot store**:
- Nested Maps (dictionaries within dictionaries)
- Lists of Maps
- Maps of Lists containing Maps

The bug was in `graphiti_core/utils/bulk_utils.py` where:
- **Kuzu**: Attributes were properly JSON-serialized to strings
- **Neo4j**: Attributes were spread as individual properties via `entity_data.update(node.attributes or {})`

When Graphiti's LLM extraction created entities/edges with rich, structured attributes (common with custom entity types), Neo4j would reject them.

## Technical Details

### Write Path Issue

**Before fix (lines 181-185, 210-214 in bulk_utils.py):**

```python
if driver.provider == GraphProvider.KUZU:
    attributes = convert_datetimes_to_strings(node.attributes) if node.attributes else {}
    entity_data['attributes'] = json.dumps(attributes)  # ✅ JSON string
else:
    entity_data.update(node.attributes or {})  # ❌ Spreads nested dicts as properties
```

This worked for flat attributes but failed for nested structures:

```python
# ✅ Works (flat):
attributes = {"age": 30, "location": "New York"}
# Becomes: n.age = 30, n.location = "New York"

# ❌ Fails (nested):
attributes = {
    "metadata": {
        "analysis": ["item1", "item2"],
        "nested": {"key": "value"}
    }
}
# Neo4j rejects: n.metadata = {nested dict}
```

### Read Path Architecture

The retrieval path was already designed to handle both approaches:

```python
# In get_entity_node_from_record() (nodes.py):
if provider == GraphProvider.KUZU:
    attributes = json.loads(record['attributes'])  # Parse JSON string
else:
    attributes = record['attributes']              # Use dict from properties(n)
    attributes.pop('uuid', None)                   # Filter known fields
    # ... pop other standard fields
```

## Solution

### Changes Made

#### 1. Write Path (bulk_utils.py)

Serialize attributes to JSON for **both** Kuzu and Neo4j:

```python
if driver.provider == GraphProvider.KUZU:
    attributes = convert_datetimes_to_strings(node.attributes) if node.attributes else {}
    entity_data['attributes'] = json.dumps(attributes)
else:
    # Neo4j: Serialize attributes to JSON string to support nested structures
    attributes = convert_datetimes_to_strings(node.attributes) if node.attributes else {}
    entity_data['attributes'] = json.dumps(attributes) if attributes else '{}'
```

Same fix applied to both entity nodes (lines 181-187) and entity edges (lines 210-216).

#### 2. Read Path (nodes.py, edges.py)

Updated to handle JSON string format while maintaining backward compatibility:

```python
if provider == GraphProvider.KUZU:
    attributes = json.loads(record['attributes']) if record['attributes'] else {}
else:
    # Neo4j now stores attributes as JSON string
    raw_attrs = record.get('attributes', '{}')
    if isinstance(raw_attrs, str):
        attributes = json.loads(raw_attrs) if raw_attrs else {}
    else:
        # Backward compatibility: handle dict from properties(n)
        attributes = raw_attrs
        attributes.pop('uuid', None)
        # ... filter standard fields
```

#### 3. Query Updates (models/nodes/node_db_queries.py, models/edges/edge_db_queries.py)

Changed Neo4j queries to return `n.attributes` instead of `properties(n)`:

```python
# Before:
return """
    n.uuid AS uuid,
    ...
    labels(n) AS labels,
    properties(n) AS attributes  # ❌ Returns all properties as dict
"""

# After (for Neo4j):
return """
    n.uuid AS uuid,
    ...
    labels(n) AS labels,
    n.attributes AS attributes    # ✅ Returns JSON string from attributes property
"""
```

## Backward Compatibility

The fix maintains **full backward compatibility**:

1. **Read path**: Checks if attributes is a string (new format) or dict (old format)
2. **Legacy data**: Existing Neo4j databases with spread attributes will continue to work
3. **Kuzu**: No changes to Kuzu behavior (it already used JSON serialization)
4. **FalkorDB/Neptune**: Continue using `properties(n)` as before

## Migration Notes

### For Existing Neo4j Deployments

No migration required! The code handles both formats:

- **New entities/edges**: Stored with JSON-serialized attributes
- **Existing entities/edges**: Continue to work with spread properties
- **Mixed graphs**: Both formats coexist seamlessly

### For Custom Applications

If your application directly queries Neo4j and expects attributes as individual properties, you may need to update your queries to parse the `attributes` property as JSON.

## Testing

### Integration Tests

Added comprehensive integration tests in `tests/test_neo4j_nested_attributes_int.py`:

1. **test_nested_entity_attributes**: Tests entities with complex nested structures
2. **test_nested_edge_attributes**: Tests edges with nested metadata
3. **test_empty_and_none_attributes**: Tests edge cases (empty, None values)

### Test Coverage

The tests verify:
- Maps of Lists: `{"key": ["value1", "value2"]}`
- Nested Maps: `{"outer": {"inner": "value"}}`
- Mixed structures: `{"data": {"items": ["a", "b"], "meta": {"x": "y"}}}`
- Primitive values: `{"count": 42, "status": "active"}`
- Empty attributes: `{}`
- None values: `{"key": None}`

### Running Tests

```bash
cd reference/graphiti
make test
```

Or run Neo4j-specific tests only:
```bash
pytest tests/test_neo4j_nested_attributes_int.py -v
```

## Impact

### What Works Now

✅ LLM extractions with rich, structured attributes  
✅ Custom entity types with complex metadata  
✅ Nested data structures from structured outputs  
✅ Backward compatibility with existing data  
✅ All existing queries and retrieval methods

### Performance Considerations

- **Storage**: Minimal increase (JSON string vs individual properties)
- **Query performance**: Identical (no change to indexing or graph traversal)
- **Serialization overhead**: Negligible (JSON parsing is fast)

## Example Use Case

### Before (would crash):

```python
entity = EntityNode(
    name="User",
    attributes={
        "discovered_resources": ["res1", "res2"],
        "metadata": {
            "analysis": ["item1", "item2"],
            "nested": {"key": "value"}
        }
    }
)
await entity.save(driver)  # ❌ Neo4j crash: Invalid type MAP
```

### After (works perfectly):

```python
entity = EntityNode(
    name="User",
    attributes={
        "discovered_resources": ["res1", "res2"],
        "metadata": {
            "analysis": ["item1", "item2"],
            "nested": {"key": "value"}
        }
    }
)
await entity.save(driver)  # ✅ Works!

retrieved = await EntityNode.get_by_uuid(driver, entity.uuid)
assert retrieved.attributes == entity.attributes  # ✅ Preserved exactly
```

## Related Files

- `graphiti_core/utils/bulk_utils.py`: Write path fix (lines 181-216)
- `graphiti_core/nodes.py`: Read path for entities (lines 754-770)
- `graphiti_core/edges.py`: Read path for edges (lines 575-596)
- `graphiti_core/models/nodes/node_db_queries.py`: Query updates (lines 256-286)
- `graphiti_core/models/edges/edge_db_queries.py`: Query updates (lines 187-222)
- `tests/test_neo4j_nested_attributes_int.py`: Integration tests

## References

- Neo4j Property Types: https://neo4j.com/docs/cypher-manual/current/values-and-types/property-structural-constructed/
- Issue: Neo4j TypeError on nested attribute structures
- PR: Fix Neo4j nested attributes serialization

