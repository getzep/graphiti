## Summary

Extends `ExtractedEntity` with an optional `attributes` field, populated during entity extraction.

## Type of Change

- [ ] Bug fix
- [x] New feature
- [ ] Performance improvement
- [ ] Documentation/Tests

## Objective

Entity attribute extraction currently requires defining `entity_types` with Pydantic models. This triggers O(n) additional LLM calls via `_extract_entity_attributes` for n entities. Without predefined schemas, no attributes are extracted.

The entity extraction pass already processes full episode context. Attribute identification is a natural byproduct of entity recognition. This change extends the extraction prompt to request attributes inline, achieving attribute discovery with zero marginal LLM calls and no schema requirement.

```python
{"name": "Acme Corp", "entity_type_id": 0, "attributes": {"employee_count": 150}}
```

New attributes are also merged into existing nodes during deduplication.

Default empty dict ensures backward compatibility.

## Testing

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All existing tests pass

## Breaking Changes

- [ ] This PR contains breaking changes

## Checklist

- [ ] Code follows project style guidelines (`make lint` passes)
- [x] Self-review completed
- [ ] Documentation updated where necessary
- [x] No secrets or sensitive information committed

## Related Issues

Closes #
