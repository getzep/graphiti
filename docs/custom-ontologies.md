# Custom Extraction Ontologies

Define per-lane extraction ontologies so that different `group_id` graphs use domain-specific entity types, relationship types, and extraction emphasis — instead of one global set.

## Why Custom Ontologies?

Graphiti's default extraction pipeline uses a single set of entity types for every episode. This works well for general-purpose memory, but falls short when you ingest domain-specific content:

- **Low-signal entities.** Generic types miss domain-specific concepts (e.g., extracting a plain "Topic" when you really need "RootCause" or "FeatureRequest").
- **Missing relationships.** Without domain-aware relationship types, the graph can't capture edges like `CAUSED_BY` or `RESOLVED_BY`.
- **No extraction guidance.** The LLM extractor has no domain context, so it guesses at salience.

Custom ontologies solve this by mapping each `group_id` to its own entity types, relationship types, and a free-text extraction emphasis prompt.

## Schema

Ontologies are defined in a YAML config file (by convention, `config/extraction_ontologies.yaml`). The top-level structure is:

```yaml
schema_version: 1

<group_id>:
  extraction_emphasis: <string>
  entity_types:
    - name: <EntityTypeName>
      description: <string>
  relationship_types:
    - name: <RELATIONSHIP_NAME>
      description: <string>
```

### Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `schema_version` | `int` | Yes | Schema version. Currently `1`. |
| `<group_id>` | mapping key | — | The `group_id` this ontology applies to. Must match the group_id used when calling `add_episode()`. |
| `extraction_emphasis` | `string` | No | Free-text prompt injected into the LLM extraction call. Tells the extractor what to prioritize. |
| `entity_types` | `list[{name, description}]` | No | Entity types to extract for this lane. Each becomes a Pydantic model at runtime. |
| `entity_types[].name` | `string` | Yes | PascalCase entity type name (e.g., `RootCause`, `FeatureRequest`). |
| `entity_types[].description` | `string` | Yes | Extraction instructions for the LLM. Be specific — this is the primary lever for extraction quality. |
| `relationship_types` | `list[{name, description}]` | No | Relationship types (graph edges) to extract. |
| `relationship_types[].name` | `string` | Yes | UPPER_SNAKE_CASE relationship name (e.g., `CAUSED_BY`, `RESOLVED_BY`). |
| `relationship_types[].description` | `string` | Yes | What this relationship means and which entity types it connects. |

### Entity Type Descriptions

The `description` field on each entity type is the most important lever for extraction quality. It's injected directly into the LLM extraction prompt. Write it as instructions:

```yaml
entity_types:
  - name: RootCause
    description: >
      The underlying technical reason an incident occurred.
      Look for statements like "the issue was caused by...",
      "root cause was...", or chains of "because" reasoning.
      Extract the specific mechanism, not just the symptom.
```

### Extraction Emphasis

The `extraction_emphasis` field provides high-level guidance to the extractor. Use it to set priorities when the LLM must decide what's worth extracting:

```yaml
extraction_emphasis: >
  Focus on root causes, blast radius, and remediation steps.
  Prioritize patterns that prevent recurrence over narrative
  details about the incident timeline.
```

## How It Works

### Loading and Resolution

1. **Startup.** The MCP server loads `extraction_ontologies.yaml` at initialization. Each `group_id` key is registered with its entity types, relationship types, and extraction emphasis.

2. **Per-episode resolution.** When `add_episode()` is called with a `group_id`, the server looks up that group's ontology. Entity type definitions are converted to Pydantic models (matching the `EntityTypeConfig` schema in `mcp_server/src/config/schema.py`).

3. **Extraction call.** The resolved entity types and extraction emphasis are passed to Graphiti's extraction pipeline, which injects them into the LLM prompt.

### Default Fallback

If a `group_id` is **not** listed in `extraction_ontologies.yaml`, extraction falls back to the entity types defined in your `GraphitiConfig` (`graphiti.entity_types` in `config.yaml`). If that list is also empty, Graphiti uses its built-in defaults.

This means you can:
- Define ontologies only for lanes that need domain-specific extraction.
- Leave general-purpose lanes (e.g., chat history) on the default types.
- Incrementally add ontologies as you identify domain-specific needs.

### Queue Service Routing

The queue service processes episodes sequentially per `group_id`. Each group's worker resolves its ontology independently, so different groups can extract in parallel with different entity types. See `mcp_server/src/services/queue_service.py` for the queue implementation.

## Minimal Example

Suppose you run a helpdesk and want to extract structured data from support tickets stored in `group_id=support_tickets`, while keeping your general chat memory in `group_id=main` with default types.

**`config/extraction_ontologies.yaml`:**

```yaml
schema_version: 1

support_tickets:
  extraction_emphasis: >
    Extract the customer's problem, product area affected,
    and resolution. Prioritize actionable patterns.

  entity_types:
    - name: CustomerIssue
      description: >
        A specific problem reported by a customer.
        Extract the symptom and severity if stated.

    - name: ResolutionStep
      description: >
        An action taken to resolve the issue.

  relationship_types:
    - name: RESOLVED_BY
      description: "CustomerIssue → ResolutionStep that fixed it"

    - name: ESCALATED_TO
      description: "CustomerIssue → team or tier it was escalated to"
```

With this config:
- Episodes added to `group_id=support_tickets` extract `CustomerIssue` and `ResolutionStep` entities with `RESOLVED_BY` / `ESCALATED_TO` edges.
- Episodes added to `group_id=main` (or any other unlisted group) use your default entity types.

## Example Config File

See [`config/extraction_ontologies.example.yaml`](../config/extraction_ontologies.example.yaml) for a complete example with three fictional lanes (customer support, product feedback, engineering incidents).

## Tips

- **Start small.** Define 3–5 entity types per lane. You can always add more after reviewing extraction results.
- **Be specific in descriptions.** Vague descriptions like "an important thing" produce noisy extractions. Tell the LLM exactly what patterns to look for.
- **Use extraction emphasis for priorities.** When a lane has many potential entities, the emphasis prompt tells the extractor what matters most.
- **Relationship descriptions should name their endpoints.** "CustomerIssue → ResolutionStep that fixed it" is better than "connects issues to fixes".
- **Review and iterate.** Run extraction on a sample, review the entities created, and refine descriptions based on what the LLM gets right/wrong.

## Operational Safety Notes

### Relationship type enforcement is guidance, not a hard gate

Current extraction uses ontology relationship types as prompt guidance. It does **not** guarantee that every emitted edge label is one of `relationship_types[].name`.

For governance-critical workflows, do not key policy logic directly off free-form relation labels unless you add a separate canonical mapping/validation layer.

### Defensive filtering for malformed legacy edges

Graph read paths now defensively skip malformed `RELATES_TO` rows that are missing required fields (`uuid`, `group_id`, `episodes`) in the main retrieval/hydration methods.

This hardening prevents legacy null-field edge rows from crashing deserialization during ingestion/dedupe flows.

### Migration-only deterministic dedupe mode

A migration-safe dedupe fallback exists for controlled backfills:
- `add_episode(..., dedupe_mode='deterministic')`

Behavior:
- exact-match dedupe still runs
- semantic LLM duplicate/contradiction resolution is skipped
- intended only for controlled migration/backfill workflows

Default remains `dedupe_mode='semantic'` for normal operation.

## Related Docs

- [Graphiti upstream docs](https://help.getzep.com/graphiti) — core runtime, drivers, entity types API
- [`config/extraction_ontologies.example.yaml`](../config/extraction_ontologies.example.yaml) — example config
- [`mcp_server/src/config/schema.py`](../mcp_server/src/config/schema.py) — `EntityTypeConfig` and `GraphitiConfig` schemas
- [`mcp_server/src/models/entity_types.py`](../mcp_server/src/models/entity_types.py) — built-in default entity types
