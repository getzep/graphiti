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
| `extraction_mode` | `string` | No | Extraction behaviour: `permissive` (default) or `constrained_soft`. See [Extraction Modes](#extraction-modes). |
| `intent_guidance` | `string` | No | Canonical field for per-lane LLM focus instructions. In `constrained_soft` mode, placed in a dedicated `<LANE_INTENT>` block (not appended). Falls back to `extraction_emphasis` if not set. |
| `extraction_emphasis` | `string` | No | Legacy alias for `intent_guidance`. Still works; prefer `intent_guidance` for new configs. |
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

## Extraction Modes

The ontology config supports two extraction modes per lane:

| Mode | Default? | Behaviour |
|---|---|---|
| `permissive` | ✅ Yes | Extract broadly; all relationship types allowed. Extraction emphasis/intent guidance is appended as additional context. |
| `constrained_soft` | No | Ontology-conformant mode. Uses dedicated prompt branches (not appended text) and code-level enforcement to reduce noise. |

### When to use `constrained_soft`

Use constrained_soft for lanes where:
- The graph should closely mirror a defined domain model (e.g. voice fingerprint analysis, rhetorical pattern extraction)
- Generic connector edges (RELATES_TO, MENTIONS, IS_RELATED_TO) create noise that harms retrieval precision
- Edge canonicalization is needed to snap near-miss names to ontology names

**Do not** use constrained_soft for general-purpose or exploratory lanes — it reduces recall for off-ontology content.

### `intent_guidance` — Lane-Level LLM Focus

`intent_guidance` is the canonical field for per-lane LLM instructions. In `constrained_soft` mode, it is placed in a dedicated `<LANE_INTENT>` block in the prompt (not appended to generic instructions, which would create conflicting directives).

`extraction_emphasis` is a backward-compatible alias — if `intent_guidance` is not set, `extraction_emphasis` is used as the fallback.

### Schema Example — constrained_soft lane

```yaml
schema_version: 1

voice_analysis:
  extraction_mode: constrained_soft
  intent_guidance: >
    Focus on rhetorical techniques, voice patterns, and structural moves.
    Extract what makes this writing distinctive — characteristic phrases,
    tone shifts, structural patterns. Do not extract generic topic nodes.
    Conform to the defined FACT_TYPES.

  entity_types:
    - name: RhetoricalMove
      description: >
        A specific writing technique used in the piece
        (e.g. cold open with absurd analogy, callback close, one-liner pivot).
        Extract the technique name and brief description.

    - name: VoiceQuality
      description: >
        A tone or register characteristic
        (e.g. irreverent-casual, deadpan-authoritative, erudite-compressed).

  relationship_types:
    - name: USES_MOVE
      description: "Piece → RhetoricalMove it employs"

    - name: EXHIBITS
      description: "Piece → VoiceQuality it demonstrates"
```

In this example:
- The LLM extraction prompt uses a **dedicated constrained_soft branch** (not the permissive prompt with appended instructions)
- Intent guidance is placed in `<LANE_INTENT>` block — not appended after conflicting base instructions
- After LLM extraction: near-miss edge names are snapped to ontology (e.g. `USE_MOVE` → `USES_MOVE`)
- Generic edges (RELATES_TO, MENTIONS, etc.) with no ontology match are dropped in code
- Off-ontology but semantically specific edges (e.g. `WRITTEN_IN_RESPONSE_TO`) are kept

### Code enforcement in constrained_soft

Two enforcement passes run after LLM extraction (in code, not prompt):

1. **Canonicalization**: near-miss relation types are snapped to the closest ontology name using `difflib.SequenceMatcher` (threshold ≥ 0.78).
2. **Noise filter**: relation types in a known generic set (`RELATES_TO`, `MENTIONS`, `IS_RELATED_TO`, `CONNECTED_TO`, etc.) are dropped when they have no ontology match.

Domain-specific off-ontology edges (e.g. `CRITICIZED_BY`, `WRITTEN_IN_RESPONSE_TO`) are **kept** — only generic connector noise is filtered.

## Operational Safety Notes

### Relationship type enforcement

**Permissive mode (default):** Extraction uses ontology relationship types as prompt guidance only. It does **not** guarantee that every emitted edge label is one of `relationship_types[].name`. Off-ontology edges may appear.

**constrained_soft mode:** After LLM extraction, two code-level enforcement passes run:
1. Near-miss canonicalization (difflib ≥ 0.78 threshold) snaps edge names to ontology names.
2. Generic off-ontology edges (RELATES_TO, MENTIONS, etc.) are dropped.

For governance-critical workflows, use `constrained_soft` mode and do not key policy logic directly off free-form relation labels without a separate validation layer.

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

## OM Ontology: `s1_observational_memory`

The Observational Memory (OM) pipeline uses a dedicated semantic domain. Unlike normal
lane ontologies (which govern Graphiti/MCP extraction), the OM ontology governs the
`om_extractor` block in `mcp_server/config/extraction_ontologies.yaml` and the entity
types emitted by `scripts/om_compressor.py`.

### Built-in OM Entity Types

The compressor's deterministic extractor uses these types (no config required):

| Type | When extracted |
|---|---|
| `Judgment` | Content contains "because" or "decision" |
| `OperationalRule` | Content contains "rule" or "always" |
| `Commitment` | Content contains "commit" or "promise" |
| `Friction` | Content contains "problem", "friction", or "blocked" |
| `WorldState` | Default fallback |

### Allowed OM Edge Types

The following edge types are enforced via an allowlist in the compressor (free-form
relation type strings are blocked at write time):

```
MOTIVATES   GENERATES   SUPERSEDES   ADDRESSES   RESOLVES
```

`SUPERSEDES` is the canonical supersession relation (active voice: new → old).
If your storage backend materializes edges under a generic relationship type
(e.g. `RELATES_TO`), score semantic relation meaning from the relation-name
property (`r.name`), not from `type(r)` alone.

### YAML Config for OM Extractor

The `om_extractor` block in `mcp_server/config/extraction_ontologies.yaml` controls
the model and prompt used for LLM-backed extraction (when enabled):

```yaml
schema_version: "2026-02-17"

om_extractor:
  model_id: "gpt-5.1-codex-mini"
  prompt_template: |-
    You are the Observational Memory extractor.
    Produce deterministic mutation candidates from transcript chunks.
    Preserve source provenance and avoid speculative claims.
```

### Adding a Dedicated OM Lane

To add a `s1_observational_memory` group for OM-specific extraction via Graphiti:

```yaml
s1_observational_memory:
  extraction_emphasis: >-
    Focus on durable observations: decisions, commitments, friction points,
    and operational rules. Extract only claims the speaker asserts as fact.
    Avoid hypotheticals, questions, and third-party assertions.

  entity_types:
    - name: WorldState
      description: >-
        A factual assertion about the current state of the world
        (e.g. "Neo4j is running on port 7687", "the team uses Slack").
    - name: Judgment
      description: >-
        A decision or reasoned conclusion (e.g. "we decided to use FalkorDB
        because it's faster for graph traversal"). Look for "because", "decided",
        "so we chose".
    - name: OperationalRule
      description: >-
        A standing policy or recurring process rule (e.g. "always use
        update-with-hotfixes.sh", "run ruff before committing"). Look for "always",
        "rule", "policy", "never".
    - name: Commitment
      description: >-
        An explicit promise or stated intent (e.g. "I'll ship the PR by Friday",
        "we're committed to the migration by end of Q1").
    - name: Friction
      description: >-
        A blocker, problem, or source of friction (e.g. "the migration is blocked
        by the FalkorDB schema drift", "backfill keeps failing on chunk 3").

  relationship_types:
    - name: MOTIVATES
      description: "WorldState or Judgment → the Commitment or action it motivates"
    - name: GENERATES
      description: "Friction → Judgment (friction generates a decision to address it)"
    - name: SUPERSEDES
      description: "New WorldState/Rule → older stale WorldState/Rule it replaces"
    - name: ADDRESSES
      description: "Judgment → Friction or Commitment it directly addresses"
    - name: RESOLVES
      description: "Judgment → Friction that it closes (written automatically on CLOSED transition)"
```

> **Note:** The compressor's built-in extractor does not require this YAML config. The
> YAML `s1_observational_memory` block is only used if you ingest OM content via the
> MCP `add_episode()` path with `group_id="s1_observational_memory"`.

---

## Related Docs

- [Graphiti upstream docs](https://help.getzep.com/graphiti) — core runtime, drivers, entity types API
- [`config/extraction_ontologies.example.yaml`](../config/extraction_ontologies.example.yaml) — example config
- [`mcp_server/src/config/schema.py`](../mcp_server/src/config/schema.py) — `EntityTypeConfig` and `GraphitiConfig` schemas
- [`mcp_server/src/models/entity_types.py`](../mcp_server/src/models/entity_types.py) — built-in default entity types
- [OM Operations Runbook](runbooks/om-operations.md) — compressor config, chunk semantics, lock ordering
