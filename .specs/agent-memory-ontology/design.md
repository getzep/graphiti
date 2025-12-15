# Design Document: Agent Memory Ontology (Graphiti Service)

## Overview

Copilot Chat and Codex both ingest conversation history into Graphiti so agents can recall durable context (preferences, terminology, ownership, project state) across sessions and workspaces. Today, each client has to “decide” what schema to use (entity types, relation types) and how to format episodes. This leads to drift and makes it hard to evolve the memory graph consistently.

This feature adds a **Graphiti-side ontology registry** so clients can select a named schema (starting with `agent_memory_v1`) and get consistent extraction behavior. It also hardens the Graphiti service’s async ingestion so queued jobs run reliably without blocking the request path.

### Goals

- Provide a **single source of truth** for the “agent memory” schema (entity types + relation types) on the Graphiti service.
- Allow clients to opt into the schema via an explicit `schema_id`, and default automatically for `<graphiti_episode …>` payloads.
- Keep ingestion **non-blocking** (fast `202 Accepted`) and resilient (job failures don’t stop the worker).
- Keep overhead low: schema should not require per-edge/per-node attribute extraction by default.

### Non-goals

- Changing Graphiti core extraction prompts or algorithms (`graphiti_core/*`).
- Enforcing authorization / ACLs based on ownership facts (ownership is modeled, not enforced).
- Mandating a client-side message format beyond the existing `<graphiti_episode …>` convention.

## Current Architecture

### Graphiti service ingest (today)

- `POST /messages` accepts `group_id` and a list of message DTOs.
- Each message is enqueued into an in-process async worker queue.
- Worker calls `graphiti.add_episode(...)` which runs Graphiti core extraction and writes nodes/edges into Neo4j.

### Pain points

- **Schema drift:** no service-level mechanism to select/standardize `entity_types`, `edge_types`, or `edge_type_map`.
- **Async ingest reliability risk:** background jobs must not depend on per-request resources that can be closed once the HTTP request finishes.

## Proposed Architecture

### Ontology registry + schema selection

Introduce a `graph_service.ontologies` module with:

- `schema_id` strings (start with `agent_memory_v1`)
- `entity_types`, `edge_types`, `edge_type_map`, `excluded_entity_types` for each schema
- a resolver: `resolve_ontology(schema_id: str | None, message_content: str) -> Ontology | None`

Ingest routing:

- Extend `AddMessagesRequest` with optional `schema_id`.
- For each message:
  - If `request.schema_id` is present, use it.
  - Else, if `message.content` contains `<graphiti_episode`, default to `agent_memory_v1`.
  - Else, use default Graphiti behavior (no custom schema).

### Reliable background ingest lifecycle

Run the async worker for the lifetime of the FastAPI app and keep a single `ZepGraphiti` instance in `app.state`:

- Create Graphiti client once during app startup (lifespan).
- Start the ingest worker at startup, stop it at shutdown.
- Ensure worker errors do not crash the loop and that queue accounting is correct (`task_done`).

This keeps the API responsive while ensuring background jobs can safely use Graphiti resources.

## Components

- `server/graph_service/ontologies/agent_memory_v1.py`
  - Defines `agent_memory_v1` schema: entity types + edge types (docstring-driven, no fields).
- `server/graph_service/ontologies/registry.py`
  - Central registry + resolver helpers.
- `server/graph_service/dto/ingest.py`
  - Adds `schema_id` to `AddMessagesRequest`.
- `server/graph_service/routers/ingest.py`
  - Selects ontology per message and passes types/maps into `graphiti.add_episode(...)`.
  - Worker resiliency.
- `server/graph_service/main.py` / `server/graph_service/zep_graphiti.py`
  - App-scoped Graphiti initialization and dependency injection.

## Data & Control Flow

```
Copilot Chat / Codex
  └─ POST /messages (group_id, messages[], schema_id?)
       └─ enqueue jobs (202 Accepted)
            └─ async worker executes sequentially
                 └─ Graphiti.add_episode(..., entity_types/edge_types/edge_type_map?)
                      └─ extract nodes + edges (LLM)
                      └─ write to Neo4j + build embeddings
  └─ POST /search (group_ids[], query)
       └─ returns relevant edges (“facts”) for recall
```

## Integration Points

- **Clients** can remain unchanged if they already wrap durable/structured memory as `<graphiti_episode …>…</graphiti_episode>`.
- **Explicit opt-in**: clients may set `schema_id=agent_memory_v1` in `POST /messages` for deterministic behavior.
- Docker compose: pass through `OPENAI_BASE_URL`, `MODEL_NAME`, `EMBEDDING_MODEL_NAME` so service behavior matches client/test environments.

## Migration / Rollout Strategy

- Backward compatible: `schema_id` is optional.
- Safe default: schema only auto-applies when `<graphiti_episode` is present, minimizing unintended changes for generic ingestion callers.
- Versioning: evolve via new schema ids (`agent_memory_v2`) rather than mutating `agent_memory_v1` semantics.

## Performance / Reliability / Security / UX Considerations

- **Performance:** keep schema types fieldless to avoid extra attribute-extraction LLM calls; ingestion remains async.
- **Reliability:** app-scoped Graphiti client ensures background jobs don’t fail due to closed drivers; worker continues after exceptions.
- **Security:** schema encourages modeling stable identifiers (hashed ids) and avoids requiring PII; clients should continue to redact secrets before promotion.

## Risks and Mitigations

- **Extraction quality depends on model/prompting:** keep ontology small and descriptive; version schema if changes needed.
- **Schema drift across deployments:** include schema id in docs/examples; keep registry centralized.
- **Operational misconfiguration:** document required env vars and health endpoints (`/healthcheck`).

## Future Enhancements

- Structured JSON episodes (`EpisodeType.json`) for deterministic ingestion of ownership/metadata without LLM parsing.
- MCP server parity: expose the same schema ids/ontology selection through MCP tools.
- Organization/team scope groups and cross-group linking policies.

