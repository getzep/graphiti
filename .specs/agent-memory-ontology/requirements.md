# Requirements Document

## Introduction

This feature standardizes “agent memory” extraction for multiple clients by defining a Graphiti service-side schema (`agent_memory_v1`) and enabling schema selection during ingestion, while keeping ingestion asynchronous and reliable.

### Goals

- Centralize agent-memory schema (relations/entities) in Graphiti service.
- Keep ingestion fast and resilient.
- Preserve backwards compatibility for existing ingestion clients.

### Non-goals

- Authorization / ACL enforcement based on ownership facts.
- Changes to Graphiti core extraction algorithms.

## Glossary

- **Schema ID**: A stable string that selects an ontology definition for ingestion (e.g. `agent_memory_v1`).
- **Ontology**: A bundle of entity type definitions, relation type definitions, and relation signature rules used by Graphiti extraction.
- **Episode**: A single ingested message/event processed into nodes and edges.
- **Fact**: An extracted edge returned by Graphiti search for use as recall memory.

## Requirements

### Requirement 1: Schema Selection on Ingest

**User Story:** As an integrator, I want to select a named schema when ingesting messages, so that extracted facts are consistent across clients.

#### Acceptance Criteria

1.1 THE Graphiti Service SHALL accept an optional `schema_id` field on `POST /messages`.
1.2 WHEN `schema_id` is provided, THE Graphiti Service SHALL apply the corresponding ontology when calling `graphiti.add_episode(...)`.
1.3 WHEN `schema_id` is not provided AND a message content contains `<graphiti_episode`, THE Graphiti Service SHALL apply `agent_memory_v1`.
1.4 WHEN `schema_id` is not provided AND a message content does not contain `<graphiti_episode`, THE Graphiti Service SHALL ingest using default Graphiti behavior.
1.5 WHEN an unknown `schema_id` is provided, THE Graphiti Service SHALL return a `422` validation error or a `400` client error with a clear message.

### Requirement 2: Durable Agent Memory Ontology v1

**User Story:** As an agent developer, I want the service to recognize common agent-memory relations (ownership, preferences, terminology, tasks), so that recall is more accurate and structured.

#### Acceptance Criteria

2.1 THE Graphiti Service SHALL define an `agent_memory_v1` ontology in a service-owned registry.
2.2 THE `agent_memory_v1` ontology SHALL define a bounded set of relation types intended for generic coding + project workflows.
2.3 THE `agent_memory_v1` ontology SHALL avoid mandatory attribute extraction (fieldless type models) to limit LLM overhead by default.

### Requirement 3: Reliable Asynchronous Ingestion

**User Story:** As an operator, I want background ingestion jobs to run reliably after `POST /messages` returns, so that clients can enqueue memory without slowing the agent loop.

#### Acceptance Criteria

3.1 THE Graphiti Service SHALL start exactly one background worker during app startup and stop it on shutdown.
3.2 THE Graphiti Service SHALL NOT close Graphiti resources that background jobs depend on before queued jobs finish.
3.3 WHEN a background job raises an exception, THE Graphiti Service SHALL log the error and continue processing subsequent jobs.

### Requirement 4: Documentation and Examples

**User Story:** As a developer, I want clear docs and a small demo showing schema-enabled ingestion and recall, so that I can validate deployments quickly.

#### Acceptance Criteria

4.1 THE repository SHALL document the health endpoint (`/healthcheck`) and basic redeploy steps.
4.2 THE repository SHALL include a minimal demo/example that uses `agent_memory_v1` ingestion and shows how to retrieve facts via `/search`.

### Requirement 5: Canonical Group IDs for Shared Memory

**User Story:** As an agent user, I want Copilot Chat and Codex to share the same durable memory, so that preferences and terminology carry across tools.

#### Acceptance Criteria

5.1 THE Graphiti Service SHALL expose an endpoint that resolves a canonical `group_id` from a `(scope, key)` pair.
5.2 THE endpoint SHALL support at least `user`, `workspace`, and `session` scopes.
5.3 WHEN given the same `(scope, key)`, THE Graphiti Service SHALL return the same `group_id` across requests and deployments.
5.4 The repository SHALL document recommended `key` derivation for the `user` scope using GitHub login (e.g. `github_login:<login>`).
