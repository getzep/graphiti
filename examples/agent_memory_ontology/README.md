# Agent Memory Ontology Demo (`agent_memory_v1`)

This demo shows how to ingest agent memory (preferences/terminology/etc.) into the Graphiti service using the service-owned `agent_memory_v1` schema, and then retrieve facts via `/search`.

## Prerequisites

- Graphiti service running (from repo root):

```bash
docker compose up -d --build graph neo4j
```

- Verify health:

```bash
curl -sS http://localhost:8000/healthcheck
```

## Ingest a memory directive

```bash
curl -sS http://localhost:8000/messages \
  -H 'content-type: application/json' \
  -d '{
    "group_id": "workspace-demo",
    "schema_id": "agent_memory_v1",
    "messages": [{
      "role_type": "user",
      "role": "user",
      "content": "<graphiti_episode kind=\"memory_directive\">preference (workspace): Keep diffs small and focused.</graphiti_episode>",
      "source_description": "demo"
    }]
  }'
```

Ingestion is asynchronous; depending on your LLM/embedding backend, processing may take a little while.

## Retrieve facts

```bash
curl -sS http://localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{
    "group_ids": ["workspace-demo"],
    "query": "What are my preferences for diffs?",
    "max_facts": 5
  }'
```

