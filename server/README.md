# graph-service

Graph service is a fast api server implementing the [graphiti](https://github.com/getzep/graphiti) package.

## Container Releases

The FastAPI server container is automatically built and published to Docker Hub when a new `graphiti-core` version is released to PyPI.

**Image:** `zepai/graphiti`

**Available tags:**
- `latest` - Latest stable release
- `0.22.1` - Specific version (matches graphiti-core version)

**Platforms:** linux/amd64, linux/arm64

The automated release workflow:
1. Triggers when `graphiti-core` PyPI release completes
2. Waits for PyPI package availability
3. Builds multi-platform Docker image
4. Tags with version number and `latest`
5. Pushes to Docker Hub

Only stable releases are built automatically (pre-release versions are skipped).

## Running Instructions

1. Ensure you have Docker and Docker Compose installed on your system.

2. Add `zepai/graphiti:latest` to your service setup

3. Make sure to pass the following environment variables to the service

   ```
   OPENAI_API_KEY=your_openai_api_key
   # Optional (useful for Azure OpenAI / local gateways)
   OPENAI_BASE_URL=https://api.openai.com/v1
   # Optional (defaults depend on graphiti-core)
   MODEL_NAME=gpt-4o-mini
   EMBEDDING_MODEL_NAME=text-embedding-3-small
   NEO4J_USER=your_neo4j_user
   NEO4J_PASSWORD=your_neo4j_password
   NEO4J_PORT=your_neo4j_port
   ```

4. This service depends on having access to a neo4j instance, you may wish to add a neo4j image to your service setup as well. Or you may wish to use neo4j cloud or a desktop version if running this locally.

   An example of docker compose setup may look like this:

   ```yml
      version: '3.8'

      services:
      graph:
         image: zepai/graphiti:latest
         ports:
            - "8000:8000"
         
         environment:
            - OPENAI_API_KEY=${OPENAI_API_KEY}
            - NEO4J_URI=bolt://neo4j:${NEO4J_PORT}
            - NEO4J_USER=${NEO4J_USER}
            - NEO4J_PASSWORD=${NEO4J_PASSWORD}
      neo4j:
         image: neo4j:5.22.0
         
         ports:
            - "7474:7474"  # HTTP
            - "${NEO4J_PORT}:${NEO4J_PORT}"  # Bolt
         volumes:
            - neo4j_data:/data
         environment:
            - NEO4J_AUTH=${NEO4J_USER}/${NEO4J_PASSWORD}

      volumes:
      neo4j_data:
   ```

5. Once you start the service, it will be available at `http://localhost:8000` (or the port you have specified in the docker compose file).

6. You may access the swagger docs at `http://localhost:8000/docs`. You may also access redocs at `http://localhost:8000/redoc`.

7. You may also access the neo4j browser at `http://localhost:7474` (the port depends on the neo4j instance you are using).

## Healthcheck

- `GET /healthcheck` returns a simple JSON payload (`{"status":"healthy"}`) when the service is up.

## Schema Selection (Agent Memory)

`POST /messages` supports an optional `schema_id` to select a service-owned ontology.

### `agent_memory_v1`

Designed for agent “memory” extraction across tools like Copilot Chat and Codex (ownership, preferences, terminology, tasks).

If `schema_id` is omitted, the service auto-selects `agent_memory_v1` when a message contains `<graphiti_episode ...>`.

Example:

```bash
curl -sS http://localhost:8000/messages \\
  -H 'content-type: application/json' \\
  -d '{
    "group_id": "workspace-demo",
    "schema_id": "agent_memory_v1",
    "messages": [{
      "role_type": "user",
      "role": "user",
      "content": "<graphiti_episode kind=\"memory_directive\">terminology (workspace): \"playbook\" means \"runbook docs\"</graphiti_episode>",
      "source_description": "demo"
    }]
  }'
```

To retrieve facts:

```bash
curl -sS http://localhost:8000/search \\
  -H 'content-type: application/json' \\
  -d '{
    "group_ids": ["workspace-demo"],
    "query": "What does playbook mean here?",
    "max_facts": 5
  }'
```

## Troubleshooting

- If `POST /messages` returns `202` but no episodes/facts appear, ensure you are running a build that keeps a single Graphiti client alive for background jobs (app-scoped client + app-scoped worker). Rebuild/redeploy the container (`docker compose up -d --build`).
