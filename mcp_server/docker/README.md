# Docker Deployment for Graphiti MCP Server

This directory ships a single Docker Compose setup based on a combined FalkorDB + MCP image.

## Quick Start

```bash
cd graphiti/mcp_server
docker compose -f docker/docker-compose.yml up --build
```

Services:
- MCP endpoint: `http://localhost:8000/mcp`
- FalkorDB (Redis): `localhost:6379`
- FalkorDB Browser: `http://localhost:3000`

## Compose File

- `docker/docker-compose.yml`: combined FalkorDB + Graphiti MCP container

## Environment Variables

Minimum:

```bash
OPENAI_API_KEY=your_key
```

Common options:

```bash
OPENAI_API_URL=https://api.openai.com/v1
OPENAI_MODEL=openai/gpt-4.1-mini
GOOGLE_API_KEY=your_google_key   # if embedder provider is Gemini
GRAPHITI_GROUP_ID=main
SEMAPHORE_LIMIT=10
FALKORDB_PASSWORD=
FALKORDB_DATABASE=default_db
```

## Build Arguments

The compose file forwards build args to `Dockerfile`:

- `GRAPHITI_CORE_VERSION` (default `0.28.1`)
- `MCP_PROVIDER_EXTRA` (default `gemini`, set `providers` for full provider bundle)
- `MCP_SERVER_VERSION` (default `1.0.1`)
- `BUILD_DATE`
- `VCS_REF`

Example:

```bash
MCP_PROVIDER_EXTRA=providers docker compose -f docker/docker-compose.yml build
```

## Standalone Image

`Dockerfile.standalone` is available when you want an MCP-only container connected to an external database.
