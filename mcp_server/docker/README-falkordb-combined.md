# FalkorDB + Graphiti MCP Server Combined Image

This Docker setup bundles FalkorDB (graph database) and the Graphiti MCP Server into a single container image for simplified deployment.

## Overview

The combined image extends the official FalkorDB Docker image to include:
- **FalkorDB**: Redis-based graph database running on port 6379
- **FalkorDB Web UI**: Graph visualization interface on port 3000
- **Graphiti MCP Server**: Knowledge graph API on port 8000

Both services are managed by a startup script that launches FalkorDB as a daemon and the MCP server in the foreground.

## Quick Start

### Using Docker Compose (Recommended)

1. Create a `.env` file in the `mcp_server` directory:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional
GRAPHITI_GROUP_ID=main
SEMAPHORE_LIMIT=10
FALKORDB_PASSWORD=
```

2. Start the combined service:

```bash
cd mcp_server
docker compose -f docker/docker-compose-falkordb-combined.yml up
```

3. Access the services:
   - MCP Server: http://localhost:8000/mcp/
   - FalkorDB Web UI: http://localhost:3000
   - FalkorDB (Redis): localhost:6379

### Using Docker Run

```bash
docker run -d \
  -p 6379:6379 \
  -p 3000:3000 \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e GRAPHITI_GROUP_ID=main \
  -v falkordb_data:/var/lib/falkordb/data \
  zepai/graphiti-falkordb:latest
```

## Building the Image

### Build with Default Version

```bash
docker compose -f docker/docker-compose-falkordb-combined.yml build
```

### Build with Specific Graphiti Version

```bash
GRAPHITI_CORE_VERSION=0.22.0 docker compose -f docker/docker-compose-falkordb-combined.yml build
```

### Build Arguments

- `GRAPHITI_CORE_VERSION`: Version of graphiti-core package (default: 0.22.0)
- `MCP_SERVER_VERSION`: MCP server version tag (default: 1.0.0rc0)
- `BUILD_DATE`: Build timestamp
- `VCS_REF`: Git commit hash

## Configuration

### Environment Variables

All environment variables from the standard MCP server are supported:

**Required:**
- `OPENAI_API_KEY`: OpenAI API key for LLM operations

**Optional:**
- `BROWSER`: Enable FalkorDB Browser web UI on port 3000 (default: "1", set to "0" to disable)
- `GRAPHITI_GROUP_ID`: Namespace for graph data (default: "main")
- `SEMAPHORE_LIMIT`: Concurrency limit for episode processing (default: 10)
- `FALKORDB_PASSWORD`: Password for FalkorDB (optional)
- `FALKORDB_DATABASE`: FalkorDB database name (default: "default_db")

**Other LLM Providers:**
- `ANTHROPIC_API_KEY`: For Claude models
- `GOOGLE_API_KEY`: For Gemini models
- `GROQ_API_KEY`: For Groq models

### Volumes

- `/var/lib/falkordb/data`: Persistent storage for graph data
- `/var/log/graphiti`: MCP server and FalkorDB Browser logs

## Service Management

### View Logs

```bash
# All logs (both services stdout/stderr)
docker compose -f docker/docker-compose-falkordb-combined.yml logs -f

# Only container logs
docker compose -f docker/docker-compose-falkordb-combined.yml logs -f graphiti-falkordb
```

### Restart Services

```bash
# Restart entire container (both services)
docker compose -f docker/docker-compose-falkordb-combined.yml restart

# Check FalkorDB status
docker compose -f docker/docker-compose-falkordb-combined.yml exec graphiti-falkordb redis-cli ping

# Check MCP server status
curl http://localhost:8000/health
```

### Disabling the FalkorDB Browser

To disable the FalkorDB Browser web UI (port 3000), set the `BROWSER` environment variable to `0`:

```bash
# Using docker run
docker run -d \
  -p 6379:6379 \
  -p 3000:3000 \
  -p 8000:8000 \
  -e BROWSER=0 \
  -e OPENAI_API_KEY=your_key \
  zepai/graphiti-falkordb:latest

# Using docker-compose
# Add to your .env file:
BROWSER=0
```

When disabled, only FalkorDB (port 6379) and the MCP server (port 8000) will run.

## Health Checks

The container includes a health check that verifies:
1. FalkorDB is responding to ping
2. MCP server health endpoint is accessible

Check health status:
```bash
docker compose -f docker/docker-compose-falkordb-combined.yml ps
```

## Architecture

### Process Structure
```
start-services.sh (PID 1)
├── redis-server (FalkorDB daemon)
├── node server.js (FalkorDB Browser - background, if BROWSER=1)
└── uv run main.py (MCP server - foreground)
```

The startup script launches FalkorDB as a background daemon, waits for it to be ready, optionally starts the FalkorDB Browser (if `BROWSER=1`), then starts the MCP server in the foreground. When the MCP server stops, the container exits.

### Directory Structure
```
/app/mcp/                    # MCP server application
├── main.py
├── src/
├── config/
│   └── config.yaml          # FalkorDB-specific configuration
└── .graphiti-core-version   # Installed version info

/var/lib/falkordb/data/      # Persistent graph storage
/var/lib/falkordb/browser/   # FalkorDB Browser web UI
/var/log/graphiti/           # MCP server and Browser logs
/start-services.sh           # Startup script
```

## Benefits of Combined Image

1. **Simplified Deployment**: Single container to manage
2. **Reduced Network Latency**: Localhost communication between services
3. **Easier Development**: One command to start entire stack
4. **Unified Logging**: All logs available via docker logs
5. **Resource Efficiency**: Shared base image and dependencies

## Troubleshooting

### FalkorDB Not Starting

Check container logs:
```bash
docker compose -f docker/docker-compose-falkordb-combined.yml logs graphiti-falkordb
```

### MCP Server Connection Issues

1. Verify FalkorDB is running:
```bash
docker compose -f docker/docker-compose-falkordb-combined.yml exec graphiti-falkordb redis-cli ping
```

2. Check MCP server health:
```bash
curl http://localhost:8000/health
```

3. View all container logs:
```bash
docker compose -f docker/docker-compose-falkordb-combined.yml logs -f
```

### Port Conflicts

If ports 6379, 3000, or 8000 are already in use, modify the port mappings in `docker-compose-falkordb-combined.yml`:

```yaml
ports:
  - "16379:6379"  # Use different external port
  - "13000:3000"
  - "18000:8000"
```

## Production Considerations

1. **Resource Limits**: Add resource constraints in docker-compose:
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
```

2. **Persistent Volumes**: Use named volumes or bind mounts for production data
3. **Monitoring**: Export logs to external monitoring system
4. **Backups**: Regular backups of `/var/lib/falkordb/data` volume
5. **Security**: Set `FALKORDB_PASSWORD` in production environments

## Comparison with Separate Containers

| Aspect | Combined Image | Separate Containers |
|--------|---------------|---------------------|
| Setup Complexity | Simple (one container) | Moderate (service dependencies) |
| Network Latency | Lower (localhost) | Higher (container network) |
| Resource Usage | Lower (shared base) | Higher (separate images) |
| Scalability | Limited | Better (scale independently) |
| Debugging | Harder (multiple processes) | Easier (isolated services) |
| Production Use | Development/Single-node | Recommended |

## See Also

- [Main MCP Server README](../README.md)
- [FalkorDB Documentation](https://docs.falkordb.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
