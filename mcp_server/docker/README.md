# Docker Deployment for Graphiti MCP Server

This directory contains Docker Compose configurations for running the Graphiti MCP server with different database backends.

## Quick Start

The default configuration uses **KuzuDB** (embedded, no external dependencies):

```bash
docker-compose up
```

## Database Options

### KuzuDB (Default) - Recommended
**File:** `docker-compose.yml`

Embedded graph database with persistent storage, no external dependencies.

```bash
docker-compose up
```

**Pros:**
- No external database required
- Fast startup
- Low resource usage
- Single file persistence

**Cons:**
- Less mature than Neo4j
- Limited tooling

### Neo4j
**File:** `docker-compose-neo4j.yml`

Full-featured graph database with web interface.

```bash
docker-compose -f docker-compose-neo4j.yml up
```

**Pros:**
- Mature, battle-tested
- Rich query language (Cypher)
- Web UI at http://localhost:7474
- Extensive tooling

**Cons:**
- Requires separate database container
- Higher resource usage (512MB+ RAM)
- Slower startup (30+ seconds)

### FalkorDB
**File:** `docker-compose-falkordb.yml`

Redis-based graph database with high performance.

```bash
docker-compose -f docker-compose-falkordb.yml up
```

**Pros:**
- Fast performance
- Redis-compatible
- Good for high throughput

**Cons:**
- Requires separate database container
- Less tooling than Neo4j

## Environment Variables

Create a `.env` file in this directory:

```bash
# Required for all configurations
OPENAI_API_KEY=your-api-key-here

# Optional
GRAPHITI_GROUP_ID=main
SEMAPHORE_LIMIT=10

# Database-specific (if using non-default values)
NEO4J_PASSWORD=yourpassword
FALKORDB_PASSWORD=yourpassword
```

## Switching Between Databases

To switch from one database to another:

1. Stop the current setup:
   ```bash
   docker-compose down  # or docker-compose -f docker-compose-[db].yml down
   ```

2. Start with the new database:
   ```bash
   docker-compose -f docker-compose-[neo4j|falkordb].yml up
   ```

Note: Data is not automatically migrated between different database types.

## Data Persistence

All configurations use Docker volumes for data persistence:

- **KuzuDB**: `kuzu_data` volume
- **Neo4j**: `neo4j_data` and `neo4j_logs` volumes  
- **FalkorDB**: `falkordb_data` volume

### Backup and Restore

**Backup:**
```bash
# Replace [volume_name] with the appropriate volume
docker run --rm -v [volume_name]:/data -v $(pwd):/backup alpine \
  tar czf /backup/backup.tar.gz -C /data .
```

**Restore:**
```bash
docker run --rm -v [volume_name]:/data -v $(pwd):/backup alpine \
  tar xzf /backup/backup.tar.gz -C /data
```

### Clear Data

To completely reset a database:

```bash
# Stop containers
docker-compose -f docker-compose-[db].yml down

# Remove volume
docker volume rm docker_[volume_name]

# Restart (creates fresh volume)
docker-compose -f docker-compose-[db].yml up
```

## Performance Comparison

| Database | Startup Time | Memory Usage | External Container | Best For |
|----------|-------------|--------------|-------------------|----------|
| KuzuDB | Instant | ~100MB | No | Development, embedded use |
| Neo4j | 30+ sec | ~512MB+ | Yes | Production, complex queries |
| FalkorDB | ~5 sec | ~200MB | Yes | High throughput, caching |

## Troubleshooting

### Port Conflicts
If port 8000 is already in use:
```bash
lsof -i :8000  # Find what's using the port
```

### Database Connection Issues
- **KuzuDB**: Check volume permissions
- **Neo4j**: Wait for health check, check logs
- **FalkorDB**: Ensure Redis is responding

### API Key Issues
Verify your `.env` file contains valid API keys:
```bash
cat .env | grep API_KEY
```

## Additional Documentation

- [KuzuDB Setup Guide](README-kuzu.md)
- [Main MCP Server Documentation](../docs/README.md)