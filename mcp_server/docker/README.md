# Docker Deployment for Graphiti MCP Server

This directory contains Docker Compose configurations for running the Graphiti MCP server with different graph database backends: KuzuDB, Neo4j, and FalkorDB.

## Quick Start

```bash
# Default configuration (KuzuDB)
docker-compose up

# Neo4j
docker-compose -f docker-compose-neo4j.yml up

# FalkorDB
docker-compose -f docker-compose-falkordb.yml up
```

## Environment Variables

Create a `.env` file in this directory with your API keys:

```bash
# Required
OPENAI_API_KEY=your-api-key-here

# Optional
GRAPHITI_GROUP_ID=main
SEMAPHORE_LIMIT=10

# Database-specific variables (see database sections below)
```

## Database Configurations

### KuzuDB

**File:** `docker-compose.yml` (default)

KuzuDB is an embedded graph database that runs within the application container.

#### Configuration

```bash
# Environment variables
KUZU_DB=/data/graphiti.kuzu  # Database file path (default: /data/graphiti.kuzu)
KUZU_MAX_CONCURRENT_QUERIES=10  # Maximum concurrent queries (default: 10)
```

#### Storage Options

**Persistent Storage (default):**
Data is stored in the `kuzu_data` Docker volume at `/data/graphiti.kuzu`.

**In-Memory Mode:**
```bash
KUZU_DB=:memory:
```
Note: Data will be lost when the container stops.

#### Data Management

**Backup:**
```bash
docker run --rm -v docker_kuzu_data:/data -v $(pwd):/backup alpine \
  tar czf /backup/kuzu-backup.tar.gz -C /data .
```

**Restore:**
```bash
docker run --rm -v docker_kuzu_data:/data -v $(pwd):/backup alpine \
  tar xzf /backup/kuzu-backup.tar.gz -C /data
```

**Clear Data:**
```bash
docker-compose down
docker volume rm docker_kuzu_data
docker-compose up  # Creates fresh volume
```

#### Gotchas
- KuzuDB data is stored in a single file/directory
- The database file can grow large with extensive data
- In-memory mode provides faster performance but no persistence

### Neo4j

**File:** `docker-compose-neo4j.yml`

Neo4j runs as a separate container service with its own web interface.

#### Configuration

```bash
# Environment variables
NEO4J_URI=bolt://neo4j:7687  # Connection URI (default: bolt://neo4j:7687)
NEO4J_USER=neo4j  # Username (default: neo4j)
NEO4J_PASSWORD=demodemo  # Password (default: demodemo)
NEO4J_DATABASE=neo4j  # Database name (default: neo4j)
USE_PARALLEL_RUNTIME=false  # Enterprise feature (default: false)
```

#### Accessing Neo4j

- **Web Interface:** http://localhost:7474
- **Bolt Protocol:** bolt://localhost:7687
- **MCP Server:** http://localhost:8000

Default credentials: `neo4j` / `demodemo`

#### Data Management

**Backup:**
```bash
# Backup both data and logs volumes
docker run --rm -v docker_neo4j_data:/data -v $(pwd):/backup alpine \
  tar czf /backup/neo4j-data-backup.tar.gz -C /data .
docker run --rm -v docker_neo4j_logs:/logs -v $(pwd):/backup alpine \
  tar czf /backup/neo4j-logs-backup.tar.gz -C /logs .
```

**Restore:**
```bash
# Restore both volumes
docker run --rm -v docker_neo4j_data:/data -v $(pwd):/backup alpine \
  tar xzf /backup/neo4j-data-backup.tar.gz -C /data
docker run --rm -v docker_neo4j_logs:/logs -v $(pwd):/backup alpine \
  tar xzf /backup/neo4j-logs-backup.tar.gz -C /logs
```

**Clear Data:**
```bash
docker-compose -f docker-compose-neo4j.yml down
docker volume rm docker_neo4j_data docker_neo4j_logs
docker-compose -f docker-compose-neo4j.yml up
```

#### Gotchas
- Neo4j takes 30+ seconds to start up - wait for the health check
- The web interface requires authentication even for local access
- Memory heap is configured for 512MB initial, 1GB max
- Page cache is set to 512MB
- Enterprise features like parallel runtime require a license

### FalkorDB

**File:** `docker-compose-falkordb.yml`

FalkorDB is a Redis-based graph database that runs as a separate container.

#### Configuration

```bash
# Environment variables
FALKORDB_URI=redis://falkordb:6379  # Connection URI (default: redis://falkordb:6379)
FALKORDB_PASSWORD=  # Password (default: empty)
FALKORDB_DATABASE=default_db  # Database name (default: default_db)
```

#### Accessing FalkorDB

- **Redis Protocol:** redis://localhost:6379
- **MCP Server:** http://localhost:8000

#### Data Management

**Backup:**
```bash
docker run --rm -v docker_falkordb_data:/data -v $(pwd):/backup alpine \
  tar czf /backup/falkordb-backup.tar.gz -C /data .
```

**Restore:**
```bash
docker run --rm -v docker_falkordb_data:/data -v $(pwd):/backup alpine \
  tar xzf /backup/falkordb-backup.tar.gz -C /data
```

**Clear Data:**
```bash
docker-compose -f docker-compose-falkordb.yml down
docker volume rm docker_falkordb_data
docker-compose -f docker-compose-falkordb.yml up
```

#### Gotchas
- FalkorDB uses Redis persistence mechanisms (AOF/RDB)
- Default configuration has no password - add one for production
- Database name is created automatically if it doesn't exist
- Redis commands can be used for debugging: `redis-cli -h localhost`

## Switching Between Databases

To switch from one database to another:

1. **Stop current setup:**
   ```bash
   docker-compose down  # or docker-compose -f docker-compose-[db].yml down
   ```

2. **Start new database:**
   ```bash
   docker-compose -f docker-compose-[neo4j|falkordb].yml up
   # or just docker-compose up for KuzuDB
   ```

Note: Data is not automatically migrated between different database types. You'll need to export from one and import to another using the MCP API.

## Troubleshooting

### Port Conflicts

If port 8000 is already in use:
```bash
# Find what's using the port
lsof -i :8000

# Change the port in docker-compose.yml
# Under ports section: "8001:8000"
```

### Container Won't Start

1. Check logs:
   ```bash
   docker-compose logs graphiti-mcp
   ```

2. Verify `.env` file exists and contains valid API keys:
   ```bash
   cat .env | grep API_KEY
   ```

3. Ensure Docker has enough resources allocated

### Database Connection Issues

**KuzuDB:**
- Check volume permissions: `docker exec graphiti-mcp ls -la /data`
- Verify database file isn't corrupted

**Neo4j:**
- Wait for health check to pass (can take 30+ seconds)
- Check Neo4j logs: `docker-compose -f docker-compose-neo4j.yml logs neo4j`
- Verify credentials match environment variables

**FalkorDB:**
- Test Redis connectivity: `redis-cli -h localhost ping`
- Check FalkorDB logs: `docker-compose -f docker-compose-falkordb.yml logs falkordb`

### Data Not Persisting

1. Verify volumes are created:
   ```bash
   docker volume ls | grep docker_
   ```

2. Check volume mounts in container:
   ```bash
   docker inspect graphiti-mcp | grep -A 5 Mounts
   ```

3. Ensure proper shutdown:
   ```bash
   docker-compose down  # Not docker-compose down -v (which removes volumes)
   ```

### Performance Issues

**KuzuDB:**
- Increase `KUZU_MAX_CONCURRENT_QUERIES`
- Consider using SSD for database file storage
- Monitor with: `docker stats graphiti-mcp`

**Neo4j:**
- Increase heap memory in docker-compose-neo4j.yml
- Adjust page cache size based on data size
- Check query performance in Neo4j browser

**FalkorDB:**
- Adjust Redis max memory policy
- Monitor with: `redis-cli -h localhost info memory`
- Consider Redis persistence settings (AOF vs RDB)

## Docker Resources

### Volumes

Each database configuration uses named volumes for data persistence:
- KuzuDB: `kuzu_data`
- Neo4j: `neo4j_data`, `neo4j_logs`
- FalkorDB: `falkordb_data`

### Networks

All configurations use the default bridge network. Services communicate using container names as hostnames.

### Resource Limits

No resource limits are set by default. To add limits, modify the docker-compose file:

```yaml
services:
  graphiti-mcp:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 1G
```

## Configuration Files

Each database has a dedicated configuration file in `../config/`:
- `config-docker-kuzu.yaml` - KuzuDB configuration
- `config-docker-neo4j.yaml` - Neo4j configuration
- `config-docker-falkordb.yaml` - FalkorDB configuration

These files are mounted read-only into the container at `/app/config/config.yaml`.