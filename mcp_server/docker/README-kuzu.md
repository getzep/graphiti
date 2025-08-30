# Running MCP Server with KuzuDB

This guide explains how to run the Graphiti MCP server with KuzuDB as the graph database backend using Docker Compose.

## Why KuzuDB?

KuzuDB is an embedded graph database that provides several advantages:
- **No external dependencies**: Unlike Neo4j, KuzuDB runs embedded within the application
- **Persistent storage**: Data is stored in a single file/directory
- **High performance**: Optimized for analytical workloads
- **Low resource usage**: Minimal memory and CPU requirements

## Quick Start

1. **Set up environment variables**:
   Create a `.env` file in the `docker` directory:
   ```bash
   OPENAI_API_KEY=your-api-key-here
   # Optional: Override default settings
   GRAPHITI_GROUP_ID=my-group
   KUZU_MAX_CONCURRENT_QUERIES=20
   ```

2. **Start the server**:
   ```bash
   docker-compose -f docker-compose-kuzu.yml up
   ```

3. **Access the MCP server**:
   The server will be available at `http://localhost:8000`

## Configuration

### Persistent Storage

By default, KuzuDB data is stored in a Docker volume at `/data/graphiti.kuzu`. This ensures your data persists across container restarts.

To use a different location, set the `KUZU_DB` environment variable:
```bash
KUZU_DB=/data/my-custom-db.kuzu
```

### In-Memory Mode

For testing or temporary usage, you can run KuzuDB in memory-only mode:
```bash
KUZU_DB=:memory:
```
Note: Data will be lost when the container stops.

### Performance Tuning

Adjust the maximum number of concurrent queries:
```bash
KUZU_MAX_CONCURRENT_QUERIES=20  # Default is 10
```

## Data Management

### Backup

To backup your KuzuDB data:
```bash
# Create a backup of the volume
docker run --rm -v mcp_server_kuzu_data:/data -v $(pwd):/backup alpine tar czf /backup/kuzu-backup.tar.gz -C /data .
```

### Restore

To restore from a backup:
```bash
# Restore from backup
docker run --rm -v mcp_server_kuzu_data:/data -v $(pwd):/backup alpine tar xzf /backup/kuzu-backup.tar.gz -C /data
```

### Clear Data

To completely clear the KuzuDB data:
```bash
# Stop the container
docker-compose -f docker-compose-kuzu.yml down

# Remove the volume
docker volume rm docker_kuzu_data

# Restart (will create fresh volume)
docker-compose -f docker-compose-kuzu.yml up
```

## Switching from Neo4j

If you're migrating from Neo4j to KuzuDB:

1. Export your data from Neo4j (if needed)
2. Stop the Neo4j-based setup: `docker-compose down`
3. Start the KuzuDB setup: `docker-compose -f docker-compose-kuzu.yml up`
4. Re-import your data through the MCP API

## Troubleshooting

### Container won't start
- Check that port 8000 is not in use: `lsof -i :8000`
- Verify your `.env` file has valid API keys

### Data not persisting
- Ensure the volume is properly mounted: `docker volume ls`
- Check volume permissions: `docker exec graphiti-mcp ls -la /data`

### Performance issues
- Increase `KUZU_MAX_CONCURRENT_QUERIES` for better parallelism
- Monitor container resources: `docker stats graphiti-mcp`

## Comparison with Neo4j Setup

| Feature | KuzuDB | Neo4j |
|---------|--------|-------|
| External Database | No | Yes |
| Memory Usage | Low (~100MB) | High (~512MB min) |
| Startup Time | Instant | 30+ seconds |
| Persistent Storage | Single file | Multiple files |
| Docker Services | 1 | 2 |
| Default Port | 8000 (MCP only) | 8000, 7474, 7687 |