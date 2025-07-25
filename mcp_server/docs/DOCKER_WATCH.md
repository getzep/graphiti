# Docker Compose Watch Development Workflow

This document describes how to use Docker Compose Watch for efficient development of the Graphiti MCP Server.

## Overview

Docker Compose Watch provides automatic file synchronization and container rebuilding when source code changes, enabling a fast development feedback loop.

## Prerequisites

- Docker Compose v2.22+ (required for `develop.watch` feature)
- Docker Engine 24.0+

## Quick Start

### Development Mode (Recommended)

```bash
# Start development environment with watch enabled
make dev

# Or manually:
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

### Production Mode

```bash
# Start production environment
make up

# Or manually:
docker compose up -d
```

## Configuration Details

### Watch Configuration

The `docker-compose.yml` includes watch configuration for the `graphiti-mcp` service:

```yaml
develop:
  watch:
    - path: ./src
      action: sync
      target: /app/src
    - path: ./pyproject.toml
      action: sync
      target: /app/pyproject.toml
    - path: ./uv.lock
      action: sync
      target: /app/uv.lock
    - path: ./Dockerfile
      action: rebuild
```

### Watch Actions

- **`sync`**: Copies files from host to container (fast)
- **`rebuild`**: Rebuilds the container image (slower, but necessary for dependency changes)

### Development vs Production

| Feature | Development | Production |
|---------|-------------|------------|
| Dockerfile | `Dockerfile.dev` | `Dockerfile` |
| Dependencies | Includes dev dependencies | Production only |
| File watching | Enabled | Disabled |
| Volume mounts | Source code mounted | No mounts |
| Hot reload | Yes | No |

## File Structure

```
mcp_server/
├── docker-compose.yml          # Base configuration
├── docker-compose.dev.yml      # Development overrides
├── Dockerfile                  # Production image
├── Dockerfile.dev              # Development image
├── .dockerignore               # Excludes unnecessary files
└── src/                        # Source code (watched)
    ├── graphiti_mcp_server.py
    └── oauth_wrapper.py
```

## Development Workflow

### 1. Start Development Environment

```bash
make dev
```

This will:
- Build the development Docker image
- Start Neo4j database
- Start the MCP server with watch enabled
- Mount source code for live development

### 2. Make Code Changes

Edit files in the `src/` directory. Changes will be automatically synchronized to the container.

### 3. View Logs

```bash
make logs
```

### 4. Stop Services

```bash
make down
```

## Environment Variables

### Development Mode

```bash
# Use development Dockerfile
export DOCKERFILE=Dockerfile.dev

# Start with development overrides
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Production Mode

```bash
# Use production Dockerfile (default)
export DOCKERFILE=Dockerfile

# Start production services
docker compose up -d
```

## Troubleshooting

### Watch Not Working

1. **Check Docker Compose version:**
   ```bash
   docker compose version
   ```
   Must be 2.22+ for watch support.

2. **Verify file permissions:**
   ```bash
   ls -la src/
   ```

3. **Check container logs:**
   ```bash
   docker compose logs graphiti-mcp
   ```

### Performance Issues

1. **Optimize `.dockerignore`:**
   - Exclude unnecessary files from build context
   - Current configuration excludes test files, cache directories, etc.

2. **Use specific watch paths:**
   - Only watch necessary directories
   - Avoid watching large directories like `node_modules/`

3. **Monitor resource usage:**
   ```bash
   docker stats
   ```

### Dependency Changes

When `pyproject.toml` or `uv.lock` changes:

1. **Automatic rebuild:** The watch configuration will trigger a container rebuild
2. **Manual rebuild:** If needed, run:
   ```bash
   docker compose build graphiti-mcp
   ```

## Best Practices

### 1. Use Development Mode for Active Development

```bash
make dev  # Includes watch, dev dependencies, volume mounts
```

### 2. Use Production Mode for Testing

```bash
make up   # Production image, no watch, no dev dependencies
```

### 3. Monitor Resource Usage

```bash
# Check container resource usage
docker stats

# View detailed container info
docker compose ps
```

### 4. Clean Up Regularly

```bash
# Remove unused containers and images
docker system prune

# Clean build cache if needed
docker builder prune
```

## Advanced Configuration

### Custom Watch Paths

Add additional watch paths in `docker-compose.dev.yml`:

```yaml
develop:
  watch:
    - path: ./config
      action: sync
      target: /app/config
    - path: ./tests
      action: sync
      target: /app/tests
```

### Environment-Specific Overrides

Create additional override files for different environments:

```bash
# Staging environment
docker compose -f docker-compose.yml -f docker-compose.staging.yml up

# Testing environment
docker compose -f docker-compose.yml -f docker-compose.test.yml up
```

## Integration with IDE

### VS Code

1. Install Docker extension
2. Use "Docker: Compose Up" command
3. Attach to running container for debugging

### PyCharm

1. Configure Docker Compose run configuration
2. Set environment variables
3. Enable remote debugging

## Monitoring and Debugging

### Health Checks

Services include health checks:

```yaml
healthcheck:
  test: ["CMD", "wget", "-O", "/dev/null", "http://localhost:7474"]
  interval: 10s
  timeout: 5s
  retries: 5
  start_period: 30s
```

### Logging

```bash
# View all logs
make logs

# View specific service logs
docker compose logs -f graphiti-mcp

# View logs with timestamps
docker compose logs -f -t graphiti-mcp
```

## Performance Optimization

### Build Optimization

1. **Layer caching:** Dependencies installed before source code
2. **Multi-stage builds:** Separate build and runtime stages
3. **Cache mounts:** UV cache mounted for faster dependency installation

### Runtime Optimization

1. **Volume mounts:** Source code mounted for instant updates
2. **Environment variables:** Optimized for development
3. **Resource limits:** Appropriate memory and CPU limits

## Security Considerations

### Development Mode

- Includes development dependencies
- Source code mounted as read-only
- Non-root user execution
- Minimal attack surface

### Production Mode

- Production dependencies only
- No source code mounts
- Optimized for security
- Regular security updates

## Next Steps

1. **Customize watch paths** for your specific development needs
2. **Add environment-specific configurations** for staging/testing
3. **Integrate with CI/CD** for automated testing
4. **Monitor performance** and optimize as needed
