# Docker Build Setup for Custom MCP Server

## Overview

This project uses GitHub Actions to automatically build a custom Docker image with MCP server changes and push it to Docker Hub. The image uses the **official graphiti-core from PyPI** (not local source).

## Key Files

### GitHub Actions Workflow
- **File**: `.github/workflows/build-custom-mcp.yml`
- **Triggers**: 
  - Automatic: Push to `main` branch with changes to `graphiti_core/`, `mcp_server/`, or the workflow file
  - Manual: Workflow dispatch from Actions tab
- **Builds**: Multi-platform image (AMD64 + ARM64)
- **Pushes to**: `lvarming/graphiti-mcp` on Docker Hub

### Dockerfile
- **File**: `mcp_server/docker/Dockerfile.standalone` (official Dockerfile)
- **NOT using custom Dockerfile** - we use the official one
- **Pulls graphiti-core**: From PyPI (official version)
- **Includes**: Custom MCP server code with added tools

## Docker Hub Configuration

### Required Secret
- **Secret name**: `DOCKERHUB_TOKEN`
- **Location**: GitHub repository → Settings → Secrets and variables → Actions
- **Permissions**: Read & Write
- **Username**: `lvarming`

### Image Tags
Each build creates multiple tags:
- `lvarming/graphiti-mcp:latest`
- `lvarming/graphiti-mcp:mcp-X.Y.Z` (MCP server version)
- `lvarming/graphiti-mcp:mcp-X.Y.Z-core-A.B.C` (with graphiti-core version)
- `lvarming/graphiti-mcp:sha-xxxxxxx` (git commit hash)

## What's in the Custom Image

✅ **Included**:
- Official graphiti-core from PyPI (e.g., v0.23.0)
- Custom MCP server code with:
  - `get_entities_by_type` tool
  - `compare_facts_over_time` tool
  - Other custom MCP tools in `mcp_server/src/graphiti_mcp_server.py`

❌ **NOT Included**:
- Local graphiti-core changes (we don't modify it)
- Custom server/ changes (we don't modify it)

## Build Process

1. **Code pushed** to main branch on GitHub
2. **Workflow triggers** automatically
3. **Extracts versions** from pyproject.toml files
4. **Builds image** using official `Dockerfile.standalone`
   - Context: `mcp_server/` directory
   - Uses graphiti-core from PyPI
   - Includes custom MCP server code
5. **Pushes to Docker Hub** with multiple tags
6. **Build summary** posted in GitHub Actions

## Usage in Deployment

### Unraid
```yaml
Repository: lvarming/graphiti-mcp:latest
```

### Docker Compose
```yaml
services:
  graphiti-mcp:
    image: lvarming/graphiti-mcp:latest
    # ... environment variables
```

### LibreChat Integration
```yaml
mcpServers:
  graphiti-memory:
    url: "http://graphiti-mcp:8000/mcp/"
```

## Important Constraints

### DO NOT modify graphiti_core/
- We use the official version from PyPI
- Local changes break upstream compatibility
- Causes Docker build issues
- Makes merging with upstream difficult

### DO modify mcp_server/
- This is where custom tools live
- Changes automatically included in next build
- Push to main triggers new build

## Monitoring Builds

Check build status at:
- https://github.com/Varming73/graphiti/actions
- Look for "Build Custom MCP Server" workflow
- Build takes ~5-10 minutes

## Troubleshooting

### Build Fails
- Check Actions tab for error logs
- Verify DOCKERHUB_TOKEN is valid
- Ensure mcp_server code is valid

### Image Not Available
- Check Docker Hub: https://hub.docker.com/r/lvarming/graphiti-mcp
- Verify build completed successfully
- Check repository is public on Docker Hub

### Wrong Version
- Tags are based on pyproject.toml versions
- Check `mcp_server/pyproject.toml` version
- Check root `pyproject.toml` for graphiti-core version

## Documentation

Full guides available in `DOCS/`:
- `GitHub-DockerHub-Setup.md` - Complete setup instructions
- `Librechat.setup.md` - LibreChat + Unraid deployment
- `README.md` - Navigation and overview
