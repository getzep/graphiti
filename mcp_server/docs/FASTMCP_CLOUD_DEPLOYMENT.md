# FastMCP Cloud Deployment Guide

This guide covers deploying the Graphiti MCP server to FastMCP Cloud, a managed hosting platform for MCP servers.

## Overview

FastMCP Cloud is a managed platform that:
- Automatically builds and deploys your MCP server from GitHub
- Provides a unique HTTPS URL for your server
- Handles SSL certificates and authentication
- Auto-redeploys on pushes to `main` branch
- Creates preview deployments for pull requests

**Note:** FastMCP Cloud is currently **free while in beta**.

## Prerequisites

Before deploying to FastMCP Cloud, you need:

1. **GitHub Account** - FastMCP Cloud integrates with GitHub repos
2. **Cloud Database** - Neo4j Aura or FalkorDB Cloud (must be internet-accessible)
3. **API Keys** - OpenAI (required), Anthropic (optional)
4. **Verified Repo** - Run the verification script first

### Pre-Deployment Verification

Run the verification script to check your server is ready:

```bash
cd mcp_server
uv run python scripts/verify_fastmcp_cloud_readiness.py
```

This checks:
- Server is discoverable via `fastmcp inspect`
- Dependencies are properly declared in `pyproject.toml`
- Environment variables are documented
- No secrets are committed to git
- Server can be imported successfully
- Entrypoint format is correct

All checks should pass before deploying.

## Deployment Steps

### Step 0: Validate Your Server Locally

**Before deploying to FastMCP Cloud, validate with BOTH static and runtime checks.**

#### Static Validation: `fastmcp inspect`

Checks that your server module can be imported and tools are registered:

```bash
cd mcp_server
uv run fastmcp inspect src/graphiti_mcp_server.py:mcp
```

**Expected successful output:**

```
Name: Graphiti Agent Memory
Version: <version>
Tools: 9 found
  - add_memory: Add an episode to memory
  - search_nodes: Search for nodes in the graph memory
  - search_memory_facts: Search the graph memory for relevant facts
  - get_episodes: Get episodes from the graph memory
  - get_entity_edge: Get an entity edge from the graph memory by its UUID
  - delete_episode: Delete an episode from the graph memory
  - delete_entity_edge: Delete an entity edge from the graph memory
  - clear_graph: Clear all data from the graph for specified group IDs
  - get_status: Get the status of the Graphiti MCP server
```

#### Runtime Validation: `fastmcp dev` (ESSENTIAL!)

**The `inspect` command only checks imports - it does NOT catch runtime initialization issues.**

Run the interactive inspector to test actual server initialization:

```bash
cd mcp_server
uv run fastmcp dev src/graphiti_mcp_server.py:mcp
```

This starts your server and opens an interactive web UI at `http://localhost:6274`.

**Critical test in the web UI:**

1. Open `http://localhost:6274` in your browser
2. Click the "get_status" tool
3. Click "Execute"
4. **Expected:** `{"status": "ok", "message": "Graphiti MCP server is running and connected to Neo4j database"}`
5. **If you see:** `{"status": "error", "message": "Graphiti service not initialized"}` - **DO NOT DEPLOY**

### Step 1: Set Up Cloud Database

#### Option A: Neo4j Aura (Recommended for Neo4j users)

1. Visit [Neo4j Aura](https://neo4j.com/cloud/aura/)
2. Create a free instance
3. Note your connection details:
   - URI: `neo4j+s://xxxxx.databases.neo4j.io`
   - Username: `neo4j`
   - Password: (generated)

#### Option B: FalkorDB Cloud

1. Visit [FalkorDB Cloud](https://cloud.falkordb.com)
2. Create an instance
3. Note your connection details:
   - URI: `redis://username:password@host:port`
   - Database: `default_db`

**Important:** Local databases (localhost) will NOT work with FastMCP Cloud. You must use a cloud-hosted database.

### Step 2: Prepare Your Repository

1. **Ensure `pyproject.toml` is complete**

   FastMCP Cloud automatically detects dependencies from `pyproject.toml`:
   ```toml
   [project]
   dependencies = [
       "fastmcp>=2.13.3",
       "graphiti-core[falkordb]>=0.23.1",
       "pydantic>=2.0.0",
       "pydantic-settings>=2.0.0",
       "python-dotenv>=1.0.0",
       # ... other dependencies
   ]
   ```

2. **Verify `.env` is in `.gitignore`**

   ```bash
   git check-ignore -v .env
   # Should output: .gitignore:XX:.env    .env
   ```

3. **Commit and push your code**

   ```bash
   git add .
   git commit -m "Prepare for FastMCP Cloud deployment"
   git push origin main
   ```

### Step 3: Create FastMCP Cloud Project

1. **Visit [fastmcp.cloud](https://fastmcp.cloud)**

2. **Sign in with your GitHub account**

3. **Create a new project:**
   - Click "Create Project"
   - Select your repository
   - Repository can be public or private

4. **Configure project settings:**

   | Setting | Value | Notes |
   |---------|-------|-------|
   | **Name** | `graphiti-mcp` | Used in your deployment URL |
   | **Entrypoint** | `mcp_server/src/graphiti_mcp_server.py:mcp` | Points to module-level server instance |
   | **Authentication** | Enabled | Recommended for production |

   **Important:** The entrypoint must point to a **module-level** `FastMCP` instance.

### Step 4: Configure Environment Variables

Set these environment variables in the FastMCP Cloud UI (**NOT** in `.env` files):

#### For Neo4j:

```bash
# Required
OPENAI_API_KEY=sk-...
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
DATABASE_PROVIDER=neo4j

# Optional
SEMAPHORE_LIMIT=10
GRAPHITI_GROUP_ID=main
```

#### For FalkorDB:

```bash
# Required
OPENAI_API_KEY=sk-...
FALKORDB_URI=redis://host:port
FALKORDB_USER=your-username
FALKORDB_PASSWORD=your-password
FALKORDB_DATABASE=default_db
DATABASE_PROVIDER=falkordb

# Optional
SEMAPHORE_LIMIT=10
GRAPHITI_GROUP_ID=main
```

**Security Note:** Environment variables set in FastMCP Cloud UI are encrypted at rest and never logged.

### Step 5: Deploy

1. **Click "Deploy"**

   FastMCP Cloud will:
   1. Clone your repository
   2. Detect dependencies from `pyproject.toml`
   3. Install dependencies using `uv`
   4. Build your FastMCP server
   5. Deploy to a unique URL
   6. Make it immediately available

2. **Monitor the build**

   Watch the build logs in the FastMCP Cloud UI. The build typically takes 2-5 minutes.

3. **Note your deployment URL**

   Your server will be accessible at:
   ```
   https://your-project-name.fastmcp.app/mcp
   ```

### Step 6: Verify Deployment

1. **Test with `fastmcp inspect`**

   ```bash
   fastmcp inspect https://your-project-name.fastmcp.app/mcp
   ```

   You should see your server info and 9 tools.

2. **Connect from Claude Desktop**

   FastMCP Cloud provides auto-generated configuration. Click "Connect" in the UI and copy the configuration.

3. **Test add_memory tool**

   Use Claude Desktop or an MCP client to test:
   ```
   Add a memory: "John prefers dark mode UI"
   ```

## Configuration Differences

### FastMCP Cloud vs Local Development

| Aspect | FastMCP Cloud | Local Development |
|--------|---------------|-------------------|
| **Entry point** | Module-level instance only | `if __name__ == "__main__"` runs |
| **Dependencies** | Auto-detected from `pyproject.toml` | Installed via `uv sync` |
| **Environment** | Set in Cloud UI | Loaded from `.env` file |
| **Transport** | Managed by platform | Configured via CLI args |
| **HTTPS** | Automatic | Manual setup |
| **Authentication** | Built-in OAuth | Configure manually |

### What Gets Ignored

FastMCP Cloud **ignores**:

- `if __name__ == "__main__"` blocks
- `.env` files (use Cloud UI instead)
- `fastmcp.json` config files (use Cloud UI)
- YAML config files (use environment variables)
- Docker configurations
- CLI arguments

FastMCP Cloud **uses**:

- Module-level `FastMCP` instance (`mcp = FastMCP(...)`)
- `pyproject.toml` or `requirements.txt`
- Environment variables from Cloud UI
- Code from your `main` branch

## Troubleshooting

### Build Failures

**Issue:** Dependencies fail to install

```
Solution:
1. Verify pyproject.toml syntax
2. Check dependency versions are available on PyPI
3. Remove any local-only dependencies (like editable installs)
4. Check that python-dotenv is included
```

**Issue:** Module import errors

```
Solution:
1. Ensure all imports use relative paths from src/
2. Check that config/, models/, etc. have __init__.py files
3. Verify the entrypoint format: mcp_server/src/graphiti_mcp_server.py:mcp
```

### Runtime Errors

**Issue:** "API key is not configured"

```
Solution:
1. Verify environment variables are set in FastMCP Cloud UI
2. Check variable names match exactly (case-sensitive)
3. Redeploy after adding environment variables
```

**Issue:** Database connection failures

```
Solution:
1. Verify database is internet-accessible (not localhost!)
2. Check credentials are correct
3. For Neo4j Aura: Use neo4j+s:// protocol
4. For FalkorDB: Check firewall allows FastMCP Cloud IPs
```

**Issue:** 429 Rate Limit Errors

```
Solution:
1. Lower SEMAPHORE_LIMIT based on your API tier:
   - OpenAI Tier 1: SEMAPHORE_LIMIT=1-2
   - OpenAI Tier 2: SEMAPHORE_LIMIT=5-8
   - OpenAI Tier 3: SEMAPHORE_LIMIT=10-15
```

**Issue:** "Graphiti service not initialized"

```
Solution:
1. This means initialization failed silently
2. Check database credentials
3. Check LLM API key
4. Run fastmcp dev locally to debug
```

## Best Practices

### 1. Use Environment Variables

All configuration should use environment variables:

```python
# Good - FastMCP Cloud compatible
import os
api_key = os.environ.get('OPENAI_API_KEY')

# Bad - Won't work on FastMCP Cloud
api_key = 'sk-hardcoded-key'
```

### 2. Module-Level Server Instance

```python
# Good - FastMCP Cloud can discover this
from fastmcp import FastMCP
mcp = FastMCP("Graphiti Agent Memory")

if __name__ == "__main__":
    # This block is IGNORED by FastMCP Cloud
    mcp.run()
```

### 3. Test Locally First

Always test locally before deploying:

```bash
# Run verification script
cd mcp_server
uv run python scripts/verify_fastmcp_cloud_readiness.py

# Test with fastmcp dev
uv run fastmcp dev src/graphiti_mcp_server.py:mcp
```

### 4. Monitor Resource Usage

- **Neo4j Aura free tier:** Limited connections
- **FalkorDB free tier:** 100 MB limit
- **OpenAI rate limits:** Tier-dependent
- **SEMAPHORE_LIMIT:** Tune based on API tier

## Security Considerations

### Secrets Management

- **DO:** Set secrets in FastMCP Cloud UI
- **DO:** Add `.env` to `.gitignore`
- **DO:** Use `.env.example` for documentation
- **DON'T:** Commit `.env` files
- **DON'T:** Hardcode API keys
- **DON'T:** Store secrets in YAML configs

### Authentication

FastMCP Cloud provides built-in authentication:

- **Enabled:** Only org members can connect (recommended)
- **Disabled:** Public access (use for demos only)

Enable authentication for production deployments.

## Summary Checklist

Before deploying to FastMCP Cloud:

- [ ] Run `uv run python scripts/verify_fastmcp_cloud_readiness.py`
- [ ] All checks pass
- [ ] Cloud database is running (Neo4j Aura or FalkorDB Cloud)
- [ ] API keys are ready (OpenAI required)
- [ ] Code is pushed to GitHub `main` branch
- [ ] `.env` is in `.gitignore`
- [ ] No secrets committed to repo

During deployment:

- [ ] Create project on fastmcp.cloud
- [ ] Configure entrypoint: `mcp_server/src/graphiti_mcp_server.py:mcp`
- [ ] Enable authentication
- [ ] Set all required environment variables in UI
- [ ] Deploy and monitor build logs

After deployment:

- [ ] Test with `fastmcp inspect <URL>`
- [ ] Connect from Claude Desktop
- [ ] Test add_memory and search tools
- [ ] Monitor database usage
- [ ] Monitor API rate limits

## Resources

- **FastMCP Cloud:** [fastmcp.cloud](https://fastmcp.cloud)
- **FastMCP Docs:** [gofastmcp.com](https://gofastmcp.com)
- **FastMCP Discord:** [discord.com/invite/aGsSC3yDF4](https://discord.com/invite/aGsSC3yDF4)
- **Neo4j Aura:** [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura)
- **FalkorDB Cloud:** [cloud.falkordb.com](https://cloud.falkordb.com)
- **Verification Script:** [`scripts/verify_fastmcp_cloud_readiness.py`](../scripts/verify_fastmcp_cloud_readiness.py)

You're now ready to deploy!
