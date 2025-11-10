# Graphiti MCP + LibreChat Multi-User Setup on Unraid (stdio Mode)

Complete guide for running Graphiti MCP Server with LibreChat on Unraid using **stdio mode** for true per-user isolation with your existing Neo4j database.

> **📦 Package:** This guide uses `graphiti-mcp-varming` - an enhanced fork of Graphiti MCP with additional tools for advanced knowledge management. Available on [PyPI](https://pypi.org/project/graphiti-mcp-varming/) and [GitHub](https://github.com/Varming73/graphiti).

## ✅ Multi-User Isolation: FULLY SUPPORTED

This guide implements **true per-user graph isolation** using LibreChat's `{{LIBRECHAT_USER_ID}}` placeholder with stdio transport.

### How It Works

- ✅ **LibreChat spawns Graphiti MCP process per user session**
- ✅ **Each process gets unique `GRAPHITI_GROUP_ID`** from `{{LIBRECHAT_USER_ID}}`
- ✅ **Complete data isolation** - Users cannot see each other's knowledge
- ✅ **Automatic and transparent** - No manual configuration needed per user
- ✅ **Scalable** - Works for unlimited users

### What You Get

- **Per-user isolation**: Each user's knowledge graph is completely separate
- **Existing Neo4j**: Connects to your running Neo4j on Unraid
- **Your custom enhancements**: Enhanced tools from your fork
- **Shared infrastructure**: One Neo4j, one LibreChat, automatic isolation

## Architecture

```
LibreChat Container
    ↓ (spawns per-user process via stdio)
Graphiti MCP Process (User A: group_id=librechat_user_abc_123)
Graphiti MCP Process (User B: group_id=librechat_user_xyz_789)
    ↓ (both connect to)
Your Neo4j Container (bolt://neo4j:7687)
    └── User A's graph (group_id: librechat_user_abc_123)
    └── User B's graph (group_id: librechat_user_xyz_789)
```

---

## Prerequisites

✅ LibreChat running in Docker on Unraid  
✅ Neo4j running in Docker on Unraid  
✅ OpenAI API key (or other supported LLM provider)  
✅ `uv` package manager available in LibreChat container (or alternative - see below)

---

## Step 1: Prepare LibreChat Container

LibreChat needs to spawn Graphiti MCP processes, which requires having the MCP server available.

### Option A: Install `uv` in LibreChat Container (Recommended - Simplest)

`uv` is the modern Python package/tool runner used by Graphiti. It will automatically download and manage the Graphiti MCP package.

```bash
# Enter LibreChat container
docker exec -it librechat bash

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (add this to ~/.bashrc for persistence)
export PATH="$HOME/.local/bin:$PATH"

# Verify installation
uvx --version
```

**That's it!** No need to pre-install Graphiti MCP - `uvx` will handle it automatically when LibreChat spawns processes.

### Option B: Pre-install Graphiti MCP Package (Alternative)

If you prefer to pre-install the package:

```bash
docker exec -it librechat bash
pip install graphiti-mcp-varming
```

Then use `python -m graphiti_mcp_server` as the command instead of `uvx`.

---

## Step 2: Verify Neo4j Network Access

The Graphiti MCP processes spawned by LibreChat need to reach your Neo4j container.

### Check Network Configuration

```bash
# Check if containers can communicate
docker exec librechat ping -c 3 neo4j

# If that fails, find Neo4j IP
docker inspect neo4j | grep IPAddress
```

### Network Options

**Option A: Same Docker Network (Recommended)**
- Put LibreChat and Neo4j on the same Docker network
- Use container name: `bolt://neo4j:7687`

**Option B: Host IP**
- Use Unraid host IP: `bolt://192.168.1.XXX:7687`
- Works across different networks

**Option C: Container IP**
- Use Neo4j's container IP from docker inspect
- Less reliable (IP may change on restart)

---

## Step 3: Configure LibreChat MCP Integration

### 3.1 Locate LibreChat Configuration

Find your LibreChat `librechat.yaml` configuration file. On Unraid, typically:
- `/mnt/user/appdata/librechat/librechat.yaml`

### 3.2 Add Graphiti MCP Configuration

Add this to your `librechat.yaml` under the `mcpServers` section:

```yaml
mcpServers:
  graphiti:
    type: stdio
    command: uvx
    args:
      - graphiti-mcp-varming
    env:
      # Multi-user isolation - THIS IS THE MAGIC! ✨
      GRAPHITI_GROUP_ID: "{{LIBRECHAT_USER_ID}}"
      
      # Neo4j connection - adjust based on your network setup
      NEO4J_URI: "bolt://neo4j:7687"
      # Or use host IP if containers on different networks:
      # NEO4J_URI: "bolt://192.168.1.XXX:7687"
      
      NEO4J_USER: "neo4j"
      NEO4J_PASSWORD: "your_neo4j_password"
      NEO4J_DATABASE: "neo4j"
      
      # LLM Configuration
      OPENAI_API_KEY: "${OPENAI_API_KEY}"
      # Or hardcode: OPENAI_API_KEY: "sk-your-key-here"
      
      # Optional: LLM model selection
      # MODEL_NAME: "gpt-4o"
      
      # Optional: Adjust concurrency based on your OpenAI tier
      # SEMAPHORE_LIMIT: "10"
      
      # Optional: Disable telemetry
      # GRAPHITI_TELEMETRY_ENABLED: "false"
    
    timeout: 60000        # 60 seconds for long operations
    initTimeout: 15000    # 15 seconds to initialize
    
    serverInstructions: true  # Use Graphiti's built-in instructions
    
    # Optional: Show in chat menu dropdown
    chatMenu: true
```

### 3.3 Key Configuration Notes

**The Magic Line:**
```yaml
GRAPHITI_GROUP_ID: "{{LIBRECHAT_USER_ID}}"
```

- LibreChat **replaces `{{LIBRECHAT_USER_ID}}`** with actual user ID at runtime
- Each user session gets a **unique environment variable**
- Graphiti MCP process reads this and uses it as the graph namespace
- **Result**: Complete per-user isolation automatically!

**Command Options:**

**Option A (Recommended):** Using `uvx` - automatically downloads from PyPI:
```yaml
command: uvx
args:
  - graphiti-mcp-varming
```

**Option B:** If you pre-installed the package with pip:
```yaml
command: python
args:
  - -m
  - graphiti_mcp_server
```

**Option C:** With FalkorDB support (if you need FalkorDB instead of Neo4j):
```yaml
command: uvx
args:
  - --with
  - graphiti-mcp-varming[falkordb]
  - graphiti-mcp-varming
env:
  # Use FalkorDB connection instead
  DATABASE_PROVIDER: "falkordb"
  REDIS_URI: "redis://falkordb:6379"
  # ... rest of config
```

**Option D:** With all LLM providers (Anthropic, Groq, Voyage, etc.):
```yaml
command: uvx
args:
  - --with
  - graphiti-mcp-varming[all]
  - graphiti-mcp-varming
```

### 3.4 Environment Variable Options

**Using LibreChat's .env file:**
```yaml
env:
  OPENAI_API_KEY: "${OPENAI_API_KEY}"  # Reads from LibreChat's .env
```

**Hardcoding (less secure):**
```yaml
env:
  OPENAI_API_KEY: "sk-your-actual-key-here"
```

**Per-user API keys (advanced):**
See the Advanced Configuration section for customUserVars setup.

---

## Step 4: Restart LibreChat

After updating the configuration:

```bash
# In Unraid terminal or SSH
docker restart librechat
```

Or use the Unraid Docker UI to restart the LibreChat container.

---

## Step 5: Verify Installation

### 5.1 Check LibreChat Logs

```bash
docker logs -f librechat
```

Look for:
- MCP server initialization messages
- No errors about missing `uvx` or connection issues

### 5.2 Test in LibreChat

1. **Log into LibreChat** as User A
2. **Start a new chat**
3. **Look for Graphiti tools** in the tool selection menu
4. **Test adding knowledge:**
   ```
   Add this to my knowledge: I prefer Python over JavaScript for backend development
   ```

5. **Verify it was stored:**
   ```
   What do you know about my programming preferences?
   ```

### 5.3 Verify Per-User Isolation

**Critical Test:**

1. **Log in as User A** (e.g., `alice@example.com`)
   - Add knowledge: "I love dark mode and use VS Code"

2. **Log in as User B** (e.g., `bob@example.com`) 
   - Try to query: "What editor preferences do you know about?"
   - Should return: **No information** (or only Bob's own data)

3. **Log back in as User A**
   - Query again: "What editor preferences do you know about?"
   - Should return: **Dark mode and VS Code** (Alice's data)

**Expected Result:** ✅ Complete isolation - users cannot see each other's knowledge!

### 5.4 Check Neo4j (Optional)

```bash
# Connect to Neo4j browser: http://your-unraid-ip:7474

# Run this Cypher query to see isolation in action:
MATCH (n)
RETURN DISTINCT n.group_id, count(n) as node_count
ORDER BY n.group_id
```

You should see different `group_id` values for different users!

---

## How It Works: The Technical Details

### The Flow

```
User "Alice" logs into LibreChat
    ↓
LibreChat replaces: GRAPHITI_GROUP_ID: "{{LIBRECHAT_USER_ID}}"
    ↓
Becomes: GRAPHITI_GROUP_ID: "librechat_user_alice_12345"
    ↓
LibreChat spawns: uvx --from graphiti-mcp graphiti-mcp
    ↓
Process receives environment: GRAPHITI_GROUP_ID=librechat_user_alice_12345
    ↓
Graphiti loads config: group_id: ${GRAPHITI_GROUP_ID:main}
    ↓
Config gets: config.graphiti.group_id = "librechat_user_alice_12345"
    ↓
All tools use this group_id for Neo4j queries
    ↓
Alice's nodes in Neo4j: { group_id: "librechat_user_alice_12345", ... }
    ↓
Bob's nodes in Neo4j: { group_id: "librechat_user_bob_67890", ... }
    ↓
Complete isolation achieved! ✅
```

### Tools with Per-User Isolation

These 7 tools automatically use the user's `group_id`:

1. **add_memory** - Store knowledge in user's graph
2. **search_nodes** - Search only user's entities
3. **get_entities_by_type** - Browse user's entities by type (your custom tool!)
4. **search_memory_facts** - Search user's relationships/facts
5. **compare_facts_over_time** - Track user's knowledge evolution (your custom tool!)
6. **get_episodes** - Retrieve user's conversation history
7. **clear_graph** - Clear only user's graph data

### Security Model

- ✅ **Users see only their data** - No cross-contamination
- ✅ **UUID-based operations are safe** - Users only know UUIDs from their own queries
- ✅ **No admin action needed** - Automatic per-user isolation
- ✅ **Scalable** - Unlimited users without configuration changes

---

## Troubleshooting

### uvx Command Not Found

**Problem:** LibreChat logs show `uvx: command not found`

**Solutions:**

1. **Install uv in LibreChat container:**
   ```bash
   docker exec -it librechat bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   uvx --version
   ```

2. **Test uvx can fetch the package:**
   ```bash
   docker exec -it librechat uvx graphiti-mcp-varming --help
   ```

3. **Use alternative command (python with pre-install):**
   ```bash
   docker exec -it librechat pip install graphiti-mcp-varming
   ```
   
   Then update config:
   ```yaml
   command: python
   args:
     - -m
     - graphiti_mcp_server
   ```

### Package Installation Fails

**Problem:** `uvx` fails to download `graphiti-mcp-varming`

**Solutions:**

1. **Check internet connectivity from container:**
   ```bash
   docker exec -it librechat ping -c 3 pypi.org
   ```

2. **Manually test installation:**
   ```bash
   docker exec -it librechat uvx graphiti-mcp-varming --help
   ```

3. **Check for proxy/firewall issues** blocking PyPI access

4. **Use pre-installation method instead** (Option B from Step 1)

### Container Can't Connect to Neo4j

**Problem:** `Connection refused to bolt://neo4j:7687`

**Solutions:**

1. **Check Neo4j is running:**
   ```bash
   docker ps | grep neo4j
   ```

2. **Verify network connectivity:**
   ```bash
   docker exec librechat ping -c 3 neo4j
   ```

3. **Use host IP instead:**
   ```yaml
   env:
     NEO4J_URI: "bolt://192.168.1.XXX:7687"
   ```

4. **Check Neo4j is listening on correct port:**
   ```bash
   docker logs neo4j | grep "Bolt enabled"
   ```

### MCP Tools Not Showing Up

**Problem:** Graphiti tools don't appear in LibreChat

**Solutions:**

1. **Check LibreChat logs:**
   ```bash
   docker logs librechat | grep -i mcp
   docker logs librechat | grep -i graphiti
   ```

2. **Verify config syntax:**
   - YAML is whitespace-sensitive!
   - Ensure proper indentation
   - Check for typos in command/args

3. **Test manual spawn:**
   ```bash
   docker exec librechat uvx --from graphiti-mcp graphiti-mcp --help
   ```

4. **Check environment variables are set:**
   ```bash
   docker exec librechat env | grep -i openai
   docker exec librechat env | grep -i neo4j
   ```

### Users Can See Each Other's Data

**Problem:** Isolation not working

**Check:**

1. **Verify placeholder syntax:**
   ```yaml
   GRAPHITI_GROUP_ID: "{{LIBRECHAT_USER_ID}}"  # Must be EXACTLY this
   ```

2. **Check LibreChat version:**
   - Placeholder support added in recent versions
   - Update LibreChat if necessary

3. **Inspect Neo4j data:**
   ```cypher
   MATCH (n)
   RETURN DISTINCT n.group_id, labels(n), count(n)
   ```
   Should show different group_ids for different users

4. **Check logs for actual group_id:**
   ```bash
   docker logs librechat | grep GRAPHITI_GROUP_ID
   ```

### OpenAI Rate Limits (429 Errors)

**Problem:** `429 Too Many Requests` errors

**Solution:** Reduce concurrent processing:

```yaml
env:
  SEMAPHORE_LIMIT: "3"  # Lower for free tier
```

**By OpenAI Tier:**
- Free tier: `SEMAPHORE_LIMIT: "1"`
- Tier 1: `SEMAPHORE_LIMIT: "3"`
- Tier 2: `SEMAPHORE_LIMIT: "8"`
- Tier 3+: `SEMAPHORE_LIMIT: "15"`

### Process Spawn Failures

**Problem:** LibreChat can't spawn MCP processes

**Check:**

1. **LibreChat has execution permissions**
2. **Enough system resources** (check RAM/CPU)
3. **Docker has sufficient memory allocated**
4. **No process limit restrictions**

---

## Advanced Configuration

### Your Custom Enhanced Tools

Your custom Graphiti MCP fork (`graphiti-mcp-varming`) includes additional tools beyond the official release:

- **`get_entities_by_type`** - Browse all entities of a specific type
- **`compare_facts_over_time`** - Track how knowledge evolves over time
- Additional functionality for advanced knowledge management

These automatically work with per-user isolation and will appear in LibreChat's tool selection!

**Package Details:**
- **PyPI**: `graphiti-mcp-varming`
- **GitHub**: https://github.com/Varming73/graphiti
- **Base**: Built on official `graphiti-core` from Zep AI

### Using Different LLM Providers

#### Anthropic (Claude)

```yaml
env:
  ANTHROPIC_API_KEY: "${ANTHROPIC_API_KEY}"
  LLM_PROVIDER: "anthropic"
  MODEL_NAME: "claude-3-5-sonnet-20241022"
```

#### Azure OpenAI

```yaml
env:
  AZURE_OPENAI_API_KEY: "${AZURE_OPENAI_API_KEY}"
  AZURE_OPENAI_ENDPOINT: "https://your-resource.openai.azure.com/"
  AZURE_OPENAI_DEPLOYMENT: "your-gpt4-deployment"
  LLM_PROVIDER: "azure_openai"
```

#### Groq

```yaml
env:
  GROQ_API_KEY: "${GROQ_API_KEY}"
  LLM_PROVIDER: "groq"
  MODEL_NAME: "mixtral-8x7b-32768"
```

#### Local Ollama

```yaml
env:
  LLM_PROVIDER: "openai"  # Ollama is OpenAI-compatible
  MODEL_NAME: "llama3"
  OPENAI_API_BASE: "http://host.docker.internal:11434/v1"
  OPENAI_API_KEY: "ollama"  # Dummy key
  EMBEDDER_PROVIDER: "sentence_transformers"
  EMBEDDER_MODEL: "all-MiniLM-L6-v2"
```

### Per-User API Keys (Advanced)

Allow users to provide their own OpenAI keys using LibreChat's customUserVars:

```yaml
mcpServers:
  graphiti:
    command: uvx
    args:
      - --from
      - graphiti-mcp
      - graphiti-mcp
    env:
      GRAPHITI_GROUP_ID: "{{LIBRECHAT_USER_ID}}"
      OPENAI_API_KEY: "{{USER_OPENAI_KEY}}"  # User-provided
      NEO4J_URI: "bolt://neo4j:7687"
      NEO4J_PASSWORD: "${NEO4J_PASSWORD}"
    customUserVars:
      USER_OPENAI_KEY:
        title: "Your OpenAI API Key"
        description: "Enter your personal OpenAI API key from <a href='https://platform.openai.com/api-keys' target='_blank'>OpenAI Platform</a>"
```

Users will be prompted to enter their API key in the LibreChat UI settings.

---

## Performance Optimization

### 1. Adjust Concurrency

Higher = faster processing, but more API calls:

```yaml
env:
  SEMAPHORE_LIMIT: "15"  # For Tier 3+ OpenAI accounts
```

### 2. Use Faster Models

For development/testing:

```yaml
env:
  MODEL_NAME: "gpt-4o-mini"  # Faster and cheaper
```

### 3. Neo4j Performance

For large graphs with many users, increase Neo4j memory:

```bash
# Edit Neo4j docker config:
NEO4J_server_memory_heap_max__size=2G
NEO4J_server_memory_pagecache_size=1G
```

### 4. Enable Neo4j Indexes

Connect to Neo4j browser (http://your-unraid-ip:7474) and run:

```cypher
// Index on group_id for faster user isolation queries
CREATE INDEX group_id_idx IF NOT EXISTS FOR (n) ON (n.group_id);

// Index on UUIDs
CREATE INDEX uuid_idx IF NOT EXISTS FOR (n) ON (n.uuid);

// Index on entity names
CREATE INDEX name_idx IF NOT EXISTS FOR (n) ON (n.name);
```

---

## Data Management

### Backup Neo4j Data (Includes All User Graphs)

```bash
# Stop Neo4j
docker stop neo4j

# Backup data volume
docker run --rm \
  -v neo4j_data:/data \
  -v /mnt/user/backups:/backup \
  alpine tar czf /backup/neo4j-backup-$(date +%Y%m%d).tar.gz -C /data .

# Restart Neo4j
docker start neo4j
```

### Restore Neo4j Data

```bash
# Stop Neo4j
docker stop neo4j

# Restore data volume
docker run --rm \
  -v neo4j_data:/data \
  -v /mnt/user/backups:/backup \
  alpine tar xzf /backup/neo4j-backup-YYYYMMDD.tar.gz -C /data

# Restart Neo4j
docker start neo4j
```

### Per-User Data Export

Export a specific user's graph:

```cypher
// In Neo4j browser
MATCH (n {group_id: "librechat_user_alice_12345"})
OPTIONAL MATCH (n)-[r]->(m {group_id: "librechat_user_alice_12345"})
RETURN n, r, m
```

---

## Security Considerations

1. **Use strong Neo4j passwords** in production
2. **Secure OpenAI API keys** - use environment variables, not hardcoded
3. **Network isolation** - consider using dedicated Docker networks
4. **Regular backups** - Automate Neo4j backups
5. **Monitor resource usage** - Set appropriate limits
6. **Update regularly** - Keep all containers updated for security patches

---

## Monitoring

### Check Process Activity

```bash
# View active Graphiti MCP processes (when users are active)
docker exec librechat ps aux | grep graphiti

# Monitor LibreChat logs
docker logs -f librechat | grep -i graphiti

# Neo4j query performance
docker logs neo4j | grep "slow query"
```

### Monitor Resource Usage

```bash
# Real-time stats
docker stats librechat neo4j

# Check Neo4j memory usage
docker exec neo4j bin/neo4j-admin server memory-recommendation
```

---

## Upgrading

### Update Graphiti MCP

**Method 1: Automatic (uvx - Recommended)**

Since LibreChat spawns processes via uvx, it automatically gets the latest version from PyPI on first run. To force an update:

```bash
# Enter LibreChat container and clear cache
docker exec -it librechat bash
rm -rf ~/.cache/uv
```

Next time LibreChat spawns a process, it will download the latest version.

**Method 2: Pre-installed Package**

If you pre-installed via pip:

```bash
docker exec -it librechat pip install --upgrade graphiti-mcp-varming
```

**Check Current Version:**

```bash
docker exec -it librechat uvx graphiti-mcp-varming --version
```

### Update Neo4j

Follow Neo4j's official upgrade guide. Always backup first!

---

## Additional Resources

- **Package**: [graphiti-mcp-varming on PyPI](https://pypi.org/project/graphiti-mcp-varming/)
- **Source Code**: [Varming's Enhanced Fork](https://github.com/Varming73/graphiti)
- [Graphiti MCP Server Documentation](../mcp_server/README.md)
- [LibreChat MCP Documentation](https://www.librechat.ai/docs/features/mcp)
- [Neo4j Operations Manual](https://neo4j.com/docs/operations-manual/current/)
- [Official Graphiti Core](https://github.com/getzep/graphiti) (by Zep AI)
- [Verification Test](./.serena/memories/librechat_integration_verification.md)

---

## Example Usage in LibreChat

Once configured, you can use Graphiti in your LibreChat conversations:

**Adding Knowledge:**
> "Remember that I prefer dark mode and use Python for backend development"

**Querying Knowledge:**
> "What do you know about my programming preferences?"

**Complex Queries:**
> "Show me all the projects I've mentioned that use Python"

**Updating Knowledge:**
> "I no longer use Python exclusively, I now also use Go"

**Using Custom Tools:**
> "Compare how my technology preferences have changed over time"

The knowledge graph will automatically track entities, relationships, and temporal information - all isolated per user!

---

**Last Updated:** November 9, 2025  
**Graphiti Version:** 0.22.0+  
**MCP Server Version:** 1.0.0+  
**Mode:** stdio (per-user process spawning)  
**Multi-User:** ✅ Fully Supported via `{{LIBRECHAT_USER_ID}}`
