# ChatGPT Integration Guide for Graphiti Memory Server

This guide explains how to integrate your deployed Graphiti MCP server with ChatGPT for persistent memory capabilities.

## Deployment URL Format

After Railway deployment, your server will be available at:
- **Base URL**: `https://graphiti-production-xxxx.up.railway.app`
- **MCP SSE Endpoint**: `https://graphiti-production-xxxx.up.railway.app/sse`

## ChatGPT Integration Methods

### Method 1: ChatGPT with Native MCP Support

If your ChatGPT client supports MCP natively:

```json
{
  "mcpServers": {
    "graphiti-memory": {
      "transport": "sse",
      "url": "https://your-railway-domain.up.railway.app/sse",
      "timeout": 30000
    }
  }
}
```

### Method 2: Custom ChatGPT Integration

For custom implementations, use HTTP requests to the SSE endpoint with proper MCP protocol formatting.

## Available Memory Tools

Your ChatGPT will have access to these memory functions:

### 1. Add Memory (`add_memory`)
Store information in the knowledge graph:
```json
{
  "name": "meeting_notes_2024",
  "episode_body": "Discussed project timeline and deliverables with team",
  "source": "text",
  "group_id": "work_meetings"
}
```

### 2. Search Memory Nodes (`search_memory_nodes`)
Find entities and concepts:
```json
{
  "query": "project timeline",
  "max_nodes": 10,
  "group_ids": ["work_meetings"]
}
```

### 3. Search Memory Facts (`search_memory_facts`)
Find relationships between entities:
```json
{
  "query": "project deliverables",
  "max_facts": 10,
  "group_ids": ["work_meetings"]
}
```

### 4. Get Recent Episodes (`get_episodes`)
Retrieve recent memories:
```json
{
  "group_id": "work_meetings",
  "last_n": 10
}
```

## Security and API Keys

Your deployment uses these credentials (set in Railway environment):

```bash
# Your OpenAI API key (keep secure in Railway environment variables)
OPENAI_API_KEY=sk-proj-your-openai-api-key-here

# Model configuration
MODEL_NAME=gpt-4.1-mini
SMALL_MODEL_NAME=gpt-4.1-nano

# Database (use Neo4j Aura for production)
NEO4J_URI=neo4j+s://your-aura-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-aura-password
```

## Data Persistence and Safety

### Memory Persistence
- **Knowledge Graph**: All memories are stored in Neo4j as a connected graph
- **Temporal Awareness**: Memories include timestamps and can track changes over time
- **Relationships**: Entities are automatically linked based on content

### Data Safety
- **Secure Transport**: All communication uses HTTPS
- **Environment Variables**: Sensitive data stored securely in Railway
- **Data Isolation**: Use `group_id` to organize different contexts
- **Backup**: Consider regular Neo4j database backups

## Usage Examples

### Personal Assistant Memory
```json
{
  "name": "user_preferences",
  "episode_body": "User prefers meetings scheduled in the morning, dislikes late-night calls, works in PST timezone",
  "source": "text",
  "group_id": "user_profile"
}
```

### Project Context
```json
{
  "name": "project_requirements",
  "episode_body": "Railway deployment needs Docker optimization, MCP SSE transport, environment variable configuration",
  "source": "text", 
  "group_id": "railway_project"
}
```

### Conversation Memory
```json
{
  "name": "conversation_context",
  "episode_body": "User asked about deploying Graphiti to Railway for ChatGPT integration. Needs MCP server with SSE transport.",
  "source": "text",
  "group_id": "current_conversation"
}
```

## Testing Your Integration

### 1. Basic Connectivity Test
Verify the SSE endpoint is accessible:
```bash
curl -H "Accept: text/event-stream" https://your-railway-domain.up.railway.app/sse
```

### 2. MCP Inspector Test
Test with official MCP tools:
```bash
npx @modelcontextprotocol/inspector --url https://your-railway-domain.up.railway.app/sse
```

Expected output should show:
- ✅ Connected to MCP server
- ✅ Tools available: add_memory, search_memory_nodes, search_memory_facts, etc.
- ✅ Server status: healthy

### 3. Memory Operations Test

**Step 1: Add test memory**
```json
{
  "tool": "add_memory",
  "arguments": {
    "name": "connection_test",
    "episode_body": "Testing ChatGPT integration with Graphiti memory server",
    "source": "text",
    "group_id": "integration_test"
  }
}
```

**Step 2: Search for the memory**
```json
{
  "tool": "search_memory_nodes", 
  "arguments": {
    "query": "ChatGPT integration test",
    "max_nodes": 5,
    "group_ids": ["integration_test"]
  }
}
```

**Step 3: Verify persistence**
The search should return the memory you just added, confirming the integration works.

## Troubleshooting

### Common Issues

**Connection Timeouts:**
- Check Railway deployment status
- Verify environment variables are set
- Test endpoint directly with curl

**Memory Not Persisting:**
- Verify Neo4j database connection
- Check Railway logs for database errors
- Ensure group_id consistency

**API Rate Limits:**
- Monitor OpenAI API usage
- Adjust SEMAPHORE_LIMIT if needed
- Check for 429 errors in logs

### Debug Steps

1. **Check Railway Logs**:
   - Look for server startup messages
   - Verify environment variable loading
   - Check for database connection confirmations

2. **Test Database Connection**:
   - Verify Neo4j credentials
   - Test connection from Railway environment
   - Check Neo4j Aura instance status

3. **Validate MCP Protocol**:
   - Use MCP Inspector for protocol validation
   - Check SSE stream format
   - Verify tool availability

## Performance Considerations

- **Concurrent Operations**: Controlled by `SEMAPHORE_LIMIT` (default: 10)
- **Memory Efficiency**: Use specific `group_id` values to organize data
- **Search Optimization**: Limit result sets with `max_nodes` and `max_facts`
- **Rate Limiting**: Monitor OpenAI API usage to avoid 429 errors

## Next Steps

1. Deploy to Railway with proper environment variables
2. Configure ChatGPT with the SSE endpoint
3. Test memory operations with simple examples
4. Implement in your ChatGPT workflow
5. Monitor performance and adjust configuration as needed

Your ChatGPT will now have persistent memory capabilities powered by Graphiti's knowledge graph!