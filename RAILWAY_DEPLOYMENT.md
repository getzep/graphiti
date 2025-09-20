# Railway Deployment Guide for Graphiti MCP Server

This guide helps you deploy the Graphiti MCP Server to Railway for use with ChatGPT and other MCP clients.

## Prerequisites

- Railway account connected to GitHub
- OpenAI API key
- Neo4j database (local, Neo4j Aura, or Railway Neo4j service)

## Railway Deployment Steps

### 1. Environment Variables

Set these environment variables in your Railway project:

**Required:**
- `OPENAI_API_KEY`: Your OpenAI API key (starts with `sk-proj-` or `sk-`)
- `MODEL_NAME`: `gpt-4.1-mini` (recommended)
- `SMALL_MODEL_NAME`: `gpt-4.1-nano` (recommended)

**Database Configuration (choose one option):**

**Option A: Local Neo4j (for development/testing)**
- `NEO4J_URI`: `bolt://localhost:7687`
- `NEO4J_USER`: `neo4j`
- `NEO4J_PASSWORD`: `password`

**Option B: Neo4j Aura Cloud (recommended for production)**
- `NEO4J_URI`: `neo4j+s://your-instance.databases.neo4j.io`
- `NEO4J_USER`: `neo4j`
- `NEO4J_PASSWORD`: Your Aura password

**Option C: Railway Neo4j Service (if available)**
- Use Railway's internal connection variables

**Optional:**
- `LLM_TEMPERATURE`: `0.0` (default)
- `SEMAPHORE_LIMIT`: `10` (concurrent operations limit)
- `GRAPHITI_TELEMETRY_ENABLED`: `false` (to disable telemetry)

### 2. Deploy to Railway

1. **Connect Repository**: Link your GitHub repository to Railway
2. **Service Configuration**: Railway will auto-detect the Dockerfile
3. **Environment Variables**: Set all required variables in Railway dashboard
4. **Deploy**: Railway will build and deploy automatically

### 3. Get Your Deployment URL

After successful deployment, Railway will provide a URL like:
`https://graphiti-production-xxxx.up.railway.app`

Your MCP SSE endpoint will be:
`https://graphiti-production-xxxx.up.railway.app/sse`

## ChatGPT Integration

### For ChatGPT with MCP Support

Configure your ChatGPT client with:

```json
{
  "mcpServers": {
    "graphiti-memory": {
      "transport": "sse",
      "url": "https://your-railway-domain.up.railway.app/sse"
    }
  }
}
```

### Custom ChatGPT Integration

If using a custom ChatGPT integration, make HTTP requests to:
- **SSE Endpoint**: `https://your-railway-domain.up.railway.app/sse`
- **Tools Available**: `add_memory`, `search_memory_nodes`, `search_memory_facts`, etc.

## Claude Desktop Integration

For Claude Desktop (requires mcp-remote bridge):

```json
{
  "mcpServers": {
    "graphiti-memory": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://your-railway-domain.up.railway.app/sse"
      ]
    }
  }
}
```

## Testing Your Deployment

### 1. Verify Server Status

Visit: `https://your-railway-domain.up.railway.app/sse`

You should see an SSE connection established.

### 2. Test with MCP Inspector

```bash
npx @modelcontextprotocol/inspector --url https://your-railway-domain.up.railway.app/sse
```

Expected tools:
- `add_memory`
- `search_memory_nodes` 
- `search_memory_facts`
- `delete_entity_edge`
- `delete_episode`
- `get_entity_edge`
- `get_episodes`
- `clear_graph`

### 3. Test Memory Operations

**Add Memory:**
```json
{
  "name": "test_memory",
  "episode_body": "This is a test memory for verification",
  "source": "text"
}
```

**Search Memory:**
```json
{
  "query": "test memory",
  "max_nodes": 5
}
```

## Database Options

### Neo4j Aura Cloud (Recommended)

1. Create a free Neo4j Aura instance at https://neo4j.com/aura/
2. Note the connection URI (starts with `neo4j+s://`)
3. Set environment variables in Railway:
   - `NEO4J_URI`: Your Aura connection string
   - `NEO4J_USER`: `neo4j`
   - `NEO4J_PASSWORD`: Your Aura password

### Local Development Database

For testing with local Neo4j:
- Ensure your local Neo4j is accessible from Railway
- Consider using ngrok or similar for temporary access
- Not recommended for production

## Security Considerations

1. **API Keys**: Never commit API keys to git. Use Railway environment variables.
2. **Database Security**: Use strong passwords and secure connection strings.
3. **Access Control**: Consider implementing authentication if needed.
4. **HTTPS**: Railway provides HTTPS by default.

## Troubleshooting

### Common Issues

**Build Failures:**
- Check Docker cache mount compatibility
- Ensure all dependencies are properly specified
- Review Railway build logs

**Connection Issues:**
- Verify environment variables are set correctly
- Check Neo4j database accessibility
- Ensure OpenAI API key is valid

**Memory/Performance:**
- Adjust `SEMAPHORE_LIMIT` for rate limiting
- Monitor Railway resource usage
- Consider Neo4j Aura for better performance

### Debug Commands

Check server logs in Railway dashboard for:
- Connection status messages
- Environment variable loading
- Database connection status
- MCP server initialization

## Support

For issues:
1. Check Railway deployment logs
2. Verify environment variables
3. Test database connectivity
4. Review MCP client configuration

For Graphiti-specific issues, see the main repository documentation.