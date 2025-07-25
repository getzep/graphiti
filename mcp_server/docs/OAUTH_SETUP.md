# OAuth Setup for Graphiti MCP Server

This document explains how OAuth authentication is implemented for the Graphiti MCP server when used with Claude Code.

## Overview

Claude Code expects MCP servers using SSE transport to support OAuth 2.1 authentication. The Graphiti MCP server implements this through an OAuth wrapper that provides the required endpoints.

## Architecture

The OAuth implementation consists of two components:

1. **MCP Server** (`graphiti_mcp_server.py`) - Runs on an internal port (default: 8021)
2. **OAuth Wrapper** (`oauth_wrapper.py`) - Runs on the public port (default: 8020) and provides OAuth endpoints

## OAuth Endpoints

The wrapper provides the following OAuth endpoints required by Claude Code:

- `/.well-known/oauth-authorization-server` - OAuth server metadata
- `/.well-known/oauth-protected-resource` - Protected resource metadata
- `/register` - Dynamic client registration
- `/sse` - Proxied SSE endpoint

## Configuration

OAuth settings can be configured via environment variables:

```bash
# OAuth Configuration
OAUTH_CLIENT_ID=graphiti-mcp                    # OAuth client ID
OAUTH_CLIENT_SECRET=your-secret-key-here         # OAuth client secret
OAUTH_ISSUER=http://localhost:8020               # OAuth issuer URL
OAUTH_AUDIENCE=graphiti-mcp                      # OAuth audience

# Server Ports
MCP_SERVER_PORT=8020                             # Public port for OAuth wrapper
MCP_INTERNAL_PORT=8021                           # Internal port for MCP server
```

## Running with OAuth

### Option 1: Using the Shell Script

```bash
cd mcp_server
./run_with_oauth.sh
```

### Option 2: Using Docker Compose

```bash
cd mcp_server
docker-compose up
```

### Option 3: Manual Setup

```bash
# Terminal 1: Start MCP server on internal port
python src/graphiti_mcp_server.py --transport sse --port $MCP_INTERNAL_PORT

# Terminal 2: Start OAuth wrapper on public port
python src/oauth_wrapper.py
```

## How It Works

1. Claude Code connects to the OAuth wrapper on port 8020
2. The wrapper handles OAuth discovery and registration requests
3. SSE requests are proxied to the actual MCP server on port 8021
4. The MCP server processes the actual tool calls and responses

## Testing OAuth Endpoints

You can test the OAuth endpoints manually:

```bash
# Check OAuth server metadata
curl http://localhost:8020/.well-known/oauth-authorization-server

# Check protected resource metadata
curl http://localhost:8020/.well-known/oauth-protected-resource

# Register a client
curl -X POST http://localhost:8020/register \
  -H "Content-Type: application/json" \
  -d '{"client_name": "Test Client"}'
```

## Security Notes

- The default `OAUTH_CLIENT_SECRET` should be changed in production
- In production, consider using a proper database for storing registered clients
- The current implementation stores clients in memory, which is lost on restart
- Consider implementing proper token validation and refresh token support for production use
