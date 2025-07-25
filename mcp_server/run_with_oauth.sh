#!/bin/bash
# Run Graphiti MCP server with OAuth support

# Set the internal MCP port (different from the public port)
export MCP_SERVER_PORT=${MCP_SERVER_PORT:-8020}

export MCP_INTERNAL_PORT=${MCP_INTERNAL_PORT:-$((MCP_SERVER_PORT + 1))}

# Start the MCP server in the background on the internal port
echo "Starting MCP server on internal port $MCP_INTERNAL_PORT..."
python src/graphiti_mcp_server.py --port $MCP_INTERNAL_PORT &
MCP_PID=$!

# Give the MCP server time to start
sleep 3

# Start the OAuth wrapper on the public port
echo "Starting OAuth wrapper on public port $MCP_SERVER_PORT..."
python oauth_wrapper.py

# Clean up - kill the MCP server when the wrapper exits
kill $MCP_PID 2>/dev/null
