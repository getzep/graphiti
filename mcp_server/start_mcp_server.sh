#!/bin/bash
# Start Graphiti MCP Server

# Set OpenAI API key (replace with your actual key)
export OPENAI_API_KEY="your-openai-api-key-here"

# Start the server
cd ~/graphiti/mcp_server
uv run src/graphiti_mcp_server/server.py
