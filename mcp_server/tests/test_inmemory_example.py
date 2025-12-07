#!/usr/bin/env python3
"""
In-memory test example for Graphiti MCP Server using FastMCP's recommended pattern.

This demonstrates the proper way to test FastMCP servers without spawning subprocesses.
Instead of using StdioServerParameters, we use the FastMCP Client with in-memory transport.

Requirements:
- pytest-asyncio
- Running FalkorDB instance (docker compose -f docker/docker-compose.yml up -d)
- OPENAI_API_KEY in environment or .env file

Usage:
    cd mcp_server
    FALKORDB_URI=redis://localhost:6379 SEMAPHORE_LIMIT=10 uv run pytest tests/test_inmemory_example.py -v
"""

import os
import sys
from pathlib import Path

import pytest

# Add src directory to Python path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Set required environment variables BEFORE importing graphiti modules
# This prevents the ValueError: invalid literal for int() with base 10: '' error
# Note: We must set these unconditionally if they're empty strings (not just unset)
def set_env_if_empty(key: str, value: str):
    """Set env var if unset OR if set to empty string."""
    if not os.environ.get(key):
        os.environ[key] = value

set_env_if_empty('SEMAPHORE_LIMIT', '10')
set_env_if_empty('MAX_REFLEXION_ITERATIONS', '0')
set_env_if_empty('USE_PARALLEL_RUNTIME', 'false')
set_env_if_empty('FALKORDB_URI', 'redis://localhost:6379')
set_env_if_empty('OPENAI_API_KEY', 'test_key_for_testing')

from fastmcp.client import Client

# Import the MCP server instance after setting env vars
from graphiti_mcp_server import mcp


@pytest.fixture
async def mcp_client():
    """
    Create an in-memory MCP client connected to the Graphiti server.

    This is the FastMCP recommended testing pattern - no subprocess spawning needed.
    The client connects directly to the FastMCP server instance in memory.
    """
    async with Client(transport=mcp) as client:
        yield client


class TestGraphitiMCPInMemory:
    """In-memory tests for Graphiti MCP server using FastMCP Client."""

    @pytest.mark.asyncio
    async def test_list_tools(self, mcp_client: Client):
        """Test that we can list all available tools."""
        tools = await mcp_client.list_tools()

        # Verify we have tools
        assert len(tools) > 0, "Expected at least one tool"

        # Check for expected tools
        tool_names = [tool.name for tool in tools]
        expected_tools = ['add_memory', 'search_memory_facts', 'search_nodes', 'get_status']

        for expected in expected_tools:
            assert expected in tool_names, f"Expected tool '{expected}' not found in {tool_names}"

        print(f"Found {len(tools)} tools: {tool_names}")

    @pytest.mark.asyncio
    async def test_get_status(self, mcp_client: Client):
        """Test the get_status tool returns server status."""
        result = await mcp_client.call_tool('get_status', {})

        # The result should contain status information
        assert result is not None
        print(f"Status result: {result}")

    @pytest.mark.asyncio
    async def test_list_resources(self, mcp_client: Client):
        """Test that we can list resources (if any)."""
        resources = await mcp_client.list_resources()
        print(f"Found {len(resources)} resources")

    @pytest.mark.asyncio
    async def test_list_prompts(self, mcp_client: Client):
        """Test that we can list prompts (if any)."""
        prompts = await mcp_client.list_prompts()
        print(f"Found {len(prompts)} prompts")


class TestGraphitiMCPToolValidation:
    """Validate tool schemas and parameters."""

    @pytest.mark.asyncio
    async def test_add_memory_tool_schema(self, mcp_client: Client):
        """Verify the add_memory tool has correct schema."""
        tools = await mcp_client.list_tools()

        add_memory = next((t for t in tools if t.name == 'add_memory'), None)
        assert add_memory is not None, "add_memory tool not found"

        # Check required parameters
        schema = add_memory.inputSchema
        assert 'properties' in schema
        assert 'name' in schema['properties']
        assert 'episode_body' in schema['properties']

        print(f"add_memory schema: {schema}")

    @pytest.mark.asyncio
    async def test_search_memory_facts_tool_schema(self, mcp_client: Client):
        """Verify the search_memory_facts tool has correct schema."""
        tools = await mcp_client.list_tools()

        search_facts = next((t for t in tools if t.name == 'search_memory_facts'), None)
        assert search_facts is not None, "search_memory_facts tool not found"

        schema = search_facts.inputSchema
        assert 'properties' in schema
        assert 'query' in schema['properties']

        print(f"search_memory_facts schema: {schema}")


# Simple test that can run without database
class TestBasicFunctionality:
    """Basic tests that don't require database connection."""

    @pytest.mark.asyncio
    async def test_mcp_server_instance_exists(self):
        """Verify the MCP server instance is properly configured."""
        assert mcp is not None
        assert mcp.name == 'Graphiti Agent Memory'

    @pytest.mark.asyncio
    async def test_client_connection(self, mcp_client: Client):
        """Test that client can connect to server."""
        # If we get here, connection succeeded
        assert mcp_client is not None


if __name__ == '__main__':
    # Run with: python tests/test_inmemory_example.py
    pytest.main([__file__, '-v', '-s'])
