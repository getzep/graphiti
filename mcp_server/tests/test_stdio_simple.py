#!/usr/bin/env python3
"""
Simple test to verify MCP server works with stdio transport.
"""

import asyncio
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_stdio():
    """Test basic MCP server functionality with stdio transport."""
    print('🚀 Testing MCP Server with stdio transport')
    print('=' * 50)

    # Configure server parameters
    server_params = StdioServerParameters(
        command='uv',
        args=['run', '../main.py', '--transport', 'stdio'],
        env={
            'NEO4J_URI': os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
            'NEO4J_USER': os.environ.get('NEO4J_USER', 'neo4j'),
            'NEO4J_PASSWORD': os.environ.get('NEO4J_PASSWORD', 'graphiti'),
            'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY', 'dummy'),
        },
    )

    try:
        async with stdio_client(server_params) as (read, write):  # noqa: SIM117
            async with ClientSession(read, write) as session:
                print('✅ Connected to server')

                # Initialize the session
                await session.initialize()
                print('✅ Session initialized')

                # Wait for server to be fully ready
                await asyncio.sleep(2)

                # List tools
                print('\n📋 Listing available tools...')
                tools = await session.list_tools()
                print(f'   Found {len(tools.tools)} tools:')
                for tool in tools.tools[:5]:
                    print(f'   - {tool.name}')

                # Test add_memory
                print('\n📝 Testing add_memory...')
                result = await session.call_tool(
                    'add_memory',
                    {
                        'name': 'Test Episode',
                        'episode_body': 'Simple test episode',
                        'group_id': 'test_group',
                        'source': 'text',
                    },
                )

                if result.content:
                    print(f'   ✅ Memory added: {result.content[0].text[:100]}')

                # Test search
                print('\n🔍 Testing search_memory_nodes...')
                result = await session.call_tool(
                    'search_memory_nodes',
                    {'query': 'test', 'group_ids': ['test_group'], 'limit': 5},
                )

                if result.content:
                    print(f'   ✅ Search completed: {result.content[0].text[:100]}')

                print('\n✅ All tests completed successfully!')
                return True

    except Exception as e:
        print(f'\n❌ Test failed: {e}')
        import traceback

        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = asyncio.run(test_stdio())
    exit(0 if success else 1)
