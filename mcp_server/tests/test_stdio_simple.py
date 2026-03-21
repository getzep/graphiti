#!/usr/bin/env python3
"""
Simple test to verify MCP server works with stdio transport.
"""

import asyncio
import os

from ingest_wait_helpers import extract_episode_uuid, wait_for_ingest_completion
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
            'UV_CACHE_DIR': '/tmp/graphiti-uv-cache',
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
                episode_uuid = extract_episode_uuid(result)
                if episode_uuid:
                    await wait_for_ingest_completion(
                        lambda tool_name, arguments: session.call_tool(tool_name, arguments),
                        episode_uuids=[episode_uuid],
                        group_id='test_group',
                        max_wait=30,
                        poll_interval=2,
                    )

                # Test search
                print('\n🔍 Testing search_nodes...')
                result = await session.call_tool(
                    'search_nodes',
                    {'query': 'test', 'group_ids': ['test_group'], 'max_nodes': 5},
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
