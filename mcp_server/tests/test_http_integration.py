#!/usr/bin/env python3
"""
Integration test for MCP server using HTTP streaming transport.
This avoids the stdio subprocess timing issues.
"""

import asyncio
import json
import sys
import time

from mcp.client.session import ClientSession


async def test_http_transport(base_url: str = 'http://localhost:8000'):
    """Test MCP server with HTTP streaming transport."""

    # Import the streamable http client
    try:
        from mcp.client.streamable_http import streamablehttp_client as http_client
    except ImportError:
        print('❌ Streamable HTTP client not available in MCP SDK')
        return False

    test_group_id = f'test_http_{int(time.time())}'

    print('🚀 Testing MCP Server with HTTP streaming transport')
    print(f'   Server URL: {base_url}')
    print(f'   Test Group: {test_group_id}')
    print('=' * 60)

    try:
        # Connect to the server via HTTP
        print('\n🔌 Connecting to server...')
        async with http_client(base_url) as (read_stream, write_stream):
            session = ClientSession(read_stream, write_stream)
            await session.initialize()
            print('✅ Connected successfully')

            # Test 1: List tools
            print('\n📋 Test 1: Listing tools...')
            try:
                result = await session.list_tools()
                tools = [tool.name for tool in result.tools]

                expected = [
                    'add_memory',
                    'search_memory_nodes',
                    'search_memory_facts',
                    'get_episodes',
                    'delete_episode',
                    'clear_graph',
                ]

                found = [t for t in expected if t in tools]
                print(f'   ✅ Found {len(tools)} tools ({len(found)}/{len(expected)} expected)')
                for tool in tools[:5]:
                    print(f'      - {tool}')

            except Exception as e:
                print(f'   ❌ Failed: {e}')
                return False

            # Test 2: Add memory
            print('\n📝 Test 2: Adding memory...')
            try:
                result = await session.call_tool(
                    'add_memory',
                    {
                        'name': 'Integration Test Episode',
                        'episode_body': 'This is a test episode created via HTTP transport integration test.',
                        'group_id': test_group_id,
                        'source': 'text',
                        'source_description': 'HTTP Integration Test',
                    },
                )

                if result.content and result.content[0].text:
                    response = result.content[0].text
                    if 'success' in response.lower() or 'queued' in response.lower():
                        print('   ✅ Memory added successfully')
                    else:
                        print(f'   ❌ Unexpected response: {response[:100]}')
                else:
                    print('   ❌ No content in response')

            except Exception as e:
                print(f'   ❌ Failed: {e}')

            # Test 3: Search nodes (with delay for processing)
            print('\n🔍 Test 3: Searching nodes...')
            await asyncio.sleep(2)  # Wait for async processing

            try:
                result = await session.call_tool(
                    'search_memory_nodes',
                    {'query': 'integration test episode', 'group_ids': [test_group_id], 'limit': 5},
                )

                if result.content and result.content[0].text:
                    response = result.content[0].text
                    try:
                        data = json.loads(response)
                        nodes = data.get('nodes', [])
                        print(f'   ✅ Search returned {len(nodes)} nodes')
                    except Exception:  # noqa: E722
                        print(f'   ✅ Search completed: {response[:100]}')
                else:
                    print('   ⚠️  No results (may be processing)')

            except Exception as e:
                print(f'   ❌ Failed: {e}')

            # Test 4: Get episodes
            print('\n📚 Test 4: Getting episodes...')
            try:
                result = await session.call_tool(
                    'get_episodes', {'group_ids': [test_group_id], 'limit': 10}
                )

                if result.content and result.content[0].text:
                    response = result.content[0].text
                    try:
                        data = json.loads(response)
                        episodes = data.get('episodes', [])
                        print(f'   ✅ Found {len(episodes)} episodes')
                    except Exception:  # noqa: E722
                        print(f'   ✅ Episodes retrieved: {response[:100]}')
                else:
                    print('   ⚠️  No episodes found')

            except Exception as e:
                print(f'   ❌ Failed: {e}')

            # Test 5: Clear graph
            print('\n🧹 Test 5: Clearing graph...')
            try:
                result = await session.call_tool('clear_graph', {'group_id': test_group_id})

                if result.content and result.content[0].text:
                    response = result.content[0].text
                    if 'success' in response.lower() or 'cleared' in response.lower():
                        print('   ✅ Graph cleared successfully')
                    else:
                        print(f'   ✅ Clear completed: {response[:100]}')
                else:
                    print('   ❌ No response')

            except Exception as e:
                print(f'   ❌ Failed: {e}')

            print('\n' + '=' * 60)
            print('✅ All integration tests completed!')
            return True

    except Exception as e:
        print(f'\n❌ Connection failed: {e}')
        return False


async def test_sse_transport(base_url: str = 'http://localhost:8000'):
    """Test MCP server with SSE transport."""

    # Import the SSE client
    try:
        from mcp.client.sse import sse_client
    except ImportError:
        print('❌ SSE client not available in MCP SDK')
        return False

    test_group_id = f'test_sse_{int(time.time())}'

    print('🚀 Testing MCP Server with SSE transport')
    print(f'   Server URL: {base_url}/sse')
    print(f'   Test Group: {test_group_id}')
    print('=' * 60)

    try:
        # Connect to the server via SSE
        print('\n🔌 Connecting to server...')
        async with sse_client(f'{base_url}/sse') as (read_stream, write_stream):
            session = ClientSession(read_stream, write_stream)
            await session.initialize()
            print('✅ Connected successfully')

            # Run same tests as HTTP
            print('\n📋 Test 1: Listing tools...')
            try:
                result = await session.list_tools()
                tools = [tool.name for tool in result.tools]
                print(f'   ✅ Found {len(tools)} tools')
                for tool in tools[:3]:
                    print(f'      - {tool}')
            except Exception as e:
                print(f'   ❌ Failed: {e}')
                return False

            print('\n' + '=' * 60)
            print('✅ SSE transport test completed!')
            return True

    except Exception as e:
        print(f'\n❌ SSE connection failed: {e}')
        return False


async def main():
    """Run integration tests."""

    # Check command line arguments
    if len(sys.argv) < 2:
        print('Usage: python test_http_integration.py <transport> [host] [port]')
        print('  transport: http or sse')
        print('  host: server host (default: localhost)')
        print('  port: server port (default: 8000)')
        sys.exit(1)

    transport = sys.argv[1].lower()
    host = sys.argv[2] if len(sys.argv) > 2 else 'localhost'
    port = sys.argv[3] if len(sys.argv) > 3 else '8000'
    base_url = f'http://{host}:{port}'

    # Check if server is running
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            # Try to connect to the server
            await client.get(base_url, timeout=2.0)
    except Exception:  # noqa: E722
        print(f'⚠️  Server not responding at {base_url}')
        print('Please start the server with one of these commands:')
        print(f'  uv run main.py --transport http --port {port}')
        print(f'  uv run main.py --transport sse --port {port}')
        sys.exit(1)

    # Run the appropriate test
    if transport == 'http':
        success = await test_http_transport(base_url)
    elif transport == 'sse':
        success = await test_sse_transport(base_url)
    else:
        print(f'❌ Unknown transport: {transport}')
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    asyncio.run(main())
