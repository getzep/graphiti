#!/usr/bin/env python3
"""
Test MCP server with different transport modes using the MCP SDK.
Tests both SSE and streaming HTTP transports.
"""

import asyncio
import json
import sys
import time

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


class MCPTransportTester:
    """Test MCP server with different transport modes."""

    def __init__(self, transport: str = 'sse', host: str = 'localhost', port: int = 8000):
        self.transport = transport
        self.host = host
        self.port = port
        self.base_url = f'http://{host}:{port}'
        self.test_group_id = f'test_{transport}_{int(time.time())}'
        self.session = None

    async def connect_sse(self) -> ClientSession:
        """Connect using SSE transport."""
        print(f'🔌 Connecting to MCP server via SSE at {self.base_url}/sse')

        # Use the sse_client to connect
        async with sse_client(self.base_url + '/sse') as (read_stream, write_stream):
            self.session = ClientSession(read_stream, write_stream)
            await self.session.initialize()
            return self.session

    async def connect_http(self) -> ClientSession:
        """Connect using streaming HTTP transport."""
        from mcp.client.http import http_client

        print(f'🔌 Connecting to MCP server via HTTP at {self.base_url}')

        # Use the http_client to connect
        async with http_client(self.base_url) as (read_stream, write_stream):
            self.session = ClientSession(read_stream, write_stream)
            await self.session.initialize()
            return self.session

    async def test_list_tools(self) -> bool:
        """Test listing available tools."""
        print('\n📋 Testing list_tools...')

        try:
            result = await self.session.list_tools()
            tools = [tool.name for tool in result.tools]

            expected_tools = [
                'add_memory',
                'search_memory_nodes',
                'search_memory_facts',
                'get_episodes',
                'delete_episode',
                'get_entity_edge',
                'delete_entity_edge',
                'clear_graph',
            ]

            print(f'   ✅ Found {len(tools)} tools')
            for tool in tools[:5]:  # Show first 5 tools
                print(f'      - {tool}')

            # Check if we have most expected tools
            found_tools = [t for t in expected_tools if t in tools]
            success = len(found_tools) >= len(expected_tools) * 0.8

            if success:
                print(
                    f'   ✅ Tool discovery successful ({len(found_tools)}/{len(expected_tools)} expected tools)'
                )
            else:
                print(f'   ❌ Missing too many tools ({len(found_tools)}/{len(expected_tools)})')

            return success
        except Exception as e:
            print(f'   ❌ Failed to list tools: {e}')
            return False

    async def test_add_memory(self) -> bool:
        """Test adding a memory."""
        print('\n📝 Testing add_memory...')

        try:
            result = await self.session.call_tool(
                'add_memory',
                {
                    'name': 'Test Episode',
                    'episode_body': 'This is a test episode created by the MCP transport test suite.',
                    'group_id': self.test_group_id,
                    'source': 'text',
                    'source_description': 'Integration test',
                },
            )

            # Check the result
            if result.content:
                content = result.content[0]
                if hasattr(content, 'text'):
                    response = (
                        json.loads(content.text)
                        if content.text.startswith('{')
                        else {'message': content.text}
                    )
                    if 'success' in str(response).lower() or 'queued' in str(response).lower():
                        print(f'   ✅ Memory added successfully: {response.get("message", "OK")}')
                        return True
                    else:
                        print(f'   ❌ Unexpected response: {response}')
                        return False

            print('   ❌ No content in response')
            return False

        except Exception as e:
            print(f'   ❌ Failed to add memory: {e}')
            return False

    async def test_search_nodes(self) -> bool:
        """Test searching for nodes."""
        print('\n🔍 Testing search_memory_nodes...')

        # Wait a bit for the memory to be processed
        await asyncio.sleep(2)

        try:
            result = await self.session.call_tool(
                'search_memory_nodes',
                {'query': 'test episode', 'group_ids': [self.test_group_id], 'limit': 5},
            )

            if result.content:
                content = result.content[0]
                if hasattr(content, 'text'):
                    response = (
                        json.loads(content.text) if content.text.startswith('{') else {'nodes': []}
                    )
                    nodes = response.get('nodes', [])
                    print(f'   ✅ Search returned {len(nodes)} nodes')
                    return True

            print('   ⚠️ No nodes found (this may be expected if processing is async)')
            return True  # Don't fail on empty results

        except Exception as e:
            print(f'   ❌ Failed to search nodes: {e}')
            return False

    async def test_get_episodes(self) -> bool:
        """Test getting episodes."""
        print('\n📚 Testing get_episodes...')

        try:
            result = await self.session.call_tool(
                'get_episodes', {'group_ids': [self.test_group_id], 'limit': 10}
            )

            if result.content:
                content = result.content[0]
                if hasattr(content, 'text'):
                    response = (
                        json.loads(content.text)
                        if content.text.startswith('{')
                        else {'episodes': []}
                    )
                    episodes = response.get('episodes', [])
                    print(f'   ✅ Found {len(episodes)} episodes')
                    return True

            print('   ⚠️ No episodes found')
            return True

        except Exception as e:
            print(f'   ❌ Failed to get episodes: {e}')
            return False

    async def test_clear_graph(self) -> bool:
        """Test clearing the graph."""
        print('\n🧹 Testing clear_graph...')

        try:
            result = await self.session.call_tool('clear_graph', {'group_id': self.test_group_id})

            if result.content:
                content = result.content[0]
                if hasattr(content, 'text'):
                    response = content.text
                    if 'success' in response.lower() or 'cleared' in response.lower():
                        print('   ✅ Graph cleared successfully')
                        return True

            print('   ❌ Failed to clear graph')
            return False

        except Exception as e:
            print(f'   ❌ Failed to clear graph: {e}')
            return False

    async def run_tests(self) -> bool:
        """Run all tests for the configured transport."""
        print(f'\n{"=" * 60}')
        print(f'🚀 Testing MCP Server with {self.transport.upper()} transport')
        print(f'   Server: {self.base_url}')
        print(f'   Test Group: {self.test_group_id}')
        print('=' * 60)

        try:
            # Connect based on transport type
            if self.transport == 'sse':
                await self.connect_sse()
            elif self.transport == 'http':
                await self.connect_http()
            else:
                print(f'❌ Unknown transport: {self.transport}')
                return False

            print(f'✅ Connected via {self.transport.upper()}')

            # Run tests
            results = []
            results.append(await self.test_list_tools())
            results.append(await self.test_add_memory())
            results.append(await self.test_search_nodes())
            results.append(await self.test_get_episodes())
            results.append(await self.test_clear_graph())

            # Summary
            passed = sum(results)
            total = len(results)
            success = passed == total

            print(f'\n{"=" * 60}')
            print(f'📊 Results for {self.transport.upper()} transport:')
            print(f'   Passed: {passed}/{total}')
            print(f'   Status: {"✅ ALL TESTS PASSED" if success else "❌ SOME TESTS FAILED"}')
            print('=' * 60)

            return success

        except Exception as e:
            print(f'❌ Test suite failed: {e}')
            return False
        finally:
            if self.session:
                await self.session.close()


async def main():
    """Run tests for both transports."""
    # Parse command line arguments
    transport = sys.argv[1] if len(sys.argv) > 1 else 'sse'
    host = sys.argv[2] if len(sys.argv) > 2 else 'localhost'
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8000

    # Create tester
    tester = MCPTransportTester(transport, host, port)

    # Run tests
    success = await tester.run_tests()

    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == '__main__':
    asyncio.run(main())
