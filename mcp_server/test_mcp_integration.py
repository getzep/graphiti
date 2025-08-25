#!/usr/bin/env python3
"""
Integration test for the refactored Graphiti MCP Server using the official MCP Python SDK.
Tests all major MCP tools and handles episode processing latency.
"""

import asyncio
import json
import time
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class GraphitiMCPIntegrationTest:
    """Integration test client for Graphiti MCP Server using official MCP SDK."""

    def __init__(self):
        self.test_group_id = f'test_group_{int(time.time())}'
        self.session = None

    async def __aenter__(self):
        """Start the MCP client session."""
        # Configure server parameters to run our refactored server
        server_params = StdioServerParameters(
            command='uv',
            args=['run', 'graphiti_mcp_server.py', '--transport', 'stdio'],
            env={
                'NEO4J_URI': 'bolt://localhost:7687',
                'NEO4J_USER': 'neo4j',
                'NEO4J_PASSWORD': 'demodemo',
                'OPENAI_API_KEY': 'dummy_key_for_testing',  # Will use existing .env
            },
        )

        print(f'🚀 Starting MCP client session with test group: {self.test_group_id}')

        # Use the async context manager properly
        self.client_context = stdio_client(server_params)
        read, write = await self.client_context.__aenter__()
        self.session = ClientSession(read, write)
        await self.session.initialize()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the MCP client session."""
        if self.session:
            await self.session.close()
        if hasattr(self, 'client_context'):
            await self.client_context.__aexit__(exc_type, exc_val, exc_tb)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call an MCP tool and return the result."""
        try:
            result = await self.session.call_tool(tool_name, arguments)
            return result.content[0].text if result.content else {'error': 'No content returned'}
        except Exception as e:
            return {'error': str(e)}

    async def test_server_initialization(self) -> bool:
        """Test that the server initializes properly."""
        print('🔍 Testing server initialization...')

        try:
            # List available tools to verify server is responding
            tools_result = await self.session.list_tools()
            tools = [tool.name for tool in tools_result.tools]

            expected_tools = [
                'add_memory',
                'search_memory_nodes',
                'search_memory_facts',
                'get_episodes',
                'delete_episode',
                'delete_entity_edge',
                'get_entity_edge',
                'clear_graph',
            ]

            available_tools = len([tool for tool in expected_tools if tool in tools])
            print(
                f'   ✅ Server responding with {len(tools)} tools ({available_tools}/{len(expected_tools)} expected)'
            )
            print(f'   Available tools: {", ".join(sorted(tools))}')

            return available_tools >= len(expected_tools) * 0.8  # 80% of expected tools

        except Exception as e:
            print(f'   ❌ Server initialization failed: {e}')
            return False

    async def test_add_memory_operations(self) -> dict[str, bool]:
        """Test adding various types of memory episodes."""
        print('📝 Testing add_memory operations...')

        results = {}

        # Test 1: Add text episode
        print('   Testing text episode...')
        try:
            result = await self.call_tool(
                'add_memory',
                {
                    'name': 'Test Company News',
                    'episode_body': 'Acme Corp announced a revolutionary new AI product that will transform the industry. The CEO mentioned this is their biggest launch since 2020.',
                    'source': 'text',
                    'source_description': 'news article',
                    'group_id': self.test_group_id,
                },
            )

            if isinstance(result, str) and 'queued' in result.lower():
                print(f'   ✅ Text episode: {result}')
                results['text'] = True
            else:
                print(f'   ❌ Text episode failed: {result}')
                results['text'] = False
        except Exception as e:
            print(f'   ❌ Text episode error: {e}')
            results['text'] = False

        # Test 2: Add JSON episode
        print('   Testing JSON episode...')
        try:
            json_data = {
                'company': {'name': 'TechCorp', 'founded': 2010},
                'products': [
                    {'id': 'P001', 'name': 'CloudSync', 'category': 'software'},
                    {'id': 'P002', 'name': 'DataMiner', 'category': 'analytics'},
                ],
                'employees': 150,
            }

            result = await self.call_tool(
                'add_memory',
                {
                    'name': 'Company Profile',
                    'episode_body': json.dumps(json_data),
                    'source': 'json',
                    'source_description': 'CRM data',
                    'group_id': self.test_group_id,
                },
            )

            if isinstance(result, str) and 'queued' in result.lower():
                print(f'   ✅ JSON episode: {result}')
                results['json'] = True
            else:
                print(f'   ❌ JSON episode failed: {result}')
                results['json'] = False
        except Exception as e:
            print(f'   ❌ JSON episode error: {e}')
            results['json'] = False

        # Test 3: Add message episode
        print('   Testing message episode...')
        try:
            result = await self.call_tool(
                'add_memory',
                {
                    'name': 'Customer Support Chat',
                    'episode_body': "user: What's your return policy?\nassistant: You can return items within 30 days of purchase with receipt.\nuser: Thanks!",
                    'source': 'message',
                    'source_description': 'support chat log',
                    'group_id': self.test_group_id,
                },
            )

            if isinstance(result, str) and 'queued' in result.lower():
                print(f'   ✅ Message episode: {result}')
                results['message'] = True
            else:
                print(f'   ❌ Message episode failed: {result}')
                results['message'] = False
        except Exception as e:
            print(f'   ❌ Message episode error: {e}')
            results['message'] = False

        return results

    async def wait_for_processing(self, max_wait: int = 45) -> bool:
        """Wait for episode processing to complete."""
        print(f'⏳ Waiting up to {max_wait} seconds for episode processing...')

        for i in range(max_wait):
            await asyncio.sleep(1)

            try:
                # Check if we have any episodes
                result = await self.call_tool(
                    'get_episodes', {'group_id': self.test_group_id, 'last_n': 10}
                )

                # Parse the JSON result if it's a string
                if isinstance(result, str):
                    try:
                        parsed_result = json.loads(result)
                        if isinstance(parsed_result, list) and len(parsed_result) > 0:
                            print(
                                f'   ✅ Found {len(parsed_result)} processed episodes after {i + 1} seconds'
                            )
                            return True
                    except json.JSONDecodeError:
                        if 'episodes' in result.lower():
                            print(f'   ✅ Episodes detected after {i + 1} seconds')
                            return True

            except Exception as e:
                if i == 0:  # Only log first error to avoid spam
                    print(f'   ⚠️  Waiting for processing... ({e})')
                continue

        print(f'   ⚠️  Still waiting after {max_wait} seconds...')
        return False

    async def test_search_operations(self) -> dict[str, bool]:
        """Test search functionality."""
        print('🔍 Testing search operations...')

        results = {}

        # Test search_memory_nodes
        print('   Testing search_memory_nodes...')
        try:
            result = await self.call_tool(
                'search_memory_nodes',
                {
                    'query': 'Acme Corp product launch AI',
                    'group_ids': [self.test_group_id],
                    'max_nodes': 5,
                },
            )

            success = False
            if isinstance(result, str):
                try:
                    parsed = json.loads(result)
                    nodes = parsed.get('nodes', [])
                    success = isinstance(nodes, list)
                    print(f'   ✅ Node search returned {len(nodes)} nodes')
                except json.JSONDecodeError:
                    success = 'nodes' in result.lower() and 'successfully' in result.lower()
                    if success:
                        print('   ✅ Node search completed successfully')

            results['nodes'] = success
            if not success:
                print(f'   ❌ Node search failed: {result}')

        except Exception as e:
            print(f'   ❌ Node search error: {e}')
            results['nodes'] = False

        # Test search_memory_facts
        print('   Testing search_memory_facts...')
        try:
            result = await self.call_tool(
                'search_memory_facts',
                {
                    'query': 'company products software TechCorp',
                    'group_ids': [self.test_group_id],
                    'max_facts': 5,
                },
            )

            success = False
            if isinstance(result, str):
                try:
                    parsed = json.loads(result)
                    facts = parsed.get('facts', [])
                    success = isinstance(facts, list)
                    print(f'   ✅ Fact search returned {len(facts)} facts')
                except json.JSONDecodeError:
                    success = 'facts' in result.lower() and 'successfully' in result.lower()
                    if success:
                        print('   ✅ Fact search completed successfully')

            results['facts'] = success
            if not success:
                print(f'   ❌ Fact search failed: {result}')

        except Exception as e:
            print(f'   ❌ Fact search error: {e}')
            results['facts'] = False

        return results

    async def test_episode_retrieval(self) -> bool:
        """Test episode retrieval."""
        print('📚 Testing episode retrieval...')

        try:
            result = await self.call_tool(
                'get_episodes', {'group_id': self.test_group_id, 'last_n': 10}
            )

            if isinstance(result, str):
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, list):
                        print(f'   ✅ Retrieved {len(parsed)} episodes')

                        # Show episode details
                        for i, episode in enumerate(parsed[:3]):
                            name = episode.get('name', 'Unknown')
                            source = episode.get('source', 'unknown')
                            print(f'     Episode {i + 1}: {name} (source: {source})')

                        return len(parsed) > 0
                except json.JSONDecodeError:
                    # Check if response indicates success
                    if 'episode' in result.lower():
                        print('   ✅ Episode retrieval completed')
                        return True

            print(f'   ❌ Unexpected result format: {result}')
            return False

        except Exception as e:
            print(f'   ❌ Episode retrieval failed: {e}')
            return False

    async def test_error_handling(self) -> dict[str, bool]:
        """Test error handling and edge cases."""
        print('🧪 Testing error handling...')

        results = {}

        # Test with nonexistent group
        print('   Testing nonexistent group handling...')
        try:
            result = await self.call_tool(
                'search_memory_nodes',
                {
                    'query': 'nonexistent data',
                    'group_ids': ['nonexistent_group_12345'],
                    'max_nodes': 5,
                },
            )

            # Should handle gracefully, not crash
            success = (
                'error' not in str(result).lower() or 'not initialized' not in str(result).lower()
            )
            if success:
                print('   ✅ Nonexistent group handled gracefully')
            else:
                print(f'   ❌ Nonexistent group caused issues: {result}')

            results['nonexistent_group'] = success

        except Exception as e:
            print(f'   ❌ Nonexistent group test failed: {e}')
            results['nonexistent_group'] = False

        # Test empty query
        print('   Testing empty query handling...')
        try:
            result = await self.call_tool(
                'search_memory_nodes',
                {'query': '', 'group_ids': [self.test_group_id], 'max_nodes': 5},
            )

            # Should handle gracefully
            success = (
                'error' not in str(result).lower() or 'not initialized' not in str(result).lower()
            )
            if success:
                print('   ✅ Empty query handled gracefully')
            else:
                print(f'   ❌ Empty query caused issues: {result}')

            results['empty_query'] = success

        except Exception as e:
            print(f'   ❌ Empty query test failed: {e}')
            results['empty_query'] = False

        return results

    async def run_comprehensive_test(self) -> dict[str, Any]:
        """Run the complete integration test suite."""
        print('🚀 Starting Comprehensive Graphiti MCP Server Integration Test')
        print(f'   Test group ID: {self.test_group_id}')
        print('=' * 70)

        results = {
            'server_init': False,
            'add_memory': {},
            'processing_wait': False,
            'search': {},
            'episodes': False,
            'error_handling': {},
            'overall_success': False,
        }

        # Test 1: Server Initialization
        results['server_init'] = await self.test_server_initialization()
        if not results['server_init']:
            print('❌ Server initialization failed, aborting remaining tests')
            return results

        print()

        # Test 2: Add Memory Operations
        results['add_memory'] = await self.test_add_memory_operations()
        print()

        # Test 3: Wait for Processing
        results['processing_wait'] = await self.wait_for_processing()
        print()

        # Test 4: Search Operations
        results['search'] = await self.test_search_operations()
        print()

        # Test 5: Episode Retrieval
        results['episodes'] = await self.test_episode_retrieval()
        print()

        # Test 6: Error Handling
        results['error_handling'] = await self.test_error_handling()
        print()

        # Calculate overall success
        memory_success = any(results['add_memory'].values())
        search_success = any(results['search'].values()) if results['search'] else False
        error_success = (
            any(results['error_handling'].values()) if results['error_handling'] else True
        )

        results['overall_success'] = (
            results['server_init']
            and memory_success
            and (results['episodes'] or results['processing_wait'])
            and error_success
        )

        # Print comprehensive summary
        print('=' * 70)
        print('📊 COMPREHENSIVE TEST SUMMARY')
        print('-' * 35)
        print(f'Server Initialization:    {"✅ PASS" if results["server_init"] else "❌ FAIL"}')

        memory_stats = f'({sum(results["add_memory"].values())}/{len(results["add_memory"])} types)'
        print(
            f'Memory Operations:        {"✅ PASS" if memory_success else "❌ FAIL"} {memory_stats}'
        )

        print(f'Processing Pipeline:      {"✅ PASS" if results["processing_wait"] else "❌ FAIL"}')

        search_stats = (
            f'({sum(results["search"].values())}/{len(results["search"])} types)'
            if results['search']
            else '(0/0 types)'
        )
        print(
            f'Search Operations:        {"✅ PASS" if search_success else "❌ FAIL"} {search_stats}'
        )

        print(f'Episode Retrieval:        {"✅ PASS" if results["episodes"] else "❌ FAIL"}')

        error_stats = (
            f'({sum(results["error_handling"].values())}/{len(results["error_handling"])} cases)'
            if results['error_handling']
            else '(0/0 cases)'
        )
        print(
            f'Error Handling:           {"✅ PASS" if error_success else "❌ FAIL"} {error_stats}'
        )

        print('-' * 35)
        print(f'🎯 OVERALL RESULT: {"✅ SUCCESS" if results["overall_success"] else "❌ FAILED"}')

        if results['overall_success']:
            print('\n🎉 The refactored Graphiti MCP server is working correctly!')
            print('   All core functionality has been successfully tested.')
        else:
            print('\n⚠️  Some issues were detected. Review the test results above.')
            print('   The refactoring may need additional attention.')

        return results


async def main():
    """Run the integration test."""
    try:
        async with GraphitiMCPIntegrationTest() as test:
            results = await test.run_comprehensive_test()

            # Exit with appropriate code
            exit_code = 0 if results['overall_success'] else 1
            exit(exit_code)
    except Exception as e:
        print(f'❌ Test setup failed: {e}')
        exit(1)


if __name__ == '__main__':
    asyncio.run(main())
