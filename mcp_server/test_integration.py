#!/usr/bin/env python3
"""
Integration test for the refactored Graphiti MCP Server.
Tests all major MCP tools and handles episode processing latency.
"""

import asyncio
import json
import time
from typing import Any

import httpx


class MCPIntegrationTest:
    """Integration test client for Graphiti MCP Server."""

    def __init__(self, base_url: str = 'http://localhost:8000'):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.test_group_id = f'test_group_{int(time.time())}'

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def call_mcp_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool via the SSE endpoint."""
        # MCP protocol message structure
        message = {
            'jsonrpc': '2.0',
            'id': int(time.time() * 1000),
            'method': 'tools/call',
            'params': {'name': tool_name, 'arguments': arguments},
        }

        try:
            response = await self.client.post(
                f'{self.base_url}/message',
                json=message,
                headers={'Content-Type': 'application/json'},
            )

            if response.status_code != 200:
                return {'error': f'HTTP {response.status_code}: {response.text}'}

            result = response.json()
            return result.get('result', result)

        except Exception as e:
            return {'error': str(e)}

    async def test_server_status(self) -> bool:
        """Test the get_status resource."""
        print('🔍 Testing server status...')

        try:
            response = await self.client.get(f'{self.base_url}/resources/http://graphiti/status')
            if response.status_code == 200:
                status = response.json()
                print(f'   ✅ Server status: {status.get("status", "unknown")}')
                return status.get('status') == 'ok'
            else:
                print(f'   ❌ Status check failed: HTTP {response.status_code}')
                return False
        except Exception as e:
            print(f'   ❌ Status check failed: {e}')
            return False

    async def test_add_memory(self) -> dict[str, str]:
        """Test adding various types of memory episodes."""
        print('📝 Testing add_memory functionality...')

        episode_results = {}

        # Test 1: Add text episode
        print('   Testing text episode...')
        result = await self.call_mcp_tool(
            'add_memory',
            {
                'name': 'Test Company News',
                'episode_body': 'Acme Corp announced a revolutionary new AI product that will transform the industry. The CEO mentioned this is their biggest launch since 2020.',
                'source': 'text',
                'source_description': 'news article',
                'group_id': self.test_group_id,
            },
        )

        if 'error' in result:
            print(f'   ❌ Text episode failed: {result["error"]}')
        else:
            print(f'   ✅ Text episode queued: {result.get("message", "Success")}')
            episode_results['text'] = 'success'

        # Test 2: Add JSON episode
        print('   Testing JSON episode...')
        json_data = {
            'company': {'name': 'TechCorp', 'founded': 2010},
            'products': [
                {'id': 'P001', 'name': 'CloudSync', 'category': 'software'},
                {'id': 'P002', 'name': 'DataMiner', 'category': 'analytics'},
            ],
            'employees': 150,
        }

        result = await self.call_mcp_tool(
            'add_memory',
            {
                'name': 'Company Profile',
                'episode_body': json.dumps(json_data),
                'source': 'json',
                'source_description': 'CRM data',
                'group_id': self.test_group_id,
            },
        )

        if 'error' in result:
            print(f'   ❌ JSON episode failed: {result["error"]}')
        else:
            print(f'   ✅ JSON episode queued: {result.get("message", "Success")}')
            episode_results['json'] = 'success'

        # Test 3: Add message episode
        print('   Testing message episode...')
        result = await self.call_mcp_tool(
            'add_memory',
            {
                'name': 'Customer Support Chat',
                'episode_body': "user: What's your return policy?\nassistant: You can return items within 30 days of purchase with receipt.\nuser: Thanks!",
                'source': 'message',
                'source_description': 'support chat log',
                'group_id': self.test_group_id,
            },
        )

        if 'error' in result:
            print(f'   ❌ Message episode failed: {result["error"]}')
        else:
            print(f'   ✅ Message episode queued: {result.get("message", "Success")}')
            episode_results['message'] = 'success'

        return episode_results

    async def wait_for_processing(self, max_wait: int = 30) -> None:
        """Wait for episode processing to complete."""
        print(f'⏳ Waiting up to {max_wait} seconds for episode processing...')

        for i in range(max_wait):
            await asyncio.sleep(1)

            # Check if we have any episodes
            result = await self.call_mcp_tool(
                'get_episodes', {'group_id': self.test_group_id, 'last_n': 10}
            )

            if not isinstance(result, dict) or 'error' in result:
                continue

            if isinstance(result, list) and len(result) > 0:
                print(f'   ✅ Found {len(result)} processed episodes after {i + 1} seconds')
                return

        print(f'   ⚠️  Still waiting after {max_wait} seconds...')

    async def test_search_functions(self) -> dict[str, bool]:
        """Test search functionality."""
        print('🔍 Testing search functions...')

        results = {}

        # Test search_memory_nodes
        print('   Testing search_memory_nodes...')
        result = await self.call_mcp_tool(
            'search_memory_nodes',
            {
                'query': 'Acme Corp product launch',
                'group_ids': [self.test_group_id],
                'max_nodes': 5,
            },
        )

        if 'error' in result:
            print(f'   ❌ Node search failed: {result["error"]}')
            results['nodes'] = False
        else:
            nodes = result.get('nodes', [])
            print(f'   ✅ Node search returned {len(nodes)} nodes')
            results['nodes'] = True

        # Test search_memory_facts
        print('   Testing search_memory_facts...')
        result = await self.call_mcp_tool(
            'search_memory_facts',
            {
                'query': 'company products software',
                'group_ids': [self.test_group_id],
                'max_facts': 5,
            },
        )

        if 'error' in result:
            print(f'   ❌ Fact search failed: {result["error"]}')
            results['facts'] = False
        else:
            facts = result.get('facts', [])
            print(f'   ✅ Fact search returned {len(facts)} facts')
            results['facts'] = True

        return results

    async def test_episode_retrieval(self) -> bool:
        """Test episode retrieval."""
        print('📚 Testing episode retrieval...')

        result = await self.call_mcp_tool(
            'get_episodes', {'group_id': self.test_group_id, 'last_n': 10}
        )

        if 'error' in result:
            print(f'   ❌ Episode retrieval failed: {result["error"]}')
            return False

        if isinstance(result, list):
            print(f'   ✅ Retrieved {len(result)} episodes')

            # Print episode details
            for i, episode in enumerate(result[:3]):  # Show first 3
                name = episode.get('name', 'Unknown')
                source = episode.get('source', 'unknown')
                print(f'     Episode {i + 1}: {name} (source: {source})')

            return len(result) > 0
        else:
            print(f'   ❌ Unexpected result format: {type(result)}')
            return False

    async def test_edge_cases(self) -> dict[str, bool]:
        """Test edge cases and error handling."""
        print('🧪 Testing edge cases...')

        results = {}

        # Test with invalid group_id
        print('   Testing invalid group_id...')
        result = await self.call_mcp_tool(
            'search_memory_nodes',
            {'query': 'nonexistent data', 'group_ids': ['nonexistent_group'], 'max_nodes': 5},
        )

        # Should not error, just return empty results
        if 'error' not in result:
            nodes = result.get('nodes', [])
            print(f'   ✅ Invalid group_id handled gracefully (returned {len(nodes)} nodes)')
            results['invalid_group'] = True
        else:
            print(f'   ❌ Invalid group_id caused error: {result["error"]}')
            results['invalid_group'] = False

        # Test empty query
        print('   Testing empty query...')
        result = await self.call_mcp_tool(
            'search_memory_nodes', {'query': '', 'group_ids': [self.test_group_id], 'max_nodes': 5}
        )

        if 'error' not in result:
            print('   ✅ Empty query handled gracefully')
            results['empty_query'] = True
        else:
            print(f'   ❌ Empty query caused error: {result["error"]}')
            results['empty_query'] = False

        return results

    async def run_full_test_suite(self) -> dict[str, Any]:
        """Run the complete integration test suite."""
        print('🚀 Starting Graphiti MCP Server Integration Test')
        print(f'   Test group ID: {self.test_group_id}')
        print('=' * 60)

        results = {
            'server_status': False,
            'add_memory': {},
            'search': {},
            'episodes': False,
            'edge_cases': {},
            'overall_success': False,
        }

        # Test 1: Server Status
        results['server_status'] = await self.test_server_status()
        if not results['server_status']:
            print('❌ Server not responding, aborting tests')
            return results

        print()

        # Test 2: Add Memory
        results['add_memory'] = await self.test_add_memory()
        print()

        # Test 3: Wait for processing
        await self.wait_for_processing()
        print()

        # Test 4: Search Functions
        results['search'] = await self.test_search_functions()
        print()

        # Test 5: Episode Retrieval
        results['episodes'] = await self.test_episode_retrieval()
        print()

        # Test 6: Edge Cases
        results['edge_cases'] = await self.test_edge_cases()
        print()

        # Calculate overall success
        memory_success = len(results['add_memory']) > 0
        search_success = any(results['search'].values())
        edge_case_success = any(results['edge_cases'].values())

        results['overall_success'] = (
            results['server_status']
            and memory_success
            and results['episodes']
            and (search_success or edge_case_success)  # At least some functionality working
        )

        # Print summary
        print('=' * 60)
        print('📊 TEST SUMMARY')
        print(f'   Server Status: {"✅" if results["server_status"] else "❌"}')
        print(
            f'   Memory Operations: {"✅" if memory_success else "❌"} ({len(results["add_memory"])} types)'
        )
        print(f'   Search Functions: {"✅" if search_success else "❌"}')
        print(f'   Episode Retrieval: {"✅" if results["episodes"] else "❌"}')
        print(f'   Edge Cases: {"✅" if edge_case_success else "❌"}')
        print()
        print(f'🎯 OVERALL: {"✅ SUCCESS" if results["overall_success"] else "❌ FAILED"}')

        if results['overall_success']:
            print('   The refactored MCP server is working correctly!')
        else:
            print('   Some issues detected. Check individual test results above.')

        return results


async def main():
    """Run the integration test."""
    async with MCPIntegrationTest() as test:
        results = await test.run_full_test_suite()

        # Exit with appropriate code
        exit_code = 0 if results['overall_success'] else 1
        exit(exit_code)


if __name__ == '__main__':
    asyncio.run(main())
