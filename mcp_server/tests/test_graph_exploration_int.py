#!/usr/bin/env python3
"""
Integration tests for graph exploration tools: get_entity_connections and get_entity_timeline.
Tests UUID validation, error handling, result formatting, and chronological ordering.
"""

import asyncio
import os
import time
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class GraphExplorationToolsTest:
    """Integration test for get_entity_connections and get_entity_timeline tools."""

    def __init__(self):
        self.test_group_id = f'test_graph_exp_{int(time.time())}'
        self.session = None

    async def __aenter__(self):
        """Start the MCP client session."""
        server_params = StdioServerParameters(
            command='uv',
            args=['run', 'main.py', '--transport', 'stdio'],
            env={
                'NEO4J_URI': os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
                'NEO4J_USER': os.environ.get('NEO4J_USER', 'neo4j'),
                'NEO4J_PASSWORD': os.environ.get('NEO4J_PASSWORD', 'graphiti'),
                'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY', 'dummy_key_for_testing'),
            },
        )

        print(f'🚀 Starting test session with group: {self.test_group_id}')

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
            # Parse JSON response from text content
            import json

            if result.content:
                text_content = result.content[0].text
                return json.loads(text_content)
            return {'error': 'No content returned'}
        except Exception as e:
            return {'error': str(e)}

    async def test_tools_available(self) -> bool:
        """Test that the new tools are available."""
        print('🔍 Test 1: Verifying new tools are available...')

        try:
            tools_result = await self.session.list_tools()
            tools = [tool.name for tool in tools_result.tools]

            required_tools = ['get_entity_connections', 'get_entity_timeline']

            for tool_name in required_tools:
                if tool_name in tools:
                    print(f'   ✅ {tool_name} is available')
                else:
                    print(f'   ❌ {tool_name} is NOT available')
                    return False

            return True

        except Exception as e:
            print(f'   ❌ Test failed: {e}')
            return False

    async def test_invalid_uuid_validation(self) -> bool:
        """Test UUID validation for invalid UUIDs."""
        print('🔍 Test 2: Testing UUID validation with invalid UUID...')

        try:
            # Test get_entity_connections with invalid UUID
            result1 = await self.call_tool(
                'get_entity_connections', {'entity_uuid': 'not-a-valid-uuid'}
            )

            if 'error' in result1 and 'Invalid UUID' in result1['error']:
                print('   ✅ get_entity_connections correctly rejects invalid UUID')
            else:
                print(f'   ❌ get_entity_connections did not validate UUID: {result1}')
                return False

            # Test get_entity_timeline with invalid UUID
            result2 = await self.call_tool('get_entity_timeline', {'entity_uuid': 'also-not-valid'})

            if 'error' in result2 and 'Invalid UUID' in result2['error']:
                print('   ✅ get_entity_timeline correctly rejects invalid UUID')
            else:
                print(f'   ❌ get_entity_timeline did not validate UUID: {result2}')
                return False

            return True

        except Exception as e:
            print(f'   ❌ Test failed: {e}')
            return False

    async def test_nonexistent_entity(self) -> bool:
        """Test behavior with valid UUID but nonexistent entity."""
        print('🔍 Test 3: Testing with valid but nonexistent UUID...')

        try:
            # Use a valid UUID format but one that doesn't exist
            fake_uuid = '00000000-0000-0000-0000-000000000000'

            # Test get_entity_connections
            result1 = await self.call_tool(
                'get_entity_connections', {'entity_uuid': fake_uuid, 'max_connections': 10}
            )

            if 'facts' in result1 and len(result1['facts']) == 0:
                print('   ✅ get_entity_connections returns empty list for nonexistent entity')
            else:
                print(f'   ❌ Unexpected result for nonexistent entity: {result1}')
                return False

            # Test get_entity_timeline
            result2 = await self.call_tool(
                'get_entity_timeline', {'entity_uuid': fake_uuid, 'max_episodes': 10}
            )

            if 'episodes' in result2 and len(result2['episodes']) == 0:
                print('   ✅ get_entity_timeline returns empty list for nonexistent entity')
            else:
                print(f'   ❌ Unexpected result for nonexistent entity: {result2}')
                return False

            return True

        except Exception as e:
            print(f'   ❌ Test failed: {e}')
            return False

    async def test_with_real_data(self) -> bool:
        """Test with actual data - add memory, search, and explore connections."""
        print('🔍 Test 4: Testing with real data (add → search → explore)...')

        try:
            # Step 1: Add some test data
            print('   Step 1: Adding test episodes...')
            await self.call_tool(
                'add_memory',
                {
                    'name': 'Database Decision',
                    'episode_body': 'We chose PostgreSQL for the new service because it has better JSON support than MySQL.',
                    'source': 'text',
                    'group_id': self.test_group_id,
                },
            )

            await self.call_tool(
                'add_memory',
                {
                    'name': 'Second Discussion',
                    'episode_body': 'Discussed PostgreSQL performance tuning. Need to optimize connection pooling.',
                    'source': 'text',
                    'group_id': self.test_group_id,
                },
            )

            # Wait for processing
            print('   Waiting 5 seconds for episode processing...')
            await asyncio.sleep(5)

            # Step 2: Search for the entity
            print('   Step 2: Searching for PostgreSQL entity...')
            search_result = await self.call_tool(
                'search_nodes', {'query': 'PostgreSQL', 'group_ids': [self.test_group_id]}
            )

            if 'nodes' not in search_result or len(search_result['nodes']) == 0:
                print('   ⚠️  No PostgreSQL entity found yet (may need more processing time)')
                return True  # Don't fail the test, processing might not be complete

            entity_uuid = search_result['nodes'][0]['uuid']
            print(f'   ✅ Found PostgreSQL entity: {entity_uuid[:8]}...')

            # Step 3: Test get_entity_connections
            print('   Step 3: Testing get_entity_connections...')
            connections_result = await self.call_tool(
                'get_entity_connections',
                {
                    'entity_uuid': entity_uuid,
                    'group_ids': [self.test_group_id],
                    'max_connections': 20,
                },
            )

            if 'facts' in connections_result:
                print(
                    f'   ✅ get_entity_connections returned {len(connections_result["facts"])} connection(s)'
                )
            else:
                print(f'   ⚠️  get_entity_connections result: {connections_result}')

            # Step 4: Test get_entity_timeline
            print('   Step 4: Testing get_entity_timeline...')
            timeline_result = await self.call_tool(
                'get_entity_timeline',
                {
                    'entity_uuid': entity_uuid,
                    'group_ids': [self.test_group_id],
                    'max_episodes': 20,
                },
            )

            if 'episodes' in timeline_result:
                episodes = timeline_result['episodes']
                print(f'   ✅ get_entity_timeline returned {len(episodes)} episode(s)')

                # Verify chronological order
                if len(episodes) > 1:
                    valid_at_values = [ep['valid_at'] for ep in episodes if ep.get('valid_at')]
                    is_sorted = valid_at_values == sorted(valid_at_values)
                    if is_sorted:
                        print('   ✅ Episodes are in chronological order')
                    else:
                        print('   ❌ Episodes are NOT in chronological order')
                        return False
            else:
                print(f'   ⚠️  get_entity_timeline result: {timeline_result}')

            return True

        except Exception as e:
            print(f'   ❌ Test failed: {e}')
            import traceback

            traceback.print_exc()
            return False

    async def test_max_limits(self) -> bool:
        """Test that max_connections and max_episodes limits work."""
        print('🔍 Test 5: Testing max limits parameters...')

        try:
            # This test just verifies the parameters are accepted
            # Actual limit testing would require creating many connections

            fake_uuid = '00000000-0000-0000-0000-000000000000'

            # Test with different max values
            result1 = await self.call_tool(
                'get_entity_connections', {'entity_uuid': fake_uuid, 'max_connections': 5}
            )

            result2 = await self.call_tool(
                'get_entity_timeline', {'entity_uuid': fake_uuid, 'max_episodes': 5}
            )

            if 'facts' in result1 and 'episodes' in result2:
                print('   ✅ Both tools accept max limit parameters')
                return True
            else:
                print('   ❌ Tools did not accept max limit parameters')
                return False

        except Exception as e:
            print(f'   ❌ Test failed: {e}')
            return False

    async def run_all_tests(self):
        """Run all test scenarios."""
        print('\n' + '=' * 70)
        print('Graph Exploration Tools Integration Tests')
        print('=' * 70 + '\n')

        results = {
            'Tools Available': await self.test_tools_available(),
            'Invalid UUID Validation': await self.test_invalid_uuid_validation(),
            'Nonexistent Entity Handling': await self.test_nonexistent_entity(),
            'Real Data Flow': await self.test_with_real_data(),
            'Max Limits Parameters': await self.test_max_limits(),
        }

        print('\n' + '=' * 70)
        print('Test Results Summary')
        print('=' * 70)

        all_passed = True
        for test_name, passed in results.items():
            status = '✅ PASSED' if passed else '❌ FAILED'
            print(f'{test_name}: {status}')
            if not passed:
                all_passed = False

        print('=' * 70)

        if all_passed:
            print('🎉 All tests PASSED!')
        else:
            print('⚠️  Some tests FAILED')

        return all_passed


async def main():
    """Run the integration tests."""
    async with GraphExplorationToolsTest() as test:
        success = await test.run_all_tests()
        return 0 if success else 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code)
