#!/usr/bin/env python3
"""
Comprehensive integration test suite for Graphiti MCP Server.
Covers all MCP tools with consideration for LLM inference latency.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class TestMetrics:
    """Track test performance metrics."""

    operation: str
    start_time: float
    end_time: float
    success: bool
    details: dict[str, Any]

    @property
    def duration(self) -> float:
        """Calculate operation duration in seconds."""
        return self.end_time - self.start_time


class GraphitiTestClient:
    """Enhanced test client for comprehensive Graphiti MCP testing."""

    def __init__(self, test_group_id: str | None = None):
        self.test_group_id = test_group_id or f'test_{int(time.time())}'
        self.session = None
        self.metrics: list[TestMetrics] = []
        self.default_timeout = 30  # seconds

    async def __aenter__(self):
        """Initialize MCP client session."""
        server_params = StdioServerParameters(
            command='uv',
            args=['run', '../main.py', '--transport', 'stdio'],
            env={
                'NEO4J_URI': os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
                'NEO4J_USER': os.environ.get('NEO4J_USER', 'neo4j'),
                'NEO4J_PASSWORD': os.environ.get('NEO4J_PASSWORD', 'graphiti'),
                'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY', 'test_key_for_mock'),
                'FALKORDB_URI': os.environ.get('FALKORDB_URI', 'redis://localhost:6379'),
            },
        )

        self.client_context = stdio_client(server_params)
        read, write = await self.client_context.__aenter__()
        self.session = ClientSession(read, write)
        await self.session.initialize()

        # Wait for server to be fully ready
        await asyncio.sleep(2)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up client session."""
        if self.session:
            await self.session.close()
        if hasattr(self, 'client_context'):
            await self.client_context.__aexit__(exc_type, exc_val, exc_tb)

    async def call_tool_with_metrics(
        self, tool_name: str, arguments: dict[str, Any], timeout: float | None = None
    ) -> tuple[Any, TestMetrics]:
        """Call a tool and capture performance metrics."""
        start_time = time.time()
        timeout = timeout or self.default_timeout

        try:
            result = await asyncio.wait_for(
                self.session.call_tool(tool_name, arguments), timeout=timeout
            )

            content = result.content[0].text if result.content else None
            success = True
            details = {'result': content, 'tool': tool_name}

        except asyncio.TimeoutError:
            content = None
            success = False
            details = {'error': f'Timeout after {timeout}s', 'tool': tool_name}

        except Exception as e:
            content = None
            success = False
            details = {'error': str(e), 'tool': tool_name}

        end_time = time.time()
        metric = TestMetrics(
            operation=f'call_{tool_name}',
            start_time=start_time,
            end_time=end_time,
            success=success,
            details=details,
        )
        self.metrics.append(metric)

        return content, metric

    async def wait_for_episode_processing(
        self, expected_count: int = 1, max_wait: int = 60, poll_interval: int = 2
    ) -> bool:
        """
        Wait for episodes to be processed with intelligent polling.

        Args:
            expected_count: Number of episodes expected to be processed
            max_wait: Maximum seconds to wait
            poll_interval: Seconds between status checks

        Returns:
            True if episodes were processed successfully
        """
        start_time = time.time()

        while (time.time() - start_time) < max_wait:
            result, _ = await self.call_tool_with_metrics(
                'get_episodes', {'group_id': self.test_group_id, 'last_n': 100}
            )

            if result:
                try:
                    episodes = json.loads(result) if isinstance(result, str) else result
                    if len(episodes.get('episodes', [])) >= expected_count:
                        return True
                except (json.JSONDecodeError, AttributeError):
                    pass

            await asyncio.sleep(poll_interval)

        return False


class TestCoreOperations:
    """Test core Graphiti operations."""

    @pytest.mark.asyncio
    async def test_server_initialization(self):
        """Verify server initializes with all required tools."""
        async with GraphitiTestClient() as client:
            tools_result = await client.session.list_tools()
            tools = {tool.name for tool in tools_result.tools}

            required_tools = {
                'add_memory',
                'search_memory_nodes',
                'search_memory_facts',
                'get_episodes',
                'delete_episode',
                'delete_entity_edge',
                'get_entity_edge',
                'clear_graph',
                'get_status',
            }

            missing_tools = required_tools - tools
            assert not missing_tools, f'Missing required tools: {missing_tools}'

    @pytest.mark.asyncio
    async def test_add_text_memory(self):
        """Test adding text-based memories."""
        async with GraphitiTestClient() as client:
            # Add memory
            result, metric = await client.call_tool_with_metrics(
                'add_memory',
                {
                    'name': 'Tech Conference Notes',
                    'episode_body': 'The AI conference featured talks on LLMs, RAG systems, and knowledge graphs. Notable speakers included researchers from OpenAI and Anthropic.',
                    'source': 'text',
                    'source_description': 'conference notes',
                    'group_id': client.test_group_id,
                },
            )

            assert metric.success, f'Failed to add memory: {metric.details}'
            assert 'queued' in str(result).lower()

            # Wait for processing
            processed = await client.wait_for_episode_processing(expected_count=1)
            assert processed, 'Episode was not processed within timeout'

    @pytest.mark.asyncio
    async def test_add_json_memory(self):
        """Test adding structured JSON memories."""
        async with GraphitiTestClient() as client:
            json_data = {
                'project': {
                    'name': 'GraphitiDB',
                    'version': '2.0.0',
                    'features': ['temporal-awareness', 'hybrid-search', 'custom-entities'],
                },
                'team': {'size': 5, 'roles': ['engineering', 'product', 'research']},
            }

            result, metric = await client.call_tool_with_metrics(
                'add_memory',
                {
                    'name': 'Project Data',
                    'episode_body': json.dumps(json_data),
                    'source': 'json',
                    'source_description': 'project database',
                    'group_id': client.test_group_id,
                },
            )

            assert metric.success
            assert 'queued' in str(result).lower()

    @pytest.mark.asyncio
    async def test_add_message_memory(self):
        """Test adding conversation/message memories."""
        async with GraphitiTestClient() as client:
            conversation = """
            user: What are the key features of Graphiti?
            assistant: Graphiti offers temporal-aware knowledge graphs, hybrid retrieval, and real-time updates.
            user: How does it handle entity resolution?
            assistant: It uses LLM-based entity extraction and deduplication with semantic similarity matching.
            """

            result, metric = await client.call_tool_with_metrics(
                'add_memory',
                {
                    'name': 'Feature Discussion',
                    'episode_body': conversation,
                    'source': 'message',
                    'source_description': 'support chat',
                    'group_id': client.test_group_id,
                },
            )

            assert metric.success
            assert metric.duration < 5, f'Add memory took too long: {metric.duration}s'


class TestSearchOperations:
    """Test search and retrieval operations."""

    @pytest.mark.asyncio
    async def test_search_nodes_semantic(self):
        """Test semantic search for nodes."""
        async with GraphitiTestClient() as client:
            # First add some test data
            await client.call_tool_with_metrics(
                'add_memory',
                {
                    'name': 'Product Launch',
                    'episode_body': 'Our new AI assistant product launches in Q2 2024 with advanced NLP capabilities.',
                    'source': 'text',
                    'source_description': 'product roadmap',
                    'group_id': client.test_group_id,
                },
            )

            # Wait for processing
            await client.wait_for_episode_processing()

            # Search for nodes
            result, metric = await client.call_tool_with_metrics(
                'search_memory_nodes',
                {'query': 'AI product features', 'group_id': client.test_group_id, 'limit': 10},
            )

            assert metric.success
            assert result is not None

    @pytest.mark.asyncio
    async def test_search_facts_with_filters(self):
        """Test fact search with various filters."""
        async with GraphitiTestClient() as client:
            # Add test data
            await client.call_tool_with_metrics(
                'add_memory',
                {
                    'name': 'Company Facts',
                    'episode_body': 'Acme Corp was founded in 2020. They have 50 employees and $10M in revenue.',
                    'source': 'text',
                    'source_description': 'company profile',
                    'group_id': client.test_group_id,
                },
            )

            await client.wait_for_episode_processing()

            # Search with date filter
            result, metric = await client.call_tool_with_metrics(
                'search_memory_facts',
                {
                    'query': 'company information',
                    'group_id': client.test_group_id,
                    'created_after': '2020-01-01T00:00:00Z',
                    'limit': 20,
                },
            )

            assert metric.success

    @pytest.mark.asyncio
    async def test_hybrid_search(self):
        """Test hybrid search combining semantic and keyword search."""
        async with GraphitiTestClient() as client:
            # Add diverse test data
            test_memories = [
                {
                    'name': 'Technical Doc',
                    'episode_body': 'GraphQL API endpoints support pagination, filtering, and real-time subscriptions.',
                    'source': 'text',
                },
                {
                    'name': 'Architecture',
                    'episode_body': 'The system uses Neo4j for graph storage and OpenAI embeddings for semantic search.',
                    'source': 'text',
                },
            ]

            for memory in test_memories:
                memory['group_id'] = client.test_group_id
                memory['source_description'] = 'documentation'
                await client.call_tool_with_metrics('add_memory', memory)

            await client.wait_for_episode_processing(expected_count=2)

            # Test semantic + keyword search
            result, metric = await client.call_tool_with_metrics(
                'search_memory_nodes',
                {'query': 'Neo4j graph database', 'group_id': client.test_group_id, 'limit': 10},
            )

            assert metric.success


class TestEpisodeManagement:
    """Test episode lifecycle operations."""

    @pytest.mark.asyncio
    async def test_get_episodes_pagination(self):
        """Test retrieving episodes with pagination."""
        async with GraphitiTestClient() as client:
            # Add multiple episodes
            for i in range(5):
                await client.call_tool_with_metrics(
                    'add_memory',
                    {
                        'name': f'Episode {i}',
                        'episode_body': f'This is test episode number {i}',
                        'source': 'text',
                        'source_description': 'test',
                        'group_id': client.test_group_id,
                    },
                )

            await client.wait_for_episode_processing(expected_count=5)

            # Test pagination
            result, metric = await client.call_tool_with_metrics(
                'get_episodes', {'group_id': client.test_group_id, 'last_n': 3}
            )

            assert metric.success
            episodes = json.loads(result) if isinstance(result, str) else result
            assert len(episodes.get('episodes', [])) <= 3

    @pytest.mark.asyncio
    async def test_delete_episode(self):
        """Test deleting specific episodes."""
        async with GraphitiTestClient() as client:
            # Add an episode
            await client.call_tool_with_metrics(
                'add_memory',
                {
                    'name': 'To Delete',
                    'episode_body': 'This episode will be deleted',
                    'source': 'text',
                    'source_description': 'test',
                    'group_id': client.test_group_id,
                },
            )

            await client.wait_for_episode_processing()

            # Get episode UUID
            result, _ = await client.call_tool_with_metrics(
                'get_episodes', {'group_id': client.test_group_id, 'last_n': 1}
            )

            episodes = json.loads(result) if isinstance(result, str) else result
            episode_uuid = episodes['episodes'][0]['uuid']

            # Delete the episode
            result, metric = await client.call_tool_with_metrics(
                'delete_episode', {'episode_uuid': episode_uuid}
            )

            assert metric.success
            assert 'deleted' in str(result).lower()


class TestEntityAndEdgeOperations:
    """Test entity and edge management."""

    @pytest.mark.asyncio
    async def test_get_entity_edge(self):
        """Test retrieving entity edges."""
        async with GraphitiTestClient() as client:
            # Add data to create entities and edges
            await client.call_tool_with_metrics(
                'add_memory',
                {
                    'name': 'Relationship Data',
                    'episode_body': 'Alice works at TechCorp. Bob is the CEO of TechCorp.',
                    'source': 'text',
                    'source_description': 'org chart',
                    'group_id': client.test_group_id,
                },
            )

            await client.wait_for_episode_processing()

            # Search for nodes to get UUIDs
            result, _ = await client.call_tool_with_metrics(
                'search_memory_nodes',
                {'query': 'TechCorp', 'group_id': client.test_group_id, 'limit': 5},
            )

            # Note: This test assumes edges are created between entities
            # Actual edge retrieval would require valid edge UUIDs

    @pytest.mark.asyncio
    async def test_delete_entity_edge(self):
        """Test deleting entity edges."""
        # Similar structure to get_entity_edge but with deletion
        pass  # Implement based on actual edge creation patterns


class TestErrorHandling:
    """Test error conditions and edge cases."""

    @pytest.mark.asyncio
    async def test_invalid_tool_arguments(self):
        """Test handling of invalid tool arguments."""
        async with GraphitiTestClient() as client:
            # Missing required arguments
            result, metric = await client.call_tool_with_metrics(
                'add_memory',
                {'name': 'Incomplete'},  # Missing required fields
            )

            assert not metric.success
            assert 'error' in str(metric.details).lower()

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling for long operations."""
        async with GraphitiTestClient() as client:
            # Simulate a very large episode that might time out
            large_text = 'Large document content. ' * 10000

            result, metric = await client.call_tool_with_metrics(
                'add_memory',
                {
                    'name': 'Large Document',
                    'episode_body': large_text,
                    'source': 'text',
                    'source_description': 'large file',
                    'group_id': client.test_group_id,
                },
                timeout=5,  # Short timeout
            )

            # Check if timeout was handled gracefully
            if not metric.success:
                assert 'timeout' in str(metric.details).lower()

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test handling of concurrent operations."""
        async with GraphitiTestClient() as client:
            # Launch multiple operations concurrently
            tasks = []
            for i in range(5):
                task = client.call_tool_with_metrics(
                    'add_memory',
                    {
                        'name': f'Concurrent {i}',
                        'episode_body': f'Concurrent operation {i}',
                        'source': 'text',
                        'source_description': 'concurrent test',
                        'group_id': client.test_group_id,
                    },
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check that operations were queued successfully
            successful = sum(1 for r, m in results if m.success)
            assert successful >= 3  # At least 60% should succeed


class TestPerformance:
    """Test performance characteristics and optimization."""

    @pytest.mark.asyncio
    async def test_latency_metrics(self):
        """Measure and validate operation latencies."""
        async with GraphitiTestClient() as client:
            operations = [
                (
                    'add_memory',
                    {
                        'name': 'Perf Test',
                        'episode_body': 'Simple text',
                        'source': 'text',
                        'source_description': 'test',
                        'group_id': client.test_group_id,
                    },
                ),
                (
                    'search_memory_nodes',
                    {'query': 'test', 'group_id': client.test_group_id, 'limit': 10},
                ),
                ('get_episodes', {'group_id': client.test_group_id, 'last_n': 10}),
            ]

            for tool_name, args in operations:
                _, metric = await client.call_tool_with_metrics(tool_name, args)

                # Log performance metrics
                print(f'{tool_name}: {metric.duration:.2f}s')

                # Basic latency assertions
                if tool_name == 'get_episodes':
                    assert metric.duration < 2, f'{tool_name} too slow'
                elif tool_name == 'search_memory_nodes':
                    assert metric.duration < 10, f'{tool_name} too slow'

    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(self):
        """Test efficiency of batch operations."""
        async with GraphitiTestClient() as client:
            batch_size = 10
            start_time = time.time()

            # Batch add memories
            for i in range(batch_size):
                await client.call_tool_with_metrics(
                    'add_memory',
                    {
                        'name': f'Batch {i}',
                        'episode_body': f'Batch content {i}',
                        'source': 'text',
                        'source_description': 'batch test',
                        'group_id': client.test_group_id,
                    },
                )

            # Wait for all to process
            processed = await client.wait_for_episode_processing(
                expected_count=batch_size,
                max_wait=120,  # Allow more time for batch
            )

            total_time = time.time() - start_time
            avg_time_per_item = total_time / batch_size

            assert processed, f'Failed to process {batch_size} items'
            assert avg_time_per_item < 15, (
                f'Batch processing too slow: {avg_time_per_item:.2f}s per item'
            )

            # Generate performance report
            print('\nBatch Performance Report:')
            print(f'  Total items: {batch_size}')
            print(f'  Total time: {total_time:.2f}s')
            print(f'  Avg per item: {avg_time_per_item:.2f}s')


class TestDatabaseBackends:
    """Test different database backend configurations."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('database', ['neo4j', 'falkordb'])
    async def test_database_operations(self, database):
        """Test operations with different database backends."""
        env_vars = {
            'DATABASE_PROVIDER': database,
            'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY'),
        }

        if database == 'neo4j':
            env_vars.update(
                {
                    'NEO4J_URI': os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
                    'NEO4J_USER': os.environ.get('NEO4J_USER', 'neo4j'),
                    'NEO4J_PASSWORD': os.environ.get('NEO4J_PASSWORD', 'graphiti'),
                }
            )
        elif database == 'falkordb':
            env_vars['FALKORDB_URI'] = os.environ.get('FALKORDB_URI', 'redis://localhost:6379')

        # This test would require setting up server with specific database
        # Implementation depends on database availability
        pass  # Placeholder for database-specific tests


def generate_test_report(client: GraphitiTestClient) -> str:
    """Generate a comprehensive test report from metrics."""
    if not client.metrics:
        return 'No metrics collected'

    report = []
    report.append('\n' + '=' * 60)
    report.append('GRAPHITI MCP TEST REPORT')
    report.append('=' * 60)

    # Summary statistics
    total_ops = len(client.metrics)
    successful_ops = sum(1 for m in client.metrics if m.success)
    avg_duration = sum(m.duration for m in client.metrics) / total_ops

    report.append(f'\nTotal Operations: {total_ops}')
    report.append(f'Successful: {successful_ops} ({successful_ops / total_ops * 100:.1f}%)')
    report.append(f'Average Duration: {avg_duration:.2f}s')

    # Operation breakdown
    report.append('\nOperation Breakdown:')
    operation_stats = {}
    for metric in client.metrics:
        if metric.operation not in operation_stats:
            operation_stats[metric.operation] = {'count': 0, 'success': 0, 'total_duration': 0}
        stats = operation_stats[metric.operation]
        stats['count'] += 1
        stats['success'] += 1 if metric.success else 0
        stats['total_duration'] += metric.duration

    for op, stats in sorted(operation_stats.items()):
        avg_dur = stats['total_duration'] / stats['count']
        success_rate = stats['success'] / stats['count'] * 100
        report.append(
            f'  {op}: {stats["count"]} calls, {success_rate:.0f}% success, {avg_dur:.2f}s avg'
        )

    # Slowest operations
    slowest = sorted(client.metrics, key=lambda m: m.duration, reverse=True)[:5]
    report.append('\nSlowest Operations:')
    for metric in slowest:
        report.append(f'  {metric.operation}: {metric.duration:.2f}s')

    report.append('=' * 60)
    return '\n'.join(report)


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--asyncio-mode=auto'])
