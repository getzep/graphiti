#!/usr/bin/env python3
"""
Asynchronous operation tests for Graphiti MCP Server.
Tests concurrent operations, queue management, and async patterns.
"""

import asyncio
import contextlib
import json
import time

import pytest
from test_fixtures import (
    TestDataGenerator,
    graphiti_test_client,
)


class TestAsyncQueueManagement:
    """Test asynchronous queue operations and episode processing."""

    @pytest.mark.asyncio
    async def test_sequential_queue_processing(self):
        """Verify episodes are processed sequentially within a group."""
        async with graphiti_test_client() as (session, group_id):
            # Add multiple episodes quickly
            episodes = []
            for i in range(5):
                result = await session.call_tool(
                    'add_memory',
                    {
                        'name': f'Sequential Test {i}',
                        'episode_body': f'Episode {i} with timestamp {time.time()}',
                        'source': 'text',
                        'source_description': 'sequential test',
                        'group_id': group_id,
                        'reference_id': f'seq_{i}',  # Add reference for tracking
                    },
                )
                episodes.append(result)

            # Wait for processing
            await asyncio.sleep(10)  # Allow time for sequential processing

            # Retrieve episodes and verify order
            result = await session.call_tool('get_episodes', {'group_id': group_id, 'last_n': 10})

            processed_episodes = json.loads(result.content[0].text)['episodes']

            # Verify all episodes were processed
            assert len(processed_episodes) >= 5, (
                f'Expected at least 5 episodes, got {len(processed_episodes)}'
            )

            # Verify sequential processing (timestamps should be ordered)
            timestamps = [ep.get('created_at') for ep in processed_episodes]
            assert timestamps == sorted(timestamps), 'Episodes not processed in order'

    @pytest.mark.asyncio
    async def test_concurrent_group_processing(self):
        """Test that different groups can process concurrently."""
        async with graphiti_test_client() as (session, _):
            groups = [f'group_{i}_{time.time()}' for i in range(3)]
            tasks = []

            # Create tasks for different groups
            for group_id in groups:
                for j in range(2):
                    task = session.call_tool(
                        'add_memory',
                        {
                            'name': f'Group {group_id} Episode {j}',
                            'episode_body': f'Content for {group_id}',
                            'source': 'text',
                            'source_description': 'concurrent test',
                            'group_id': group_id,
                        },
                    )
                    tasks.append(task)

            # Execute all tasks concurrently
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            execution_time = time.time() - start_time

            # Verify all succeeded
            failures = [r for r in results if isinstance(r, Exception)]
            assert not failures, f'Concurrent operations failed: {failures}'

            # Check that execution was actually concurrent (should be faster than sequential)
            # Sequential would take at least 6 * processing_time
            assert execution_time < 30, f'Concurrent execution too slow: {execution_time}s'

    @pytest.mark.asyncio
    async def test_queue_overflow_handling(self):
        """Test behavior when queue reaches capacity."""
        async with graphiti_test_client() as (session, group_id):
            # Attempt to add many episodes rapidly
            tasks = []
            for i in range(100):  # Large number to potentially overflow
                task = session.call_tool(
                    'add_memory',
                    {
                        'name': f'Overflow Test {i}',
                        'episode_body': f'Episode {i}',
                        'source': 'text',
                        'source_description': 'overflow test',
                        'group_id': group_id,
                    },
                )
                tasks.append(task)

            # Execute with gathering to catch any failures
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successful queuing
            successful = sum(1 for r in results if not isinstance(r, Exception))

            # Should handle overflow gracefully
            assert successful > 0, 'No episodes were queued successfully'

            # Log overflow behavior
            if successful < 100:
                print(f'Queue overflow: {successful}/100 episodes queued')


class TestConcurrentOperations:
    """Test concurrent tool calls and operations."""

    @pytest.mark.asyncio
    async def test_concurrent_search_operations(self):
        """Test multiple concurrent search operations."""
        async with graphiti_test_client() as (session, group_id):
            # First, add some test data
            data_gen = TestDataGenerator()

            add_tasks = []
            for _ in range(5):
                task = session.call_tool(
                    'add_memory',
                    {
                        'name': 'Search Test Data',
                        'episode_body': data_gen.generate_technical_document(),
                        'source': 'text',
                        'source_description': 'search test',
                        'group_id': group_id,
                    },
                )
                add_tasks.append(task)

            await asyncio.gather(*add_tasks)
            await asyncio.sleep(15)  # Wait for processing

            # Now perform concurrent searches
            search_queries = [
                'architecture',
                'performance',
                'implementation',
                'dependencies',
                'latency',
            ]

            search_tasks = []
            for query in search_queries:
                task = session.call_tool(
                    'search_memory_nodes',
                    {
                        'query': query,
                        'group_id': group_id,
                        'limit': 10,
                    },
                )
                search_tasks.append(task)

            start_time = time.time()
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            search_time = time.time() - start_time

            # Verify all searches completed
            failures = [r for r in results if isinstance(r, Exception)]
            assert not failures, f'Search operations failed: {failures}'

            # Verify concurrent execution efficiency
            assert search_time < len(search_queries) * 2, 'Searches not executing concurrently'

    @pytest.mark.asyncio
    async def test_mixed_operation_concurrency(self):
        """Test different types of operations running concurrently."""
        async with graphiti_test_client() as (session, group_id):
            operations = []

            # Add memory operation
            operations.append(
                session.call_tool(
                    'add_memory',
                    {
                        'name': 'Mixed Op Test',
                        'episode_body': 'Testing mixed operations',
                        'source': 'text',
                        'source_description': 'test',
                        'group_id': group_id,
                    },
                )
            )

            # Search operation
            operations.append(
                session.call_tool(
                    'search_memory_nodes',
                    {
                        'query': 'test',
                        'group_id': group_id,
                        'limit': 5,
                    },
                )
            )

            # Get episodes operation
            operations.append(
                session.call_tool(
                    'get_episodes',
                    {
                        'group_id': group_id,
                        'last_n': 10,
                    },
                )
            )

            # Get status operation
            operations.append(session.call_tool('get_status', {}))

            # Execute all concurrently
            results = await asyncio.gather(*operations, return_exceptions=True)

            # Check results
            for i, result in enumerate(results):
                assert not isinstance(result, Exception), f'Operation {i} failed: {result}'


class TestAsyncErrorHandling:
    """Test async error handling and recovery."""

    @pytest.mark.asyncio
    async def test_timeout_recovery(self):
        """Test recovery from operation timeouts."""
        async with graphiti_test_client() as (session, group_id):
            # Create a very large episode that might time out
            large_content = 'x' * 1000000  # 1MB of data

            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    session.call_tool(
                        'add_memory',
                        {
                            'name': 'Timeout Test',
                            'episode_body': large_content,
                            'source': 'text',
                            'source_description': 'timeout test',
                            'group_id': group_id,
                        },
                    ),
                    timeout=2.0,  # Short timeout - expected to timeout
                )

            # Verify server is still responsive after timeout
            status_result = await session.call_tool('get_status', {})
            assert status_result is not None, 'Server unresponsive after timeout'

    @pytest.mark.asyncio
    async def test_cancellation_handling(self):
        """Test proper handling of cancelled operations."""
        async with graphiti_test_client() as (session, group_id):
            # Start a long-running operation
            task = asyncio.create_task(
                session.call_tool(
                    'add_memory',
                    {
                        'name': 'Cancellation Test',
                        'episode_body': TestDataGenerator.generate_technical_document(),
                        'source': 'text',
                        'source_description': 'cancel test',
                        'group_id': group_id,
                    },
                )
            )

            # Cancel after a short delay
            await asyncio.sleep(0.1)
            task.cancel()

            # Verify cancellation was handled
            with pytest.raises(asyncio.CancelledError):
                await task

            # Server should still be operational
            result = await session.call_tool('get_status', {})
            assert result is not None

    @pytest.mark.asyncio
    async def test_exception_propagation(self):
        """Test that exceptions are properly propagated in async context."""
        async with graphiti_test_client() as (session, group_id):
            # Call with invalid arguments
            with pytest.raises(ValueError):
                await session.call_tool(
                    'add_memory',
                    {
                        # Missing required fields
                        'group_id': group_id,
                    },
                )

            # Server should remain operational
            status = await session.call_tool('get_status', {})
            assert status is not None


class TestAsyncPerformance:
    """Performance tests for async operations."""

    @pytest.mark.asyncio
    async def test_async_throughput(self, performance_benchmark):
        """Measure throughput of async operations."""
        async with graphiti_test_client() as (session, group_id):
            num_operations = 50
            start_time = time.time()

            # Create many concurrent operations
            tasks = []
            for i in range(num_operations):
                task = session.call_tool(
                    'add_memory',
                    {
                        'name': f'Throughput Test {i}',
                        'episode_body': f'Content {i}',
                        'source': 'text',
                        'source_description': 'throughput test',
                        'group_id': group_id,
                    },
                )
                tasks.append(task)

            # Execute all
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time

            # Calculate metrics
            successful = sum(1 for r in results if not isinstance(r, Exception))
            throughput = successful / total_time

            performance_benchmark.record('async_throughput', throughput)

            # Log results
            print('\nAsync Throughput Test:')
            print(f'  Operations: {num_operations}')
            print(f'  Successful: {successful}')
            print(f'  Total time: {total_time:.2f}s')
            print(f'  Throughput: {throughput:.2f} ops/s')

            # Assert minimum throughput
            assert throughput > 1.0, f'Throughput too low: {throughput:.2f} ops/s'

    @pytest.mark.asyncio
    async def test_latency_under_load(self, performance_benchmark):
        """Test operation latency under concurrent load."""
        async with graphiti_test_client() as (session, group_id):
            # Create background load
            background_tasks = []
            for i in range(10):
                task = asyncio.create_task(
                    session.call_tool(
                        'add_memory',
                        {
                            'name': f'Background {i}',
                            'episode_body': TestDataGenerator.generate_technical_document(),
                            'source': 'text',
                            'source_description': 'background',
                            'group_id': f'background_{group_id}',
                        },
                    )
                )
                background_tasks.append(task)

            # Measure latency of operations under load
            latencies = []
            for _ in range(5):
                start = time.time()
                await session.call_tool('get_status', {})
                latency = time.time() - start
                latencies.append(latency)
                performance_benchmark.record('latency_under_load', latency)

            # Clean up background tasks
            for task in background_tasks:
                task.cancel()

            # Analyze latencies
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)

            print('\nLatency Under Load:')
            print(f'  Average: {avg_latency:.3f}s')
            print(f'  Max: {max_latency:.3f}s')

            # Assert acceptable latency
            assert avg_latency < 2.0, f'Average latency too high: {avg_latency:.3f}s'
            assert max_latency < 5.0, f'Max latency too high: {max_latency:.3f}s'


class TestAsyncStreamHandling:
    """Test handling of streaming responses and data."""

    @pytest.mark.asyncio
    async def test_large_response_streaming(self):
        """Test handling of large streamed responses."""
        async with graphiti_test_client() as (session, group_id):
            # Add many episodes
            for i in range(20):
                await session.call_tool(
                    'add_memory',
                    {
                        'name': f'Stream Test {i}',
                        'episode_body': f'Episode content {i}',
                        'source': 'text',
                        'source_description': 'stream test',
                        'group_id': group_id,
                    },
                )

            # Wait for processing
            await asyncio.sleep(30)

            # Request large result set
            result = await session.call_tool(
                'get_episodes',
                {
                    'group_id': group_id,
                    'last_n': 100,  # Request all
                },
            )

            # Verify response handling
            episodes = json.loads(result.content[0].text)['episodes']
            assert len(episodes) >= 20, f'Expected at least 20 episodes, got {len(episodes)}'

    @pytest.mark.asyncio
    async def test_incremental_processing(self):
        """Test incremental processing of results."""
        async with graphiti_test_client() as (session, group_id):
            # Add episodes incrementally
            for batch in range(3):
                batch_tasks = []
                for i in range(5):
                    task = session.call_tool(
                        'add_memory',
                        {
                            'name': f'Batch {batch} Item {i}',
                            'episode_body': f'Content for batch {batch}',
                            'source': 'text',
                            'source_description': 'incremental test',
                            'group_id': group_id,
                        },
                    )
                    batch_tasks.append(task)

                # Process batch
                await asyncio.gather(*batch_tasks)

                # Wait for this batch to process
                await asyncio.sleep(10)

                # Verify incremental results
                result = await session.call_tool(
                    'get_episodes',
                    {
                        'group_id': group_id,
                        'last_n': 100,
                    },
                )

                episodes = json.loads(result.content[0].text)['episodes']
                expected_min = (batch + 1) * 5
                assert len(episodes) >= expected_min, (
                    f'Batch {batch}: Expected at least {expected_min} episodes'
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--asyncio-mode=auto'])
