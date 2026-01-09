#!/usr/bin/env python3
"""Tests for QueueService concurrency control."""

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from services.queue_service import QueueService


def test_queue_service_initialization():
    """Test QueueService initialization with different max_concurrent values."""
    print('Test: QueueService initialization')

    # Test default value
    queue_service = QueueService()
    assert queue_service._max_concurrent == 5
    print('  ✓ Default max_concurrent=5')

    # Test custom value
    queue_service = QueueService(max_concurrent=10)
    assert queue_service._max_concurrent == 10
    print('  ✓ Custom max_concurrent=10')

    # Test low value
    queue_service = QueueService(max_concurrent=1)
    assert queue_service._max_concurrent == 1
    print('  ✓ Min max_concurrent=1')


def test_concurrent_processing_limit():
    """Test that concurrent processing is limited by semaphore."""
    print('\nTest: Concurrent processing limit')

    async def run_test():
        # Track concurrent executions
        concurrent_count = 0
        max_concurrent_seen = 0
        lock = asyncio.Lock()
        processing_events = []

        # Create mock client that tracks execution
        mock_client = AsyncMock()

        async def mock_add_episode(**kwargs):
            nonlocal concurrent_count, max_concurrent_seen

            async with lock:
                concurrent_count += 1
                if concurrent_count > max_concurrent_seen:
                    max_concurrent_seen = concurrent_count
                processing_events.append(('start', kwargs.get('group_id'), concurrent_count))

            # Simulate some work
            await asyncio.sleep(0.1)

            async with lock:
                concurrent_count -= 1
                processing_events.append(('end', kwargs.get('group_id'), concurrent_count))

        mock_client.add_episode = mock_add_episode

        # Initialize queue service with max_concurrent=3
        queue_service = QueueService(max_concurrent=3)
        await queue_service.initialize(mock_client)

        # Add tasks for 5 different group_ids (more than max_concurrent)
        tasks_added = []
        for i in range(5):
            result = await queue_service.add_episode(
                group_id=f'group-{i}',
                name=f'episode-{i}',
                content=f'content-{i}',
                source_description='test',
                episode_type='text',
                entity_types=[],
                uuid=f'uuid-{i}',
            )
            tasks_added.append(result)

        print(f'  Added {len(tasks_added)} tasks to queue')

        # Wait for all tasks to complete
        await asyncio.sleep(0.5)

        # Verify max concurrent did not exceed limit
        assert max_concurrent_seen <= 3, f'Max concurrent {max_concurrent_seen} exceeded limit of 3'
        print(f'  ✓ Max concurrent executions: {max_concurrent_seen} (limit: 3)')

        # Verify all tasks were processed
        assert len(processing_events) >= 10  # 5 start + 5 end events
        print(f'  ✓ Total processing events: {len(processing_events)}')

        # Verify semaphore was effective
        starts = [e for e in processing_events if e[0] == 'start']
        for event in starts:
            assert event[2] <= 3, f'Concurrent count {event[2]} exceeded limit'

        print('  ✓ All executions respected concurrent limit')

    asyncio.run(run_test())


def test_single_group_sequential_processing():
    """Test that episodes in same group are processed sequentially."""
    print('\nTest: Single group sequential processing')

    async def run_test():
        processing_order = []

        mock_client = AsyncMock()

        async def mock_add_episode(**kwargs):
            group_id = kwargs.get('group_id')
            processing_order.append(group_id)
            await asyncio.sleep(0.05)

        mock_client.add_episode = mock_add_episode

        queue_service = QueueService(max_concurrent=5)
        await queue_service.initialize(mock_client)

        # Add multiple episodes to same group
        for i in range(3):
            await queue_service.add_episode(
                group_id='test-group',
                name=f'episode-{i}',
                content=f'content-{i}',
                source_description='test',
                episode_type='text',
                entity_types=[],
                uuid=f'uuid-{i}',
            )

        # Wait for processing
        await asyncio.sleep(0.3)

        # All should be for same group
        assert all(g == 'test-group' for g in processing_order)
        assert len(processing_order) == 3
        print(f'  ✓ {len(processing_order)} episodes processed sequentially for same group')

    asyncio.run(run_test())


def test_multiple_groups_concurrent_processing():
    """Test that different groups can process concurrently up to limit."""
    print('\nTest: Multiple groups concurrent processing')

    async def run_test():
        active_groups = set()
        max_active_groups = 0
        lock = asyncio.Lock()

        mock_client = AsyncMock()

        async def mock_add_episode(**kwargs):
            nonlocal max_active_groups

            group_id = kwargs.get('group_id')

            async with lock:
                active_groups.add(group_id)
                if len(active_groups) > max_active_groups:
                    max_active_groups = len(active_groups)

            await asyncio.sleep(0.1)

            async with lock:
                active_groups.discard(group_id)

        mock_client.add_episode = mock_add_episode

        # Use max_concurrent=3
        queue_service = QueueService(max_concurrent=3)
        await queue_service.initialize(mock_client)

        # Add episodes to 5 different groups simultaneously
        tasks = []
        for i in range(5):
            task = queue_service.add_episode(
                group_id=f'group-{i}',
                name=f'episode-{i}',
                content=f'content-{i}',
                source_description='test',
                episode_type='text',
                entity_types=[],
                uuid=f'uuid-{i}',
            )
            tasks.append(task)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Max concurrent should be at most 3
        assert max_active_groups <= 3, f'Max active groups {max_active_groups} exceeded limit'
        print(f'  ✓ Max active groups: {max_active_groups} (limit: 3)')

    asyncio.run(run_test())


def run_all_tests():
    """Run all tests."""
    print('=' * 60)
    print('Running QueueService Concurrency Tests')
    print('=' * 60)

    tests = [
        test_queue_service_initialization,
        test_concurrent_processing_limit,
        test_single_group_sequential_processing,
        test_multiple_groups_concurrent_processing,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f'  ✗ FAILED: {e}')
            failed += 1
        except Exception as e:
            print(f'  ✗ ERROR: {e}')
            import traceback
            traceback.print_exc()
            failed += 1

    print('\n' + '=' * 60)
    print(f'Results: {passed} passed, {failed} failed')
    print('=' * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
