#!/usr/bin/env python3
"""Tests for SmartMemoryWriter."""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graphiti_core.nodes import EpisodeType

from classifiers.base import ClassificationResult, MemoryCategory
from classifiers.rule_based import RuleBasedClassifier
from services.smart_writer import SmartMemoryWriter, WriteResult
from utils.project_config import ProjectConfig


def create_test_config(shared_group_ids=None, shared_entity_types=None):
    """Helper to create test ProjectConfig."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config_data = {'group_id': 'test-project'}
        if shared_group_ids:
            config_data['shared_group_ids'] = shared_group_ids
        if shared_entity_types:
            config_data['shared_entity_types'] = shared_entity_types
        json.dump(config_data, f)
        config_path = Path(f.name)

    return ProjectConfig(
        group_id='test-project',
        config_path=config_path,
        shared_group_ids=shared_group_ids or [],
        shared_entity_types=shared_entity_types or [],
    )


def test_write_result_creation():
    """Test WriteResult creation."""
    print('Test: WriteResult creation')
    result = WriteResult(success=True, written_to=['group1', 'group2'], category='shared')
    assert result.success is True
    assert result.written_to == ['group1', 'group2']
    assert result.category == 'shared'
    print('  ✓ WriteResult created successfully')


def test_write_result_failure():
    """Test WriteResult with failure."""
    print('\nTest: WriteResult with failure')
    result = WriteResult(success=False, written_to=[], category='unknown', error='Test error')
    assert result.success is False
    assert result.error == 'Test error'
    print('  ✓ WriteResult failure created correctly')


def test_smart_writer_initialization():
    """Test SmartMemoryWriter initialization."""
    print('\nTest: SmartMemoryWriter initialization')
    classifier = RuleBasedClassifier()
    mock_client = MagicMock()

    writer = SmartMemoryWriter(
        classifier=classifier, graphiti_client=mock_client, queue_service=AsyncMock()
    )

    assert writer.classifier == classifier
    assert writer.graphiti_client == mock_client
    print('  ✓ SmartMemoryWriter initialized correctly')


def test_should_use_smart_writer():
    """Test should_use_smart_writer method."""
    print('\nTest: should_use_smart_writer method')
    classifier = RuleBasedClassifier()
    mock_client = MagicMock()
    writer = SmartMemoryWriter(classifier, mock_client, queue_service=AsyncMock())

    # Config with shared groups
    config_with_shared = create_test_config(
        shared_group_ids=['shared1'], shared_entity_types=['Preference']
    )
    assert writer.should_use_smart_writer(config_with_shared) is True

    # Config without shared groups
    config_without_shared = create_test_config()
    assert writer.should_use_smart_writer(config_without_shared) is False

    print('  ✓ should_use_smart_writer works correctly')


def test_add_memory_shared():
    """Shared memory routes to every shared group via the queue (background design)."""
    classifier = RuleBasedClassifier()
    mock_client = AsyncMock()
    mock_queue = AsyncMock()
    writer = SmartMemoryWriter(classifier, mock_client, queue_service=mock_queue)

    config = create_test_config(
        shared_group_ids=['user-common'], shared_entity_types=['Preference']
    )

    async def run_test():
        result = await writer.add_memory(
            name='User preference',
            episode_body='User preference: 4-space indentation',
            project_config=config,
        )
        # add_memory returns immediately with a task id; classification is background
        assert result.success is True
        assert result.task_id is not None

        # Drive the background classification/routing deterministically
        await writer._classify_and_queue(
            name='User preference',
            episode_body='User preference: 4-space indentation',
            project_config=config,
            metadata=None,
            uuid=None,
            task_id='t-shared',
        )

        mock_queue.add_episode.assert_called_once()
        kwargs = mock_queue.add_episode.call_args.kwargs
        assert kwargs['group_id'] == 'user-common'
        assert kwargs['name'] == 'User preference'
        assert '4-space indentation' in kwargs['content']

        print('  ✓ Shared memory queued for shared group')

    asyncio.run(run_test())


def test_add_memory_project_specific():
    """Project-specific memory routes to the project group via the queue."""
    classifier = RuleBasedClassifier()
    mock_client = AsyncMock()
    mock_queue = AsyncMock()
    writer = SmartMemoryWriter(classifier, mock_client, queue_service=mock_queue)

    config = create_test_config(
        shared_group_ids=['user-common'], shared_entity_types=['Preference']
    )

    async def run_test():
        await writer._classify_and_queue(
            name='API config',
            episode_body='The API endpoint is at /api/v1/users',
            project_config=config,
            metadata=None,
            uuid=None,
            task_id='t-project',
        )

        mock_queue.add_episode.assert_called_once()
        assert mock_queue.add_episode.call_args.kwargs['group_id'] == 'test-project'

        print('  ✓ Project-specific memory queued for project group')

    asyncio.run(run_test())


def test_add_memory_multiple_shared_groups():
    """Shared memory fans out to every configured shared group."""
    classifier = RuleBasedClassifier()
    mock_client = AsyncMock()
    mock_queue = AsyncMock()
    writer = SmartMemoryWriter(classifier, mock_client, queue_service=mock_queue)

    config = create_test_config(
        shared_group_ids=['user-common', 'team-standards'], shared_entity_types=['Preference']
    )

    async def run_test():
        await writer._classify_and_queue(
            name='Shared preference',
            episode_body='User preference: dark theme',
            project_config=config,
            metadata=None,
            uuid=None,
            task_id='t-multi',
        )

        assert mock_queue.add_episode.call_count == 2
        routed = {c.kwargs['group_id'] for c in mock_queue.add_episode.call_args_list}
        assert routed == {'user-common', 'team-standards'}

        print('  ✓ Shared memory queued for every shared group')

    asyncio.run(run_test())


def test_add_memory_with_error():
    """A failing classifier is absorbed by the background task; nothing is queued."""
    mock_classifier = MagicMock()
    mock_classifier.classify = AsyncMock(side_effect=Exception('LLM unavailable'))
    mock_client = AsyncMock()
    mock_queue = AsyncMock()
    writer = SmartMemoryWriter(mock_classifier, mock_client, queue_service=mock_queue)

    config = create_test_config(
        shared_group_ids=['user-common'], shared_entity_types=['Preference']
    )

    async def run_test():
        result = await writer.add_memory(
            name='Test', episode_body='Test content', project_config=config
        )
        assert result.success is True  # fire-and-forget: intake still succeeds

        # The background pipeline fails safe: classifier failure queues the
        # episode to the project group instead of dropping it
        await writer._classify_and_queue(
            name='Test',
            episode_body='Test content',
            project_config=config,
            metadata=None,
            uuid=None,
            task_id='t-error',
        )
        mock_queue.add_episode.assert_called_once()
        kwargs = mock_queue.add_episode.call_args.kwargs
        assert kwargs['group_id'] == 'test-project'
        assert kwargs['content'] == 'Test content'

        print('  ✓ Classifier failure fails safe into the project group')

    asyncio.run(run_test())


def test_write_to_group_params():
    """_queue_for_group passes the expected parameters to the queue service."""
    classifier = RuleBasedClassifier()
    mock_client = AsyncMock()
    mock_queue = AsyncMock()
    writer = SmartMemoryWriter(classifier, mock_client, queue_service=mock_queue)

    config = create_test_config(shared_group_ids=['shared1'], shared_entity_types=['Preference'])

    async def run_test():
        await writer._classify_and_queue(
            name='Test memory',
            episode_body='User preference: test content',  # contains "preference" -> SHARED
            project_config=config,
            metadata={'timestamp': '2024-01-01', 'source': 'test'},
            uuid=None,
            task_id='t-params',
        )

        mock_queue.add_episode.assert_called_once()
        kwargs = mock_queue.add_episode.call_args.kwargs
        assert kwargs['name'] == 'Test memory'
        assert kwargs['content'] == 'User preference: test content'
        assert kwargs['group_id'] == 'shared1'
        assert kwargs['source_description'] == 'Smart Memory Writer'
        assert kwargs['uuid'] is None
        # 'test' is not a known EpisodeType: falls back to text
        assert kwargs['episode_type'] == EpisodeType.text

        print('  ✓ Parameters passed correctly')

    asyncio.run(run_test())


def test_add_memory_mixed_with_split_content():
    """MIXED memory queues the shared part to shared groups and the project part to the project."""
    mock_classifier = MagicMock()
    mock_classifier.classify = AsyncMock(
        return_value=ClassificationResult(
            category=MemoryCategory.MIXED,
            confidence=0.8,
            reasoning='Contains both shared and project-specific content',
            shared_part='User prefers dark mode for all projects',
            project_part='Project uses React with TypeScript at /api/v1/users',
        )
    )
    mock_client = AsyncMock()
    mock_queue = AsyncMock()
    writer = SmartMemoryWriter(mock_classifier, mock_client, queue_service=mock_queue)

    config = create_test_config(
        shared_group_ids=['user-common'], shared_entity_types=['Preference']
    )

    async def run_test():
        await writer._classify_and_queue(
            name='Mixed memory',
            episode_body='User prefers dark mode. Project uses React at /api/v1/users.',
            project_config=config,
            metadata=None,
            uuid=None,
            task_id='t-mixed-split',
        )

        assert mock_queue.add_episode.call_count == 2
        calls = mock_queue.add_episode.call_args_list

        shared_kwargs = calls[0].kwargs
        assert shared_kwargs['group_id'] == 'user-common'
        assert 'dark mode for all projects' in shared_kwargs['content']
        assert 'React' not in shared_kwargs['content']

        project_kwargs = calls[1].kwargs
        assert project_kwargs['group_id'] == 'test-project'
        assert 'React with TypeScript' in project_kwargs['content']

        print('  ✓ MIXED memory split across shared and project groups')

    asyncio.run(run_test())


def test_add_memory_mixed_without_split_content():
    """MIXED memory without split parts queues the full content to both targets."""
    mock_classifier = MagicMock()
    mock_classifier.classify = AsyncMock(
        return_value=ClassificationResult(
            category=MemoryCategory.MIXED,
            confidence=0.8,
            reasoning='Contains both shared and project-specific content',
            shared_part='',
            project_part='',
        )
    )
    mock_client = AsyncMock()
    mock_queue = AsyncMock()
    writer = SmartMemoryWriter(mock_classifier, mock_client, queue_service=mock_queue)

    config = create_test_config(
        shared_group_ids=['user-common'], shared_entity_types=['Preference']
    )

    async def run_test():
        await writer._classify_and_queue(
            name='Mixed memory',
            episode_body='User prefers dark mode. Project uses React.',
            project_config=config,
            metadata=None,
            uuid=None,
            task_id='t-mixed-full',
        )

        assert mock_queue.add_episode.call_count == 2
        for call in mock_queue.add_episode.call_args_list:
            assert 'User prefers dark mode. Project uses React.' in call.kwargs['content']

        print('  ✓ MIXED memory without split falls back to full content')

    asyncio.run(run_test())


def run_all_tests():
    """Run all tests."""
    print('=' * 60)
    print('Running SmartMemoryWriter Tests')
    print('=' * 60)

    tests = [
        test_write_result_creation,
        test_write_result_failure,
        test_smart_writer_initialization,
        test_should_use_smart_writer,
        test_add_memory_shared,
        test_add_memory_project_specific,
        test_add_memory_multiple_shared_groups,
        test_write_to_group_params,
        test_add_memory_mixed_with_split_content,
        test_add_memory_mixed_without_split_content,
        # test_add_memory_with_error,  # Temporarily disabled due to test isolation issues
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


def test_fail_safe_with_caller_uuid_when_classifier_crashes():
    """Classifier crash + caller uuid: fail-safe must not NameError on fanout."""
    from unittest.mock import AsyncMock, MagicMock

    mock_classifier = MagicMock()
    mock_classifier.classify = AsyncMock(side_effect=Exception('LLM down'))
    mock_client = AsyncMock()
    mock_queue = AsyncMock()
    writer = SmartMemoryWriter(mock_classifier, mock_client, queue_service=mock_queue)

    config = create_test_config(
        shared_group_ids=['user-common'], shared_entity_types=['Preference']
    )

    async def run():
        await writer._classify_and_queue(
            name='U',
            episode_body='content',
            project_config=config,
            metadata=None,
            uuid='caller-uuid-1',
            task_id='t-nameerror',
        )
        mock_queue.add_episode.assert_called_once()
        assert mock_queue.add_episode.call_args.kwargs['uuid'] == 'caller-uuid-1'

    asyncio.run(run())
    print('  ✓ fail-safe keeps caller uuid, no NameError')
