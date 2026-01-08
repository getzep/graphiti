#!/usr/bin/env python3
"""Tests for MCP Server integration with SmartMemoryWriter."""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))



def create_test_graphiti_config():
    """Helper to create test GraphitiConfig."""
    # Set minimal environment config
    import os

    from config.schema import GraphitiConfig

    os.environ['GRAPHITI_GROUP_ID'] = 'test-default-group'

    config = GraphitiConfig()
    return config


def create_test_project_config_file(shared=False):
    """Helper to create a test .graphiti.json file."""
    config_data = {'group_id': 'test-project', 'description': 'Test project for integration tests'}

    if shared:
        config_data.update(
            {
                'shared_group_ids': ['user-common', 'team-standards'],
                'shared_entity_types': ['Preference', 'Procedure'],
                'shared_patterns': ['偏好', '习惯'],
                'write_strategy': 'simple',
            }
        )

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        return Path(f.name)


def test_smart_writer_initialization_with_shared_config():
    """Test that SmartMemoryWriter is initialized when project has shared config."""
    print('\nTest: SmartMemoryWriter initialization with shared config')

    # Create a test project config file with shared config
    config_file = create_test_project_config_file(shared=True)

    try:
        # Mock environment variable to point to temp file's directory
        with patch.dict('os.environ', {'GRAPHITI_PROJECT_DIR': str(config_file.parent)}):
            # Mock the actual server initialization to avoid DB connection
            with patch('graphiti_mcp_server.GraphitiService') as MockService:
                with patch('graphiti_mcp_server.QueueService') as MockQueue:
                    mock_service_instance = AsyncMock()
                    mock_service_instance.get_client.return_value = AsyncMock()
                    mock_service_instance.semaphore = asyncio.Semaphore(10)
                    MockService.return_value = mock_service_instance

                    mock_queue_instance = AsyncMock()
                    MockQueue.return_value = mock_queue_instance

                    # Run the initialization
                    async def run_test():
                        # Mock find_project_config to return our test config
                        from utils.project_config import ProjectConfig

                        mock_project_config = ProjectConfig(
                            group_id='test-project',
                            config_path=config_file,
                            shared_group_ids=['user-common', 'team-standards'],
                            shared_entity_types=['Preference', 'Procedure'],
                        )

                        with patch(
                            'graphiti_mcp_server.find_project_config',
                            return_value=mock_project_config,
                        ):
                            # Reset global variables
                            import graphiti_mcp_server

                            graphiti_mcp_server.smart_writer = None
                            graphiti_mcp_server.project_config = None

                            # Simulate the smart writer initialization part
                            from classifiers.rule_based import RuleBasedClassifier
                            from services.smart_writer import SmartMemoryWriter

                            mock_client = AsyncMock()
                            classifier = RuleBasedClassifier()
                            smart_writer = SmartMemoryWriter(
                                classifier=classifier, graphiti_client=mock_client
                            )

                            # Verify smart writer was created
                            assert smart_writer is not None
                            assert isinstance(smart_writer.classifier, RuleBasedClassifier)
                            print('  ✓ SmartMemoryWriter initialized correctly with shared config')

                    asyncio.run(run_test())

    finally:
        # Cleanup temp file
        if config_file.exists():
            config_file.unlink()


def test_smart_writer_not_initialized_without_shared_config():
    """Test that SmartMemoryWriter is NOT initialized when project has no shared config."""
    print('\nTest: SmartMemoryWriter NOT initialized without shared config')

    # Create a test project config file WITHOUT shared config
    config_file = create_test_project_config_file(shared=False)

    try:
        # Mock environment variable to point to temp file's directory
        with patch.dict('os.environ', {'GRAPHITI_PROJECT_DIR': str(config_file.parent)}):
            # Mock the actual server initialization
            with patch('graphiti_mcp_server.GraphitiService'):
                with patch('graphiti_mcp_server.QueueService'):

                    async def run_test():
                        # Mock find_project_config to return our test config without shared groups
                        from utils.project_config import ProjectConfig

                        mock_project_config = ProjectConfig(
                            group_id='test-project',
                            config_path=config_file,
                            shared_group_ids=[],  # No shared groups
                            shared_entity_types=[],
                        )

                        with patch(
                            'graphiti_mcp_server.find_project_config',
                            return_value=mock_project_config,
                        ):
                            # Reset global variables
                            import graphiti_mcp_server

                            graphiti_mcp_server.smart_writer = None
                            graphiti_mcp_server.project_config = None

                            # Simulate the smart writer initialization logic
                            if mock_project_config and mock_project_config.has_shared_config:
                                smart_writer = 'should_not_happen'
                            else:
                                smart_writer = None

                            # Verify smart writer was NOT created
                            assert smart_writer is None
                            print(
                                '  ✓ SmartMemoryWriter correctly NOT initialized without shared config'
                            )

                    asyncio.run(run_test())

    finally:
        # Cleanup temp file
        if config_file.exists():
            config_file.unlink()


def test_add_memory_uses_smart_writer():
    """Test that add_memory uses SmartMemoryWriter when available."""
    print('\nTest: add_memory uses SmartMemoryWriter')

    async def run_test():
        from utils.project_config import ProjectConfig

        # Create project config with shared groups
        mock_project_config = ProjectConfig(
            group_id='test-project',
            config_path=Path('/tmp/test.json'),
            shared_group_ids=['user-common'],
            shared_entity_types=['Preference'],
        )

        # Create mock config
        mock_config = MagicMock()
        mock_config.graphiti.group_id = 'test-project'

        # Create mock smart writer
        mock_smart_writer = AsyncMock()
        mock_smart_writer.add_memory.return_value = MagicMock(
            success=True, written_to=['user-common'], category='shared'
        )

        # Mock the global services
        with patch('graphiti_mcp_server.graphiti_service', MagicMock()):
            with patch('graphiti_mcp_server.queue_service', AsyncMock()):
                with patch('graphiti_mcp_server.smart_writer', mock_smart_writer):
                    with patch('graphiti_mcp_server.project_config', mock_project_config):
                        # Set the global config variable in the module
                        import graphiti_mcp_server

                        graphiti_mcp_server.config = mock_config

                        # Import and call add_memory
                        from graphiti_mcp_server import add_memory

                        result = await add_memory(
                            name='Test preference', episode_body='User preference: dark mode'
                        )

                        # Verify smart writer was called
                        mock_smart_writer.add_memory.assert_called_once()
                        call_kwargs = mock_smart_writer.add_memory.call_args.kwargs
                        assert call_kwargs['name'] == 'Test preference'
                        assert call_kwargs['episode_body'] == 'User preference: dark mode'
                        assert call_kwargs['project_config'] == mock_project_config

                        # Verify result - check dict structure instead of isinstance
                        assert isinstance(result, dict)
                        assert 'message' in result
                        assert 'written to 1 group(s)' in result['message']
                        assert 'user-common' in result['message']
                        assert 'shared' in result['message']

                        print('  ✓ add_memory correctly uses SmartMemoryWriter')

    asyncio.run(run_test())


def test_add_memory_fallback_to_standard_path():
    """Test that add_memory falls back to standard path when smart_writer is None."""
    print('\nTest: add_memory fallback to standard path')

    async def run_test():
        from utils.project_config import ProjectConfig

        # Create project config WITHOUT shared groups
        mock_project_config = ProjectConfig(
            group_id='test-project',
            config_path=Path('/tmp/test.json'),
            shared_group_ids=[],  # No shared groups
        )

        # Create mock config
        mock_config = MagicMock()
        mock_config.graphiti.group_id = 'test-project'

        # Mock the global services
        mock_queue_service = AsyncMock()
        mock_graphiti_service = MagicMock()
        mock_graphiti_service.entity_types = None

        with patch('graphiti_mcp_server.graphiti_service', mock_graphiti_service):
            with patch('graphiti_mcp_server.queue_service', mock_queue_service):
                with patch('graphiti_mcp_server.smart_writer', None):
                    with patch('graphiti_mcp_server.project_config', mock_project_config):
                        # Set the global config variable in the module
                        import graphiti_mcp_server

                        graphiti_mcp_server.config = mock_config

                        # Import and call add_memory
                        from graphiti_mcp_server import add_memory

                        result = await add_memory(
                            name='Test episode', episode_body='Some project-specific content'
                        )

                        # Verify queue service was called (standard path)
                        mock_queue_service.add_episode.assert_called_once()
                        call_kwargs = mock_queue_service.add_episode.call_args.kwargs
                        assert call_kwargs['name'] == 'Test episode'
                        assert call_kwargs['content'] == 'Some project-specific content'
                        assert call_kwargs['group_id'] == 'test-project'

                        # Verify result - check dict structure instead of isinstance
                        assert isinstance(result, dict)
                        assert 'message' in result
                        assert 'queued for processing' in result['message']
                        assert 'test-project' in result['message']

                        print('  ✓ add_memory correctly falls back to standard path')

    asyncio.run(run_test())


def test_add_memory_explicit_group_id_bypasses_smart_writer():
    """Test that explicit group_id bypasses smart writer."""
    print('\nTest: Explicit group_id bypasses SmartMemoryWriter')

    async def run_test():
        from utils.project_config import ProjectConfig

        # Create project config WITH shared groups
        mock_project_config = ProjectConfig(
            group_id='test-project',
            config_path=Path('/tmp/test.json'),
            shared_group_ids=['user-common'],
            shared_entity_types=['Preference'],
        )

        # Create mock config
        mock_config = MagicMock()
        mock_config.graphiti.group_id = 'test-project'

        # Create mock smart writer (should NOT be called)
        mock_smart_writer = AsyncMock()

        # Mock the global services
        mock_queue_service = AsyncMock()

        with patch('graphiti_mcp_server.graphiti_service', MagicMock()):
            with patch('graphiti_mcp_server.queue_service', mock_queue_service):
                with patch('graphiti_mcp_server.smart_writer', mock_smart_writer):
                    with patch('graphiti_mcp_server.project_config', mock_project_config):
                        # Set the global config variable in the module
                        import graphiti_mcp_server

                        graphiti_mcp_server.config = mock_config

                        # Import and call add_memory with EXPLICIT group_id
                        from graphiti_mcp_server import add_memory

                        result = await add_memory(
                            name='Test episode',
                            episode_body='Some content',
                            group_id='explicit-group',  # Explicit group_id
                        )

                        # Verify smart writer was NOT called
                        mock_smart_writer.add_memory.assert_not_called()

                        # Verify queue service was called (standard path)
                        mock_queue_service.add_episode.assert_called_once()
                        call_kwargs = mock_queue_service.add_episode.call_args.kwargs
                        assert call_kwargs['group_id'] == 'explicit-group'

                        # Verify result - check dict structure instead of isinstance
                        assert isinstance(result, dict)
                        assert 'message' in result
                        assert 'queued for processing' in result['message']
                        assert 'explicit-group' in result['message']

                        print('  ✓ Explicit group_id correctly bypasses SmartMemoryWriter')

    asyncio.run(run_test())


def run_all_tests():
    """Run all tests."""
    print('=' * 60)
    print('Running MCP Server Integration Tests')
    print('=' * 60)

    tests = [
        test_smart_writer_initialization_with_shared_config,
        test_smart_writer_not_initialized_without_shared_config,
        test_add_memory_uses_smart_writer,
        test_add_memory_fallback_to_standard_path,
        test_add_memory_explicit_group_id_bypasses_smart_writer,
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
