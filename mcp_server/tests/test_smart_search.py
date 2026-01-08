#!/usr/bin/env python3
"""Tests for Smart Search with auto-include shared groups."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_search_nodes_auto_includes_shared_groups():
    """Test that search_nodes auto-includes shared groups."""
    print('\nTest: search_nodes auto-includes shared groups')

    async def run_test():

        from utils.project_config import ProjectConfig

        # Create project config with shared groups
        mock_project_config = ProjectConfig(
            group_id='test-project',
            config_path=Path('/tmp/test.json'),
            shared_group_ids=['user-common', 'team-standards'],
            shared_entity_types=['Preference'],
        )

        # Create mock config
        mock_config = MagicMock()
        mock_config.graphiti.group_id = 'test-project'

        # Mock the global services - get_client should return the mock client directly (not awaited)
        mock_client = AsyncMock()
        mock_results = MagicMock()
        mock_results.nodes = []
        mock_client.search_.return_value = mock_results
        mock_graphiti_service = AsyncMock()
        mock_graphiti_service.get_client = AsyncMock(return_value=mock_client)

        with patch('graphiti_mcp_server.graphiti_service', mock_graphiti_service):
            with patch('graphiti_mcp_server.project_config', mock_project_config):
                # Set the global config variable in the module
                import graphiti_mcp_server

                graphiti_mcp_server.config = mock_config

                # Import and call search_nodes
                from graphiti_mcp_server import search_nodes

                result = await search_nodes(query='test query')

                # Verify search_ was called with project + shared groups
                mock_client.search_.assert_called_once()
                call_kwargs = mock_client.search_.call_args.kwargs

                effective_group_ids = call_kwargs['group_ids']
                assert 'test-project' in effective_group_ids
                assert 'user-common' in effective_group_ids
                assert 'team-standards' in effective_group_ids
                assert len(effective_group_ids) == 3

                print('  ✓ search_nodes correctly auto-includes shared groups')

    asyncio.run(run_test())


def test_search_facts_auto_includes_shared_groups():
    """Test that search_memory_facts auto-includes shared groups."""
    print('\nTest: search_memory_facts auto-includes shared groups')

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

        # Mock the global services
        mock_client = AsyncMock()
        mock_client.search.return_value = []
        mock_graphiti_service = AsyncMock()
        mock_graphiti_service.get_client = AsyncMock(return_value=mock_client)

        with patch('graphiti_mcp_server.graphiti_service', mock_graphiti_service):
            with patch('graphiti_mcp_server.project_config', mock_project_config):
                # Set the global config variable in the module
                import graphiti_mcp_server

                graphiti_mcp_server.config = mock_config

                # Import and call search_memory_facts
                from graphiti_mcp_server import search_memory_facts

                result = await search_memory_facts(query='test query')

                # Verify search was called with project + shared groups
                mock_client.search.assert_called_once()
                call_kwargs = mock_client.search.call_args.kwargs

                effective_group_ids = call_kwargs['group_ids']
                assert 'test-project' in effective_group_ids
                assert 'user-common' in effective_group_ids
                assert len(effective_group_ids) == 2

                print('  ✓ search_memory_facts correctly auto-includes shared groups')

    asyncio.run(run_test())


def test_get_episodes_auto_includes_shared_groups():
    """Test that get_episodes auto-includes shared groups."""
    print('\nTest: get_episodes auto-includes shared groups')

    async def run_test():
        from graphiti_core.nodes import EpisodicNode

        from utils.project_config import ProjectConfig

        # Create project config with shared groups
        mock_project_config = ProjectConfig(
            group_id='test-project',
            config_path=Path('/tmp/test.json'),
            shared_group_ids=['user-common', 'team-standards'],
            shared_entity_types=['Preference'],
        )

        # Create mock config
        mock_config = MagicMock()
        mock_config.graphiti.group_id = 'test-project'

        # Mock the global services
        mock_client = MagicMock()
        mock_client.driver = AsyncMock()
        EpisodicNode.get_by_group_ids = AsyncMock(return_value=[])
        mock_graphiti_service = AsyncMock()
        mock_graphiti_service.get_client = AsyncMock(return_value=mock_client)

        with patch('graphiti_mcp_server.graphiti_service', mock_graphiti_service):
            with patch('graphiti_mcp_server.project_config', mock_project_config):
                # Set the global config variable in the module
                import graphiti_mcp_server

                graphiti_mcp_server.config = mock_config

                # Import and call get_episodes
                from graphiti_mcp_server import get_episodes

                result = await get_episodes()

                # Verify get_by_group_ids was called with project + shared groups
                EpisodicNode.get_by_group_ids.assert_called_once()
                call_args = EpisodicNode.get_by_group_ids.call_args

                effective_group_ids = call_args[0][1]  # Second positional arg
                assert 'test-project' in effective_group_ids
                assert 'user-common' in effective_group_ids
                assert 'team-standards' in effective_group_ids
                assert len(effective_group_ids) == 3

                print('  ✓ get_episodes correctly auto-includes shared groups')

    asyncio.run(run_test())


def test_search_explicit_group_ids_not_modified():
    """Test that explicit group_ids are not modified."""
    print('\nTest: Explicit group_ids are not modified')

    async def run_test():
        from utils.project_config import ProjectConfig

        # Create project config with shared groups
        mock_project_config = ProjectConfig(
            group_id='test-project',
            config_path=Path('/tmp/test.json'),
            shared_group_ids=['user-common', 'team-standards'],
            shared_entity_types=['Preference'],
        )

        # Create mock config
        mock_config = MagicMock()
        mock_config.graphiti.group_id = 'test-project'

        # Mock the global services
        mock_client = AsyncMock()
        mock_results = MagicMock()
        mock_results.nodes = []
        mock_client.search_.return_value = mock_results
        mock_graphiti_service = AsyncMock()
        mock_graphiti_service.get_client = AsyncMock(return_value=mock_client)

        with patch('graphiti_mcp_server.graphiti_service', mock_graphiti_service):
            with patch('graphiti_mcp_server.project_config', mock_project_config):
                # Set the global config variable in the module
                import graphiti_mcp_server

                graphiti_mcp_server.config = mock_config

                # Import and call search_nodes with explicit group_ids
                from graphiti_mcp_server import search_nodes

                result = await search_nodes(
                    query='test query', group_ids=['explicit-group-1', 'explicit-group-2']
                )

                # Verify search_ was called ONLY with explicit groups
                mock_client.search_.assert_called_once()
                call_kwargs = mock_client.search_.call_args.kwargs

                effective_group_ids = call_kwargs['group_ids']
                assert effective_group_ids == ['explicit-group-1', 'explicit-group-2']
                assert 'user-common' not in effective_group_ids
                assert 'team-standards' not in effective_group_ids

                print('  ✓ Explicit group_ids are not modified')

    asyncio.run(run_test())


def test_search_without_shared_config():
    """Test that search works without shared config (backward compatibility)."""
    print('\nTest: Search works without shared config')

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
        mock_client = AsyncMock()
        mock_results = MagicMock()
        mock_results.nodes = []
        mock_client.search_.return_value = mock_results
        mock_graphiti_service = AsyncMock()
        mock_graphiti_service.get_client = AsyncMock(return_value=mock_client)

        with patch('graphiti_mcp_server.graphiti_service', mock_graphiti_service):
            with patch('graphiti_mcp_server.project_config', mock_project_config):
                # Set the global config variable in the module
                import graphiti_mcp_server

                graphiti_mcp_server.config = mock_config

                # Import and call search_nodes
                from graphiti_mcp_server import search_nodes

                result = await search_nodes(query='test query')

                # Verify search_ was called only with project group
                mock_client.search_.assert_called_once()
                call_kwargs = mock_client.search_.call_args.kwargs

                effective_group_ids = call_kwargs['group_ids']
                assert effective_group_ids == ['test-project']

                print('  ✓ Search works correctly without shared config')

    asyncio.run(run_test())


def test_search_deduplicates_groups():
    """Test that duplicate group_ids are removed."""
    print('\nTest: Duplicate group_ids are deduplicated')

    async def run_test():
        from utils.project_config import ProjectConfig

        # Create project config with shared group that matches project
        mock_project_config = ProjectConfig(
            group_id='test-project',
            config_path=Path('/tmp/test.json'),
            shared_group_ids=['test-project', 'user-common'],  # test-project is duplicated
            shared_entity_types=['Preference'],
        )

        # Create mock config
        mock_config = MagicMock()
        mock_config.graphiti.group_id = 'test-project'

        # Mock the global services
        mock_client = AsyncMock()
        mock_results = MagicMock()
        mock_results.nodes = []
        mock_client.search_.return_value = mock_results
        mock_graphiti_service = AsyncMock()
        mock_graphiti_service.get_client = AsyncMock(return_value=mock_client)

        with patch('graphiti_mcp_server.graphiti_service', mock_graphiti_service):
            with patch('graphiti_mcp_server.project_config', mock_project_config):
                # Set the global config variable in the module
                import graphiti_mcp_server

                graphiti_mcp_server.config = mock_config

                # Import and call search_nodes
                from graphiti_mcp_server import search_nodes

                result = await search_nodes(query='test query')

                # Verify deduplication
                mock_client.search_.assert_called_once()
                call_kwargs = mock_client.search_.call_args.kwargs

                effective_group_ids = call_kwargs['group_ids']
                # Should have test-project only once despite being in both places
                assert effective_group_ids.count('test-project') == 1
                assert 'user-common' in effective_group_ids
                assert len(effective_group_ids) == 2

                print('  ✓ Duplicate group_ids are correctly deduplicated')

    asyncio.run(run_test())


def run_all_tests():
    """Run all tests."""
    print('=' * 60)
    print('Running Smart Search Tests')
    print('=' * 60)

    tests = [
        test_search_nodes_auto_includes_shared_groups,
        test_search_facts_auto_includes_shared_groups,
        test_get_episodes_auto_includes_shared_groups,
        test_search_explicit_group_ids_not_modified,
        test_search_without_shared_config,
        test_search_deduplicates_groups,
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
