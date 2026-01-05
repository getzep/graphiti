"""Test stdio mode project isolation with .graphiti.json config files.

This test verifies that when the MCP server starts with GRAPHITI_PROJECT_DIR
set, it correctly detects and uses the group_id from the project's .graphiti.json.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Set environment variables before importing the server module
os.environ['GRAPHITI_PROJECT_DIR'] = '/tmp/test_project'


@pytest.mark.asyncio
async def test_stdio_mode_uses_project_group_id():
    """Test that stdio mode correctly uses group_id from .graphiti.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        config_file = project_dir / '.graphiti.json'

        # Create a .graphiti.json config file
        config_file.write_text('{"group_id": "test-project-123"}')

        # Set environment variable to point to this project
        os.environ['GRAPHITI_PROJECT_DIR'] = str(project_dir)

        # Import and use the project config detection
        from src.utils.project_config import find_project_config

        # This simulates what initialize_server() does
        config = find_project_config(project_dir)

        assert config is not None, "Project config should be found"
        assert config.group_id == 'test-project-123', \
            f"Expected group_id 'test-project-123', got '{config.group_id}'"


@pytest.mark.asyncio
async def test_stdio_mode_without_config_file():
    """Test that stdio mode works without .graphiti.json (uses default)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)

        # No .graphiti.json file created
        from src.utils.project_config import find_project_config

        config = find_project_config(project_dir)

        # When no config file exists, should return None
        # The server will then use the default group_id from its config
        assert config is None, "No config should be found when .graphiti.json doesn't exist"


@pytest.mark.asyncio
async def test_multiple_projects_isolated():
    """Test that different projects get different group_ids."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two projects with different configs
        project_a = Path(tmpdir) / 'project_a'
        project_b = Path(tmpdir) / 'project_b'
        project_a.mkdir()
        project_b.mkdir()

        (project_a / '.graphiti.json').write_text('{"group_id": "project-a"}')
        (project_b / '.graphiti.json').write_text('{"group_id": "project-b"}')

        # Simulate starting server for project A
        os.environ['GRAPHITI_PROJECT_DIR'] = str(project_a)
        from src.utils.project_config import find_project_config
        config_a = find_project_config(project_a)
        assert config_a.group_id == 'project-a'

        # Simulate starting server for project B
        os.environ['GRAPHITI_PROJECT_DIR'] = str(project_b)
        config_b = find_project_config(project_b)
        assert config_b.group_id == 'project-b'

        # Verify they are different
        assert config_a.group_id != config_b.group_id


@pytest.mark.asyncio
async def test_subdirectory_finds_project_config():
    """Test that working in a subdirectory still finds the project config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        config_file = project_dir / '.graphiti.json'
        config_file.write_text('{"group_id": "my-app"}')

        # Create a subdirectory (simulating working in src/ or tests/)
        subdir = project_dir / 'src' / 'components'
        subdir.mkdir(parents=True)

        # Even when working in subdir, should find the project root config
        os.environ['GRAPHITI_PROJECT_DIR'] = str(subdir)

        from src.utils.project_config import find_project_config
        config = find_project_config(subdir)
        assert config is not None
        assert config.group_id == 'my-app'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
