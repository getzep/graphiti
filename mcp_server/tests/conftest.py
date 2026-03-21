"""
Pytest configuration for MCP server tests.
This file prevents pytest from loading the parent project's conftest.py
"""

import sys
from pathlib import Path

import pytest

# Add src directory and tests directory to Python path for imports
src_path = Path(__file__).parent.parent / 'src'
tests_path = Path(__file__).parent
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(tests_path))
sys.path.insert(0, str(repo_root))

from test_fixtures import performance_benchmark  # noqa: E402,F401

from config.schema import GraphitiConfig  # noqa: E402


@pytest.fixture
def config():
    """Provide a default GraphitiConfig for tests."""
    return GraphitiConfig()
