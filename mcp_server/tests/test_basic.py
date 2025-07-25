"""Basic tests for the mcp-server project."""

import asyncio

import pytest


def test_basic_functionality():
    """Basic test to ensure pytest is working."""
    assert True


@pytest.mark.asyncio
async def test_async_functionality():
    """Test that async tests work."""
    await asyncio.sleep(0.001)  # Minimal async operation
    assert True


def test_import_available():
    """Test that the main module file exists and can be found."""
    import os

    assert os.path.exists("src/graphiti_mcp_server.py")
