import os
import sys

import pytest

# This code adds the project root directory to the Python path, allowing imports to work correctly when running tests.
# Without this file, you might encounter ModuleNotFoundError when trying to import modules from your project, especially when running tests.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from tests.helpers_test import graph_driver, mock_embedder

__all__ = ['graph_driver', 'mock_embedder']


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    integration_marker = pytest.mark.integration

    for item in items:
        if 'graph_driver' not in item.fixturenames:
            continue
        if item.get_closest_marker('integration') is not None:
            continue
        item.add_marker(integration_marker)
