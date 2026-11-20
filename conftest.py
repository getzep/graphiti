import os
import sys

# This code adds the project root directory to the Python path, allowing imports to work correctly when running tests.
# Without this file, you might encounter ModuleNotFoundError when trying to import modules from your project, especially when running tests.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

import pytest  # noqa: E402

from graphiti_core.prompts.embedding_guard import PromptScan, scan_message_prompts  # noqa: E402
from tests.helpers_test import graph_driver, mock_embedder  # noqa: E402

# Exclude mcp_server from test collection - it has its own test suite and conftest
collect_ignore_glob = ['mcp_server/*']

__all__ = ['graph_driver', 'mock_embedder']


def _check_embedding_scan(scan: PromptScan, marked: bool) -> None:
    if scan.records and not marked:
        pytest.fail(scan.format_failure())


@pytest.fixture(autouse=True)
def no_embedding_in_prompts(request):
    with scan_message_prompts() as scan:
        yield
    _check_embedding_scan(
        scan,
        request.node.get_closest_marker('allow_embedding_in_prompt') is not None,
    )
