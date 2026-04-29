"""Tests for the top-level ``graphiti_core`` package exports."""

from __future__ import annotations

import re

import graphiti_core


def test_version_attribute_exposed() -> None:
    """``graphiti_core`` should expose a ``__version__`` string attribute.

    Downstream tools (notably the MCP server, see issue #1104) rely on this
    attribute to surface the installed library version at runtime.
    """
    assert hasattr(graphiti_core, '__version__'), (
        "graphiti_core.__version__ is missing; downstream loggers will "
        'report "unknown".'
    )

    version = graphiti_core.__version__
    assert isinstance(version, str)
    assert version, 'graphiti_core.__version__ should not be empty'


def test_version_attribute_included_in_all() -> None:
    assert '__version__' in graphiti_core.__all__


def test_version_is_semver_like_when_installed() -> None:
    """When the package metadata is resolvable, the version should parse.

    A source checkout that has not been installed via ``pip`` will report
    ``'unknown'``; that fallback is tolerated here so the test does not require
    an editable install.
    """
    version = graphiti_core.__version__
    if version == 'unknown':
        return
    assert re.match(r'^\d+\.\d+\.\d+', version), (
        f'Unexpected version format: {version!r}'
    )
