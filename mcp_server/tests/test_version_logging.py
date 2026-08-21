import importlib
import sys
from pathlib import Path

import graphiti_core

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

_get_graphiti_core_version = importlib.import_module(
    'graphiti_mcp_server'
)._get_graphiti_core_version


def test_get_graphiti_core_version_prefers_package_attribute(monkeypatch, tmp_path):
    version_file = tmp_path / '.graphiti-core-version'
    version_file.write_text('fallback-version\n')
    monkeypatch.setattr(graphiti_core, '__version__', '1.2.3', raising=False)

    assert _get_graphiti_core_version(version_file) == '1.2.3'


def test_get_graphiti_core_version_falls_back_to_version_file(monkeypatch, tmp_path):
    version_file = tmp_path / '.graphiti-core-version'
    version_file.write_text('0.29.1\n')
    monkeypatch.setattr(graphiti_core, '__version__', 'unknown', raising=False)

    assert _get_graphiti_core_version(version_file) == '0.29.1'


def test_get_graphiti_core_version_returns_none_when_unavailable(monkeypatch, tmp_path):
    version_file = tmp_path / 'missing-version-file'
    monkeypatch.setattr(graphiti_core, '__version__', 'unknown', raising=False)

    assert _get_graphiti_core_version(version_file) is None
