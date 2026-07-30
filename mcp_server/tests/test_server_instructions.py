"""Tests for configurable MCP server instructions."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv('GRAPHITI_MCP_INSTRUCTIONS', raising=False)
    monkeypatch.delenv('GRAPHITI_MCP_INSTRUCTIONS_FILE', raising=False)


def _resolve():
    from graphiti_mcp_server import resolve_server_instructions

    return resolve_server_instructions()


def test_defaults_to_builtin_instructions():
    from graphiti_mcp_server import GRAPHITI_MCP_INSTRUCTIONS

    assert _resolve() == GRAPHITI_MCP_INSTRUCTIONS


def test_inline_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv('GRAPHITI_MCP_INSTRUCTIONS', 'route facts to their own group')
    assert _resolve() == 'route facts to their own group'


def test_file_env_var_is_read(monkeypatch, tmp_path):
    f = tmp_path / 'instructions.md'
    f.write_text('line one\nline two\n', encoding='utf-8')
    monkeypatch.setenv('GRAPHITI_MCP_INSTRUCTIONS_FILE', str(f))
    assert _resolve() == 'line one\nline two\n'


def test_inline_takes_precedence_over_file(monkeypatch, tmp_path):
    f = tmp_path / 'instructions.md'
    f.write_text('from file', encoding='utf-8')
    monkeypatch.setenv('GRAPHITI_MCP_INSTRUCTIONS_FILE', str(f))
    monkeypatch.setenv('GRAPHITI_MCP_INSTRUCTIONS', 'from env')
    assert _resolve() == 'from env'


def test_unreadable_file_falls_back_instead_of_raising(monkeypatch, tmp_path):
    """A bad path must not stop the server from starting."""
    from graphiti_mcp_server import GRAPHITI_MCP_INSTRUCTIONS

    monkeypatch.setenv('GRAPHITI_MCP_INSTRUCTIONS_FILE', str(tmp_path / 'does-not-exist.md'))
    assert _resolve() == GRAPHITI_MCP_INSTRUCTIONS
