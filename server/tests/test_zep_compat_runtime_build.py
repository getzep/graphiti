"""Guards on how `build_graphiti` wires graphiti_core.

The zep_compat layer is written against the graphiti-core fork that lives one
directory up, not against the registry release. When the shim's dependency
resolved to PyPI graphiti-core instead, nothing failed at import or at startup:
`build_graphiti` is called lazily, the first time a graph is opened, so the
mismatch only surfaced as every episode in an ingest batch dying with

    OpenAIGenericClient.__init__() got an unexpected keyword argument
    'structured_output_mode'

These tests construct the real clients (no network: the OpenAI client and the
FalkorDB redis client both connect lazily) so that a wrong graphiti-core is
caught by `make test`, before a build ever runs.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import graphiti_core
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from graph_service.zep_compat.runtime import build_graphiti

FORK_ROOT = Path(__file__).resolve().parents[2]


def test_graphiti_core_is_the_sibling_fork():
    resolved = Path(graphiti_core.__file__).resolve()
    assert resolved.is_relative_to(FORK_ROOT), (
        f'graphiti_core resolved to {resolved}, not the fork at {FORK_ROOT}. '
        'Check [tool.uv.sources] in server/pyproject.toml and re-run `uv sync`.'
    )


def test_openai_generic_client_accepts_structured_output_mode():
    assert 'structured_output_mode' in inspect.signature(OpenAIGenericClient.__init__).parameters


def test_build_graphiti_constructs_with_structured_output(monkeypatch):
    monkeypatch.setenv('GRAPHITI_DB_BACKEND', 'falkordb')
    monkeypatch.setenv('GRAPHITI_RERANKER', 'none')
    monkeypatch.delenv('GRAPHITI_STRUCTURED_OUTPUT_MODE', raising=False)

    client = build_graphiti('build_graphiti_regression').llm_client

    assert isinstance(client, OpenAIGenericClient)
    assert client.structured_output_mode == 'json_schema'
