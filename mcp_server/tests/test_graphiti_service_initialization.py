#!/usr/bin/env python3
"""Unit tests for GraphitiService.initialize() client-construction error handling.

LLM, embedder, and cross-encoder construction failures must all be fatal:
Graphiti.__init__() silently substitutes an unconfigured default client for
any of these left as None, which would bypass the server's configured
provider entirely rather than fail loudly.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

# Add the src directory to the path (mirrors the other factory tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import graphiti_mcp_server
from config.schema import DatabaseConfig, GraphitiConfig
from services.factories import CrossEncoderFactory, EmbedderFactory, LLMClientFactory


def _service() -> graphiti_mcp_server.GraphitiService:
    return graphiti_mcp_server.GraphitiService(
        GraphitiConfig(database=DatabaseConfig(provider='neo4j'))
    )


@pytest.mark.asyncio
async def test_llm_client_configuration_error_is_not_swallowed(monkeypatch):
    def fail_llm_setup(*_args):
        raise ValueError('llm setup failed')

    monkeypatch.setattr(LLMClientFactory, 'create', fail_llm_setup)

    with pytest.raises(ValueError, match='llm setup failed'):
        await _service().initialize()


@pytest.mark.asyncio
async def test_embedder_configuration_error_is_not_swallowed(monkeypatch):
    def fail_embedder_setup(*_args):
        raise ValueError('embedder setup failed')

    monkeypatch.setattr(LLMClientFactory, 'create', lambda *_args: Mock())
    monkeypatch.setattr(EmbedderFactory, 'create', fail_embedder_setup)

    with pytest.raises(ValueError, match='embedder setup failed'):
        await _service().initialize()


@pytest.mark.asyncio
async def test_reranker_configuration_error_is_not_swallowed(monkeypatch):
    def fail_reranker_setup(*_args):
        raise ValueError('reranker setup failed')

    monkeypatch.setattr(LLMClientFactory, 'create', lambda *_args: Mock())
    monkeypatch.setattr(EmbedderFactory, 'create', lambda *_args: Mock())
    monkeypatch.setattr(CrossEncoderFactory, 'create', fail_reranker_setup)

    with pytest.raises(ValueError, match='reranker setup failed'):
        await _service().initialize()
