#!/usr/bin/env python3
"""Unit tests for MCP search recipe selection."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# Add the src directory to the path (mirrors the other MCP unit tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graphiti_core.search.search_config import EdgeReranker, NodeReranker, SearchResults
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    NODE_HYBRID_SEARCH_CROSS_ENCODER,
)

import graphiti_mcp_server
from config.schema import GraphitiConfig


def install_fake_service(monkeypatch):
    client = AsyncMock()
    client.search_.return_value = SearchResults()
    service = AsyncMock()
    service.get_client.return_value = client
    monkeypatch.setattr(graphiti_mcp_server, 'graphiti_service', service)
    return client


@pytest.mark.asyncio
async def test_search_nodes_uses_configured_cross_encoder_recipe(monkeypatch):
    client = install_fake_service(monkeypatch)
    monkeypatch.setattr(
        graphiti_mcp_server,
        'config',
        GraphitiConfig(graphiti={'node_reranker': 'cross_encoder'}),
        raising=False,
    )

    await graphiti_mcp_server.search_nodes('query', max_nodes=7)

    search_config = client.search_.await_args.kwargs['config']
    assert search_config.node_config.reranker is NodeReranker.cross_encoder
    assert search_config.limit == 7
    assert NODE_HYBRID_SEARCH_CROSS_ENCODER.limit == 10


@pytest.mark.asyncio
async def test_search_nodes_center_node_keeps_distance_recipe(monkeypatch):
    client = install_fake_service(monkeypatch)
    monkeypatch.setattr(
        graphiti_mcp_server,
        'config',
        GraphitiConfig(graphiti={'node_reranker': 'cross_encoder'}),
        raising=False,
    )

    await graphiti_mcp_server.search_nodes('query', center_node_uuid='center')

    search_config = client.search_.await_args.kwargs['config']
    assert search_config.node_config.reranker is NodeReranker.node_distance


@pytest.mark.asyncio
async def test_search_memory_facts_uses_configured_cross_encoder_recipe(monkeypatch):
    client = install_fake_service(monkeypatch)
    monkeypatch.setattr(
        graphiti_mcp_server,
        'config',
        GraphitiConfig(graphiti={'fact_reranker': 'cross_encoder'}),
        raising=False,
    )

    await graphiti_mcp_server.search_memory_facts('query', max_facts=8)

    search_config = client.search_.await_args.kwargs['config']
    assert search_config.edge_config.reranker is EdgeReranker.cross_encoder
    assert search_config.limit == 8
    assert EDGE_HYBRID_SEARCH_CROSS_ENCODER.limit == 10


@pytest.mark.asyncio
async def test_search_memory_facts_center_node_keeps_distance_recipe(monkeypatch):
    client = install_fake_service(monkeypatch)
    monkeypatch.setattr(
        graphiti_mcp_server,
        'config',
        GraphitiConfig(graphiti={'fact_reranker': 'cross_encoder'}),
        raising=False,
    )

    await graphiti_mcp_server.search_memory_facts('query', center_node_uuid='center')

    search_config = client.search_.await_args.kwargs['config']
    assert search_config.edge_config.reranker is EdgeReranker.node_distance
