#!/usr/bin/env python3
"""Unit tests for MCP search recipe selection."""

import importlib
import sys
from datetime import datetime, timezone
from inspect import unwrap
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

# Add the src directory to the path (mirrors the other MCP unit tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti import Graphiti
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_config import EdgeReranker, NodeReranker, SearchResults
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    NODE_HYBRID_SEARCH_CROSS_ENCODER,
)

import graphiti_mcp_server
from config.schema import GraphitiConfig
from services.factories import CrossEncoderFactory


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
        GraphitiConfig.model_validate({'graphiti': {'node_reranker': 'cross_encoder'}}),
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
        GraphitiConfig.model_validate({'graphiti': {'node_reranker': 'cross_encoder'}}),
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
        GraphitiConfig.model_validate({'graphiti': {'fact_reranker': 'cross_encoder'}}),
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
        GraphitiConfig.model_validate({'graphiti': {'fact_reranker': 'cross_encoder'}}),
        raising=False,
    )

    await graphiti_mcp_server.search_memory_facts('query', center_node_uuid='center')

    search_config = client.search_.await_args.kwargs['config']
    assert search_config.edge_config.reranker is EdgeReranker.node_distance


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', ['openai', 'gemini'])
@pytest.mark.parametrize('tool', ['search_nodes', 'search_memory_facts'])
@pytest.mark.parametrize('recipe', ['rrf', 'cross_encoder'])
async def test_mcp_search_calls_selected_reranker_only_when_enabled(
    monkeypatch, provider, tool, recipe
):
    """Exercise real core ranking, replacing only retrieval and external API calls."""
    setting = 'node_reranker' if tool == 'search_nodes' else 'fact_reranker'
    config = GraphitiConfig.model_validate(
        {
            'llm': {
                'providers': {'openai': {'api_key': 'test-key'}, 'gemini': {'api_key': 'test-key'}}
            },
            'reranker': {'provider': provider},
            'graphiti': {setting: recipe},
        }
    )
    reranker = CrossEncoderFactory.create(config.llm, config.embedder, config.reranker)
    rank = AsyncMock(side_effect=lambda query, passages: [(p, 1.0) for p in reversed(passages)])
    monkeypatch.setattr(reranker, 'rank', rank)
    core_search = importlib.import_module('graphiti_core.search.search')
    nodes = [EntityNode(name=name, group_id='test') for name in ['first', 'second']]
    edges = [
        EntityEdge(
            name='test',
            fact=fact,
            group_id='test',
            source_node_uuid=nodes[0].uuid,
            target_node_uuid=nodes[1].uuid,
            created_at=datetime.now(timezone.utc),
        )
        for fact in ['first fact', 'second fact']
    ]
    for scope, candidates in [('node', nodes), ('edge', edges)]:
        for method in ['fulltext', 'similarity', 'bfs']:
            monkeypatch.setattr(
                core_search,
                f'{scope}_{method}_search',
                AsyncMock(return_value=[] if method == 'bfs' else candidates),
            )
    clients = SimpleNamespace(
        driver=Mock(),
        embedder=SimpleNamespace(create=AsyncMock(return_value=[0.0] * 768)),
        cross_encoder=reranker,
    )

    async def search_with_core(**kwargs):
        return await core_search.search(clients=clients, **kwargs)

    client = SimpleNamespace(search_=search_with_core)
    client.search = MethodType(unwrap(Graphiti.search), SimpleNamespace(clients=clients))
    service = SimpleNamespace(get_client=AsyncMock(return_value=client))
    monkeypatch.setattr(graphiti_mcp_server, 'graphiti_service', service)
    monkeypatch.setattr(graphiti_mcp_server, 'config', config, raising=False)

    response = await getattr(graphiti_mcp_server, tool)('query', group_ids=['test'])

    if tool == 'search_nodes':
        result_ids = [node['uuid'] for node in response['nodes']]
        expected_ids = [node.uuid for node in nodes]
    else:
        result_ids = [edge['uuid'] for edge in response['facts']]
        expected_ids = [edge.uuid for edge in edges]
    if recipe == 'cross_encoder':
        rank.assert_awaited_once()
        assert result_ids == list(reversed(expected_ids))
    else:
        rank.assert_not_awaited()
        assert result_ids == expected_ids
