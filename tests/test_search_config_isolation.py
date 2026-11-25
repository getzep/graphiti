"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Regression tests for #2201: Graphiti.search() used to assign num_results on
# the shared module-level recipes EDGE_HYBRID_SEARCH_RRF /
# EDGE_HYBRID_SEARCH_NODE_DISTANCE, so concurrent searches leaked limits into
# each other and into every other importer of the recipe. These tests are
# database-free: the low-level search function is stubbed and the config it
# receives is captured.

import asyncio
from types import SimpleNamespace

import pytest

import graphiti_core.graphiti as graphiti_module
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.graphiti import Graphiti
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_NODE_DISTANCE,
    EDGE_HYBRID_SEARCH_RRF,
)

pytest_plugins = ('pytest_asyncio',)


def _make_graphiti() -> Graphiti:
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.clients = SimpleNamespace(driver=SimpleNamespace(provider=GraphProvider.NEO4J))
    return graphiti


@pytest.fixture
def captured_configs(monkeypatch):
    captured: dict[str, list] = {}

    async def fake_search(clients, query, group_ids, config, *args, **kwargs):
        captured.setdefault(query, []).append(config)
        return SimpleNamespace(edges=[])

    monkeypatch.setattr(graphiti_module, 'search', fake_search)
    return captured


@pytest.mark.asyncio
async def test_search_does_not_mutate_shared_rrf_recipe(captured_configs):
    original_limit = EDGE_HYBRID_SEARCH_RRF.limit
    graphiti = _make_graphiti()

    await graphiti.search('query', num_results=3)

    config = captured_configs['query'][0]
    assert config.limit == 3
    assert config is not EDGE_HYBRID_SEARCH_RRF
    assert EDGE_HYBRID_SEARCH_RRF.limit == original_limit


@pytest.mark.asyncio
async def test_search_does_not_mutate_shared_node_distance_recipe(captured_configs):
    original_limit = EDGE_HYBRID_SEARCH_NODE_DISTANCE.limit
    graphiti = _make_graphiti()

    await graphiti.search('query', center_node_uuid='center', num_results=7)

    config = captured_configs['query'][0]
    assert config.limit == 7
    assert config is not EDGE_HYBRID_SEARCH_NODE_DISTANCE
    assert EDGE_HYBRID_SEARCH_NODE_DISTANCE.limit == original_limit


@pytest.mark.asyncio
async def test_concurrent_searches_keep_independent_limits(captured_configs):
    rrf_limit = EDGE_HYBRID_SEARCH_RRF.limit
    graphiti = _make_graphiti()

    await asyncio.gather(
        graphiti.search('small', num_results=1),
        graphiti.search('large', num_results=100),
    )

    assert captured_configs['small'][0].limit == 1
    assert captured_configs['large'][0].limit == 100
    assert captured_configs['small'][0] is not captured_configs['large'][0]
    assert EDGE_HYBRID_SEARCH_RRF.limit == rrf_limit
