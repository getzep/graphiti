from types import SimpleNamespace

import pytest

import graphiti_core.graphiti as graphiti_module
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.graphiti import Graphiti
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_NODE_DISTANCE,
    EDGE_HYBRID_SEARCH_RRF,
)


@pytest.mark.parametrize(
    ('center_node_uuid', 'recipe', 'num_results'),
    [
        (None, EDGE_HYBRID_SEARCH_RRF, 3),
        ('center-node-uuid', EDGE_HYBRID_SEARCH_NODE_DISTANCE, 7),
    ],
)
async def test_search_does_not_mutate_shared_recipe(
    monkeypatch, center_node_uuid, recipe, num_results
):
    graphiti = object.__new__(Graphiti)
    graphiti.clients = SimpleNamespace(
        driver=SimpleNamespace(provider=GraphProvider.NEO4J),
    )
    original_limit = recipe.limit
    received_configs = []

    async def fake_search(_clients, _query, _group_ids, config, _search_filter, **_kwargs):
        received_configs.append(config)
        return SearchResults()

    monkeypatch.setattr(graphiti_module, 'search', fake_search)

    try:
        await graphiti.search(
            'query',
            center_node_uuid=center_node_uuid,
            num_results=num_results,
        )

        assert received_configs[0].limit == num_results
        assert recipe.limit == original_limit
    finally:
        recipe.limit = original_limit
