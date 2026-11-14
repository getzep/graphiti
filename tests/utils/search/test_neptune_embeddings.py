"""
Copyright 2026, Zep Software, Inc.

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

from math import isfinite
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import CommunityNode, EntityNode
from graphiti_core.search.search import node_search
from graphiti_core.search.search_config import NodeReranker, NodeSearchConfig, NodeSearchMethod
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    get_embeddings_for_communities,
    get_embeddings_for_edges,
    get_embeddings_for_nodes,
    maximal_marginal_relevance,
)
from graphiti_core.utils.datetime_utils import utc_now


@pytest.fixture(
    params=[
        (get_embeddings_for_nodes, EntityNode, 'name_embedding', 'node_uuids'),
        (get_embeddings_for_communities, CommunityNode, 'name_embedding', 'community_uuids'),
        (get_embeddings_for_edges, EntityEdge, 'fact_embedding', 'edge_uuids'),
    ],
    ids=['nodes', 'communities', 'edges'],
)
def embedding_loader(request):
    loader, model, field, parameter = request.param
    kwargs = {'name': 'candidate', 'group_id': 'test'}
    if model is EntityEdge:
        kwargs.update(source_node_uuid='source', target_node_uuid='target', fact='a fact')
    candidate = model(uuid='candidate', created_at=utc_now(), **kwargs)
    return loader, candidate, field, parameter


@pytest.mark.parametrize('provider', [GraphProvider.NEPTUNE, GraphProvider.NEO4J])
async def test_embedding_loaders_return_numeric_vectors_for_mmr(embedding_loader, provider):
    loader, candidate, field, parameter = embedding_loader
    values = ['1.0', '0.0', '-2.5e-1'] if provider == GraphProvider.NEPTUNE else [1.0, 0.0, -0.25]
    driver = MagicMock(spec=GraphDriver)
    driver.provider = provider
    driver.graph_operations_interface = None
    driver.search_interface = None
    driver.execute_query = AsyncMock(
        return_value=(
            [
                {'uuid': candidate.uuid, field: values},
                {'uuid': None, field: values},
                {'uuid': 'missing-embedding', field: None},
            ],
            None,
            None,
        )
    )

    embeddings = await loader(driver, [candidate])

    assert embeddings == {'candidate': [1.0, 0.0, -0.25]}
    assert all(isinstance(value, float) for value in embeddings['candidate'])
    assert driver.execute_query.await_args.kwargs[parameter] == ['candidate']
    uuids, scores = maximal_marginal_relevance([1.0, 0.0, 0.0], embeddings)
    assert uuids == ['candidate']
    assert len(scores) == 1
    assert isfinite(scores[0])


async def test_neptune_embedding_loaders_accept_empty_results(embedding_loader):
    loader, _, _, _ = embedding_loader
    driver = MagicMock(spec=GraphDriver)
    driver.provider = GraphProvider.NEPTUNE
    driver.graph_operations_interface = None
    driver.search_interface = None
    driver.execute_query = AsyncMock(return_value=([], None, None))

    assert await loader(driver, []) == {}


async def test_neptune_node_search_reranks_loaded_string_embeddings(monkeypatch):
    nodes = [
        EntityNode(uuid='relevant', name='relevant', group_id='test'),
        EntityNode(uuid='other', name='other', group_id='test'),
    ]
    monkeypatch.setattr(
        'graphiti_core.search.search.node_similarity_search', AsyncMock(return_value=nodes)
    )
    driver = MagicMock(spec=GraphDriver)
    driver.provider = GraphProvider.NEPTUNE
    driver.graph_operations_interface = None
    driver.execute_query = AsyncMock(
        return_value=(
            [
                {'uuid': 'other', 'name_embedding': ['0.0', '1.0']},
                {'uuid': 'relevant', 'name_embedding': ['1.0', '0.0']},
            ],
            None,
            None,
        )
    )

    results, scores = await node_search(
        driver,
        MagicMock(),
        query='relevant',
        query_vector=[1.0, 0.0],
        group_ids=['test'],
        config=NodeSearchConfig(
            search_methods=[NodeSearchMethod.cosine_similarity], reranker=NodeReranker.mmr
        ),
        search_filter=SearchFilters(),
    )

    assert [node.uuid for node in results] == ['relevant', 'other']
    assert scores == pytest.approx([0.5, 0.0])
