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

from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from graphiti_core.nodes import EntityNode
from graphiti_core.search.search import search
from graphiti_core.search.search_config import (
    NodeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    SearchConfig,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import maximal_marginal_relevance


@pytest.mark.parametrize('scale', [0.5, 1.0, 10.0])
@pytest.mark.parametrize(
    ('min_score', 'expected_uuids', 'expected_scores'),
    [(-2.0, ['c', 'a', 'b'], [0.3, 0.0, -0.08]), (0.0, ['c', 'a'], [0.3, 0.0])],
)
def test_mmr_query_scale_preserves_ranking_scores_and_filtering(
    scale, min_score, expected_uuids, expected_scores
):
    candidates = {'a': [1.0, 0.0, 0.0], 'b': [0.8, 0.6, 0.0], 'c': [0.0, 0.0, 1.0]}

    uuids, scores = maximal_marginal_relevance(
        [0.8 * scale, 0.0, 0.6 * scale], candidates, min_score=min_score
    )

    assert uuids == expected_uuids
    assert scores == pytest.approx(expected_scores)


def test_mmr_zero_query_remains_finite_without_dividing_by_zero():
    with np.errstate(divide='raise', invalid='raise'):
        uuids, scores = maximal_marginal_relevance([0.0, 0.0], {'a': [1.0, 0.0], 'b': [0.0, 1.0]})

    assert uuids == ['a', 'b']
    assert scores == [0.0, 0.0]


@pytest.mark.asyncio
async def test_search_mmr_keeps_matches_with_non_unit_query_embedding(monkeypatch):
    nodes = [EntityNode(uuid=uuid, name=uuid, group_id='group') for uuid in ['a', 'b']]
    monkeypatch.setattr(
        'graphiti_core.search.search.node_fulltext_search', AsyncMock(return_value=nodes)
    )
    monkeypatch.setattr(
        'graphiti_core.search.search.get_embeddings_for_nodes',
        AsyncMock(return_value={'a': [0.7, 0.0], 'b': [0.56, 0.42]}),
    )
    clients = SimpleNamespace(
        driver=SimpleNamespace(),
        embedder=SimpleNamespace(create=AsyncMock(return_value=[0.7, 0.0])),
        cross_encoder=SimpleNamespace(),
    )

    results = await search(
        clients,
        query='matching fact',
        group_ids=['group'],
        config=SearchConfig(
            node_config=NodeSearchConfig(
                search_methods=[NodeSearchMethod.bm25], reranker=NodeReranker.mmr
            ),
            limit=2,
        ),
        search_filter=SearchFilters(),
    )

    assert [node.uuid for node in results.nodes] == ['a', 'b']
    assert results.node_reranker_scores == pytest.approx([0.1, 0.0])
