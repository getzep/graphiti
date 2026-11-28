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

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from graphiti_core.decorators import handle_multiple_group_ids
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_config import SearchConfig, SearchResults


def _make_node(name: str, uuid: str, group_id: str = 'default') -> EntityNode:
    return EntityNode(
        name=name,
        uuid=uuid,
        summary=f'{name} summary',
        group_id=group_id,
        created_at=datetime.now(timezone.utc),
    )


def _make_edge(name: str, uuid: str, group_id: str = 'default') -> EntityEdge:
    return EntityEdge(
        name=name,
        uuid=uuid,
        group_id=group_id,
        source_node_uuid='src',
        target_node_uuid='tgt',
        created_at=datetime.now(timezone.utc),
        fact=f'{name} fact',
    )


def test_merge_empty():
    merged = SearchResults.merge([])
    assert isinstance(merged, SearchResults)
    assert len(merged.nodes) == 0
    assert len(merged.edges) == 0


def test_merge_single_result():
    node = _make_node('Node1', 'n-1')
    res = SearchResults(nodes=[node], node_reranker_scores=[0.85])
    merged = SearchResults.merge([res])
    assert len(merged.nodes) == 1
    assert merged.nodes[0].uuid == 'n-1'
    assert merged.node_reranker_scores == [0.85]


def test_merge_sorts_descending_by_score():
    n1 = _make_node('Node Low', 'n-1', group_id='g1')
    n2 = _make_node('Node High', 'n-2', group_id='g2')
    n3 = _make_node('Node Mid', 'n-3', group_id='g1')

    res1 = SearchResults(nodes=[n1, n3], node_reranker_scores=[0.40, 0.60])
    res2 = SearchResults(nodes=[n2], node_reranker_scores=[0.95])

    merged = SearchResults.merge([res1, res2])

    assert [n.uuid for n in merged.nodes] == ['n-2', 'n-3', 'n-1']
    assert merged.node_reranker_scores == [0.95, 0.60, 0.40]


def test_merge_deduplicates_by_uuid_keeping_highest_score():
    n_g1 = _make_node('Shared Node', 'shared-uuid', group_id='g1')
    n_g2 = _make_node('Shared Node', 'shared-uuid', group_id='g2')
    other = _make_node('Other Node', 'other-uuid', group_id='g1')

    res1 = SearchResults(nodes=[n_g1, other], node_reranker_scores=[0.30, 0.50])
    res2 = SearchResults(nodes=[n_g2], node_reranker_scores=[0.90])

    merged = SearchResults.merge([res1, res2])

    assert len(merged.nodes) == 2
    assert [n.uuid for n in merged.nodes] == ['shared-uuid', 'other-uuid']
    assert merged.node_reranker_scores == [0.90, 0.50]


def test_merge_respects_limit():
    nodes_g1 = [_make_node(f'N1_{i}', f'n1-{i}') for i in range(5)]
    scores_g1 = [0.5, 0.4, 0.3, 0.2, 0.1]
    nodes_g2 = [_make_node(f'N2_{i}', f'n2-{i}') for i in range(5)]
    scores_g2 = [0.95, 0.85, 0.75, 0.65, 0.55]

    res1 = SearchResults(nodes=nodes_g1, node_reranker_scores=scores_g1)
    res2 = SearchResults(nodes=nodes_g2, node_reranker_scores=scores_g2)

    merged = SearchResults.merge([res1, res2], limit=3)

    assert len(merged.nodes) == 3
    assert len(merged.node_reranker_scores) == 3
    assert [n.uuid for n in merged.nodes] == ['n2-0', 'n2-1', 'n2-2']
    assert merged.node_reranker_scores == [0.95, 0.85, 0.75]


def test_merge_without_scores_preserves_order_and_deduplicates():
    n1 = _make_node('Node 1', 'n-1')
    n2 = _make_node('Node 2', 'n-2')
    n3 = _make_node('Node 1 duplicate', 'n-1')

    res1 = SearchResults(nodes=[n1, n2], node_reranker_scores=[])
    res2 = SearchResults(nodes=[n3], node_reranker_scores=[])

    merged = SearchResults.merge([res1, res2])

    assert len(merged.nodes) == 2
    assert [n.uuid for n in merged.nodes] == ['n-1', 'n-2']
    assert merged.node_reranker_scores == []


def test_merge_edges_and_scores():
    e1 = _make_edge('Edge Low', 'e-1')
    e2 = _make_edge('Edge High', 'e-2')

    res1 = SearchResults(edges=[e1], edge_reranker_scores=[0.2])
    res2 = SearchResults(edges=[e2], edge_reranker_scores=[0.99])

    merged = SearchResults.merge([res1, res2])

    assert [e.uuid for e in merged.edges] == ['e-2', 'e-1']
    assert merged.edge_reranker_scores == [0.99, 0.2]


class _FakeDriver:
    def __init__(self, database: str, provider: GraphProvider = GraphProvider.FALKORDB):
        self._database = database
        self.provider = provider

    def clone(self, database: str) -> '_FakeDriver':
        return _FakeDriver(database, self.provider)


class _HostWithSearchResults:
    def __init__(self, driver: _FakeDriver):
        self.clients = SimpleNamespace(driver=driver)
        self.max_coroutines = None

    @handle_multiple_group_ids
    async def search_(
        self,
        query: str,
        config: SearchConfig | None = None,
        group_ids: list[str] | None = None,
        driver=None,
    ) -> SearchResults:
        db = getattr(driver, '_database', 'default')
        if db == 'group_a':
            node = _make_node('Node A', 'uuid-a', group_id='group_a')
            return SearchResults(nodes=[node], node_reranker_scores=[0.4])
        else:
            node = _make_node('Node B', 'uuid-b', group_id='group_b')
            return SearchResults(nodes=[node], node_reranker_scores=[0.95])


@pytest.mark.asyncio
async def test_decorator_merges_search_results_with_config_limit():
    driver = _FakeDriver('reggraph')
    host = _HostWithSearchResults(driver)
    config = SearchConfig(limit=1)

    result = await host.search_('test query', config=config, group_ids=['group_a', 'group_b'])

    assert isinstance(result, SearchResults)
    assert len(result.nodes) == 1
    assert result.nodes[0].uuid == 'uuid-b'
    assert result.node_reranker_scores == [0.95]
