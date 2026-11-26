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

from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_config import SearchResults


def test_merge_sorts_by_score_descending():
    n1 = EntityNode(name='Entity Low Relevance', summary='Group 1', group_id='g1')
    n2 = EntityNode(name='Entity High Relevance', summary='Group 2', group_id='g2')
    res1 = SearchResults(nodes=[n1], node_reranker_scores=[0.40])
    res2 = SearchResults(nodes=[n2], node_reranker_scores=[0.98])

    merged = SearchResults.merge([res1, res2])

    assert merged.nodes[0].name == 'Entity High Relevance'
    assert merged.node_reranker_scores == [0.98, 0.40]


def test_merge_applies_limit():
    nodes = [EntityNode(name=f'n{i}', summary='', group_id='g') for i in range(5)]
    scores = [0.1, 0.9, 0.5, 0.8, 0.2]
    res = SearchResults(nodes=nodes, node_reranker_scores=scores)
    merged = SearchResults.merge([res], limit=2)
    assert len(merged.nodes) == 2
    assert merged.node_reranker_scores == [0.9, 0.8]
    assert [n.name for n in merged.nodes] == ['n1', 'n3']


def test_merge_dedupes_uuid_keeping_highest_score():
    shared = EntityNode(name='Shared', summary='a', group_id='g1')
    other = EntityNode(name='Other', summary='b', group_id='g2')
    res1 = SearchResults(nodes=[shared], node_reranker_scores=[0.40])
    shared_again = shared.model_copy()
    res2 = SearchResults(nodes=[shared_again, other], node_reranker_scores=[0.95, 0.50])

    merged = SearchResults.merge([res1, res2])
    assert len(merged.nodes) == 2
    assert merged.nodes[0].uuid == shared.uuid
    assert merged.node_reranker_scores[0] == 0.95


def test_merge_without_scores_preserves_order_and_empty_scores():
    n1 = EntityNode(name='a', summary='', group_id='g1')
    n2 = EntityNode(name='b', summary='', group_id='g2')
    res1 = SearchResults(nodes=[n1])
    res2 = SearchResults(nodes=[n2])
    merged = SearchResults.merge([res1, res2])
    assert [n.name for n in merged.nodes] == ['a', 'b']
    assert merged.node_reranker_scores == []


def test_merge_respects_limit_across_multiple_groups():
    """
    When searching across multiple groups with a limit, the decorator merges
    per-group results and applies the limit globally, not per-group (#2014).
    """
    # Simulate per-group results: group A returns 2, group B returns 2
    nodes_a = [EntityNode(name=f'a{i}', summary='Group A', group_id='gA') for i in range(2)]
    nodes_b = [EntityNode(name=f'b{i}', summary='Group B', group_id='gB') for i in range(2)]

    # Both groups have scores
    res_a = SearchResults(nodes=nodes_a, node_reranker_scores=[0.9, 0.7])
    res_b = SearchResults(nodes=nodes_b, node_reranker_scores=[0.8, 0.6])

    # Merge with limit=2 should return 2 total, not 2 per group
    merged = SearchResults.merge([res_a, res_b], limit=2)

    assert len(merged.nodes) == 2
    assert len(merged.node_reranker_scores) == 2
    # Top 2 scores are 0.9 and 0.8
    assert merged.node_reranker_scores == [0.9, 0.8]
    assert merged.nodes[0].name == 'a0'  # score 0.9
    assert merged.nodes[1].name == 'b0'  # score 0.8
