"""SearchResults.merge ranking, limit, and UUID dedupe (#2014)."""

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
