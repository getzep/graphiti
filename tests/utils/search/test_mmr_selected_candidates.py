import tracemalloc
from datetime import datetime, timezone
from math import sqrt
from types import SimpleNamespace

import numpy as np
import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.search.search import edge_search
from graphiti_core.search.search_config import EdgeReranker, EdgeSearchConfig, EdgeSearchMethod
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import maximal_marginal_relevance

QUERY = [1.0, 0.0, 0.0]
CANDIDATES = {
    'best': [0.8, 0.6, 0.0],
    'duplicate': [0.8, 0.6, 0.0],
    'diverse': [0.6, 0.0, 0.8],
}


def test_unselected_duplicate_does_not_penalize_first_result():
    without_duplicate = {key: CANDIDATES[key] for key in ('best', 'diverse')}
    before, _ = maximal_marginal_relevance(QUERY, without_duplicate)
    after, scores = maximal_marginal_relevance(QUERY, CANDIDATES)

    assert before[0] == after[0] == 'best'
    assert scores[0] == pytest.approx(0.4)


@pytest.mark.parametrize(
    ('min_score', 'expected_uuids', 'expected_scores'),
    [
        (-2.0, ['best', 'diverse', 'duplicate'], [0.4, 0.06, -0.1]),
        (0.0, ['best', 'diverse'], [0.4, 0.06]),
        (0.4, ['best'], [0.4]),
        (0.41, [], []),
    ],
)
def test_mmr_scores_redundancy_against_selected_results(min_score, expected_uuids, expected_scores):
    uuids, scores = maximal_marginal_relevance(QUERY, CANDIDATES, min_score=min_score)

    assert uuids == expected_uuids
    assert scores == pytest.approx(expected_scores)


def test_lambda_one_preserves_relevance_ranking_and_scores():
    uuids, scores = maximal_marginal_relevance(QUERY, CANDIDATES, mmr_lambda=1.0)

    assert uuids == ['best', 'duplicate', 'diverse']
    assert scores == pytest.approx([0.8, 0.8, 0.6])


def test_lambda_zero_uses_input_order_for_initial_tie_then_diversity():
    uuids, scores = maximal_marginal_relevance(QUERY, CANDIDATES, mmr_lambda=0.0)

    assert uuids == ['best', 'diverse', 'duplicate']
    assert scores == pytest.approx([0.0, -0.48, -1.0])


def test_negative_similarity_preserves_selection_order_not_score_order():
    uuids, scores = maximal_marginal_relevance(
        [1.0, 0.0],
        {'same': [1.0, 0.0], 'opposite': [-1.0, 0.0]},
        mmr_lambda=0.25,
    )

    assert uuids == ['same', 'opposite']
    # Opposite gains a diversity bonus after same is selected.
    assert scores == pytest.approx([0.25, 0.5])


def test_empty_candidates():
    assert maximal_marginal_relevance(QUERY, {}) == ([], [])


def test_nan_scores_do_not_pass_threshold():
    assert maximal_marginal_relevance([float('nan'), 0.0, 0.0], CANDIDATES) == ([], [])


def test_single_candidate_has_no_redundancy_penalty():
    uuids, scores = maximal_marginal_relevance(QUERY, {'best': CANDIDATES['best']})

    assert uuids == ['best']
    assert scores == pytest.approx([0.4])


def test_candidate_magnitudes_do_not_change_ranking():
    scaled = {key: [10 * value for value in vector] for key, vector in CANDIDATES.items()}
    uuids, scores = maximal_marginal_relevance(QUERY, scaled)

    assert uuids == ['best', 'diverse', 'duplicate']
    assert scores == pytest.approx([0.4, 0.06, -0.1])


@pytest.mark.parametrize('min_score', [-2.0, 0.51])
def test_mmr_workspace_stays_below_pairwise_matrix_size(min_score):
    candidates = {str(index): [1.0, 0.0] for index in range(1024)}
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    try:
        uuids, _ = maximal_marginal_relevance([1.0, 0.0], candidates, min_score=min_score)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()

    assert uuids == (list(candidates) if min_score < 0.5 else [])
    # A 1024 x 1024 float64 matrix alone needs 8 MiB. Leave ample room for
    # normalized two-dimensional vectors and linear ranking workspace.
    assert peak - baseline < 4 * 1024 * 1024


def _reference_mmr(query, candidates, weight, min_score):
    """Direct greedy definition, recomputing redundancy from selected vectors."""
    vectors = {
        key: [value / sqrt(sum(x * x for x in vector)) for value in vector]
        for key, vector in candidates.items()
    }

    def dot(left, right):
        return sum(x * y for x, y in zip(left, right, strict=True))

    selected = []
    scores = []
    remaining = list(vectors)
    while remaining:

        def marginal_score(key):
            redundancy = max((dot(vectors[key], vectors[other]) for other in selected), default=0.0)
            return weight * dot(query, vectors[key]) - (1 - weight) * redundancy

        best = max(remaining, key=marginal_score)
        score = marginal_score(best)
        if score < min_score:
            break
        selected.append(best)
        scores.append(score)
        remaining.remove(best)

    return selected, scores


@pytest.mark.parametrize('weight', [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize('min_score', [-2.0, 0.0])
def test_matches_greedy_definition(weight, min_score):
    rng = np.random.default_rng(42)
    candidates = {
        str(index): vector.tolist() for index, vector in enumerate(rng.normal(size=(9, 5)))
    }
    query = rng.normal(size=5).tolist()
    expected_uuids, expected_scores = _reference_mmr(query, candidates, weight, min_score)

    uuids, scores = maximal_marginal_relevance(query, candidates, weight, min_score)

    assert uuids == expected_uuids
    assert scores == pytest.approx(expected_scores)


@pytest.mark.asyncio
@pytest.mark.parametrize('limit', [1, 2])
async def test_edge_search_keeps_relevant_memory_with_default_threshold(limit):
    edges = [
        EntityEdge(
            uuid=key,
            source_node_uuid='source',
            target_node_uuid='target',
            name='RELATES_TO',
            group_id='test-group',
            fact=key,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        for key in CANDIDATES
    ]

    class SearchBackend:
        async def edge_fulltext_search(self, driver, query, search_filter, group_ids, limit):
            return edges[:limit]

    class EmbeddingStorage:
        async def edge_load_embeddings_bulk(self, driver, requested_edges):
            return {edge.uuid: CANDIDATES[edge.uuid] for edge in requested_edges}

    driver = SimpleNamespace(
        search_interface=SearchBackend(),
        graph_operations_interface=EmbeddingStorage(),
    )
    results, scores = await edge_search(
        driver=driver,
        cross_encoder=SimpleNamespace(),
        query='relevant memory',
        query_vector=QUERY,
        group_ids=['test-group'],
        config=EdgeSearchConfig(search_methods=[EdgeSearchMethod.bm25], reranker=EdgeReranker.mmr),
        search_filter=SearchFilters(),
        limit=limit,
    )

    assert [edge.uuid for edge in results] == ['best', 'diverse'][:limit]
    assert scores == pytest.approx([0.4, 0.06][:limit])
