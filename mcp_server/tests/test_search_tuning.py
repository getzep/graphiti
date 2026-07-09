import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graphiti_core.search.search_config import EdgeReranker, NodeReranker
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters

from utils.search_tuning import (
    apply_liveness_filter,
    build_edge_search_config,
    build_node_search_config,
)


def test_edge_mmr_sets_lambda_and_reranker():
    cfg = build_edge_search_config('mmr', mmr_lambda=0.5, limit=6, min_score=0.0)
    assert cfg.edge_config.reranker == EdgeReranker.mmr
    assert cfg.edge_config.mmr_lambda == 0.5
    assert cfg.limit == 6
    # MMR must never filter: the caller's min_score (0.0 here) is overridden with
    # the no-filter sentinel, or MMR's mostly-negative scores would drop results.
    assert cfg.reranker_min_score == -2.0


def test_node_mmr_forces_no_filter_min_score():
    cfg = build_node_search_config('mmr', mmr_lambda=0.5, limit=6, min_score=0.0)
    assert cfg.reranker_min_score == -2.0


def test_edge_rrf_keeps_rrf():
    cfg = build_edge_search_config('rrf', mmr_lambda=0.5, limit=4, min_score=0.0)
    assert cfg.edge_config.reranker == EdgeReranker.rrf
    assert cfg.limit == 4


def test_edge_cross_encoder_sets_min_score():
    cfg = build_edge_search_config('cross_encoder', mmr_lambda=0.5, limit=6, min_score=0.2)
    assert cfg.edge_config.reranker == EdgeReranker.cross_encoder
    assert cfg.reranker_min_score == 0.2


def test_edge_center_node_forces_node_distance():
    cfg = build_edge_search_config(
        'mmr', mmr_lambda=0.5, limit=6, min_score=0.0, center_node_uuid='abc'
    )
    assert cfg.edge_config.reranker == EdgeReranker.node_distance


def test_node_mmr():
    cfg = build_node_search_config('mmr', mmr_lambda=0.5, limit=6, min_score=0.0)
    assert cfg.node_config.reranker == NodeReranker.mmr
    assert cfg.node_config.mmr_lambda == 0.5


def test_does_not_mutate_singleton():
    before = EDGE_HYBRID_SEARCH_RRF.limit
    _ = build_edge_search_config('mmr', mmr_lambda=0.3, limit=999, min_score=0.0)
    assert EDGE_HYBRID_SEARCH_RRF.limit == before
    assert EDGE_HYBRID_SEARCH_RRF.edge_config.reranker == EdgeReranker.rrf


def test_unknown_reranker_raises():
    raised = False
    try:
        build_edge_search_config('bogus', mmr_lambda=0.5, limit=6, min_score=0.0)
    except ValueError:
        raised = True
    assert raised


def test_liveness_filter_from_none():
    sf = apply_liveness_filter(None)
    # expired_at: single AND-group with one IS NULL clause
    assert sf.expired_at is not None
    assert sf.expired_at[0][0].comparison_operator == ComparisonOperator.is_null
    # invalid_at: two OR-groups (IS NULL) OR (> now)
    assert sf.invalid_at is not None
    assert len(sf.invalid_at) == 2
    assert sf.invalid_at[0][0].comparison_operator == ComparisonOperator.is_null
    assert sf.invalid_at[1][0].comparison_operator == ComparisonOperator.greater_than


def test_liveness_filter_preserves_explicit_expired_at():
    explicit = SearchFilters(
        expired_at=[[DateFilter(comparison_operator=ComparisonOperator.is_not_null)]]
    )
    sf = apply_liveness_filter(explicit)
    # caller intent wins: we do NOT overwrite an explicit expired_at
    assert sf.expired_at[0][0].comparison_operator == ComparisonOperator.is_not_null


def test_liveness_filter_preserves_edge_types():
    explicit = SearchFilters(edge_types=['FIXED_BY'])
    sf = apply_liveness_filter(explicit)
    assert sf.edge_types == ['FIXED_BY']
    assert sf.expired_at is not None  # liveness still added
