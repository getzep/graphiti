"""Search-config and filter helpers for the MCP search tools.

Composes graphiti_core primitives without mutating the shared recipe
singletons: every builder returns a deep copy.
"""

from __future__ import annotations

from datetime import datetime, timezone

from graphiti_core.search.search_config import EdgeReranker, NodeReranker, SearchConfig
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    EDGE_HYBRID_SEARCH_NODE_DISTANCE,
    EDGE_HYBRID_SEARCH_RRF,
    NODE_HYBRID_SEARCH_CROSS_ENCODER,
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import (
    ComparisonOperator,
    DateFilter,
    SearchFilters,
)

VALID_RERANKERS = {'rrf', 'mmr', 'cross_encoder'}

# graphiti_core's maximal_marginal_relevance uses -2.0 as its "keep everything"
# sentinel. MMR scores span negatives (the diversity penalty can exceed the
# small query-similarity term), so any min_score >= 0 silently drops most
# candidates. reranker_min_score is only meaningful for the cross-encoder's
# calibrated 0-1 scores; for MMR we always disable filtering and let it reorder.
MMR_NO_FILTER_SCORE = -2.0


def build_edge_search_config(
    reranker: str,
    mmr_lambda: float,
    limit: int,
    min_score: float,
    center_node_uuid: str | None = None,
) -> SearchConfig:
    if reranker not in VALID_RERANKERS:
        raise ValueError(f'unknown reranker: {reranker!r}')

    if center_node_uuid is not None:
        config = EDGE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(deep=True)
    elif reranker == 'cross_encoder':
        config = EDGE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
    elif reranker == 'mmr':
        config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        assert config.edge_config is not None  # RRF recipe always sets edge_config
        config.edge_config.reranker = EdgeReranker.mmr
        config.edge_config.mmr_lambda = mmr_lambda
        min_score = MMR_NO_FILTER_SCORE  # MMR must not filter; it only reorders
    else:  # 'rrf'
        config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)

    config.limit = limit
    config.reranker_min_score = min_score
    return config


def build_node_search_config(
    reranker: str,
    mmr_lambda: float,
    limit: int,
    min_score: float,
    center_node_uuid: str | None = None,
) -> SearchConfig:
    if reranker not in VALID_RERANKERS:
        raise ValueError(f'unknown reranker: {reranker!r}')

    if center_node_uuid is not None:
        config = NODE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(deep=True)
    elif reranker == 'cross_encoder':
        config = NODE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
    elif reranker == 'mmr':
        config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        assert config.node_config is not None  # RRF recipe always sets node_config
        config.node_config.reranker = NodeReranker.mmr
        config.node_config.mmr_lambda = mmr_lambda
        min_score = MMR_NO_FILTER_SCORE  # MMR must not filter; it only reorders
    else:  # 'rrf'
        config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)

    config.limit = limit
    config.reranker_min_score = min_score
    return config


def apply_liveness_filter(search_filter: SearchFilters | None) -> SearchFilters:
    """Return a copy of ``search_filter`` that excludes superseded/expired edges.

    Adds ``expired_at IS NULL`` and ``(invalid_at IS NULL OR invalid_at > now)``
    unless the caller already constrained those fields (explicit caller intent
    wins). Node search is unaffected — only edges carry temporal invalidation.
    """
    sf = search_filter.model_copy(deep=True) if search_filter is not None else SearchFilters()
    now = datetime.now(timezone.utc)

    if sf.expired_at is None:
        sf.expired_at = [[DateFilter(comparison_operator=ComparisonOperator.is_null)]]

    if sf.invalid_at is None:
        sf.invalid_at = [
            [DateFilter(comparison_operator=ComparisonOperator.is_null)],
            [DateFilter(date=now, comparison_operator=ComparisonOperator.greater_than)],
        ]

    return sf
